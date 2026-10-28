"""FastAPI lifespan: startup discovery/pruning and the shutdown close order.

Kept out of the composition root (``bootstrap.server``) so that module stays
the object-graph builder. The process-wide state — the composed ``platform``,
the MCP server, the sealed memory store, the logger — is owned by
``bootstrap.server`` and is read through that module at *call* time. That is
deliberate: the state is the composition root's single instance, and a test
that patches ``server.platform`` / ``server.memory_store`` (or the shutdown
helpers) must drive the real lifespan, not a stale local binding.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

if TYPE_CHECKING:
    from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run the startup jobs, then hold the server open until shutdown.

    Startup work is best-effort (checkpoint cleanup, A2A discovery, task
    retention) and a failure is logged instead of refusing to serve: none of
    it is required for the request path. The shutdown path is the opposite —
    it must release every handle, and one failing close must not skip the
    next one.
    """
    from . import server

    logger = server.logger
    try:
        from ai_assistant.adapters.persistence.checkpointing import cleanup_old_checkpoints
        cleanup_old_checkpoints()
    except Exception as e:
        logger.warning(f"Lỗi dọn dẹp checkpoint: {e}")

    registry = getattr(getattr(app, "state", None), "a2a_registry", None)
    if registry is not None:
        try:
            await server._discover_a2a_peers(registry)
        except Exception as e:
            logger.warning(f"A2A discovery failed: {e}")

    platform = server.platform
    try:
        retention_days = server.task_retention_days()
        if retention_days > 0:
            removed = platform.tasks.prune(retention_days)
            if removed:
                logger.info(
                    "Pruned %d task(s) past the %d-day retention window",
                    removed, retention_days,
                )
    except Exception as e:
        logger.warning(f"Task retention pruning failed: {e}")

    total = time.monotonic() - server._SERVER_START_TIME
    logger.info("Server ready in %.1fs — http://%s:%d", total,
                server.agent_bind_host(), server.agent_bind_port())
    print(f"[SUCCESS] AI Server started successfully ({total:.1f}s)", flush=True)
    try:
        if not platform.settings.enable_mcp:
            # MCP disabled by deployment profile: still yield, otherwise the
            # asynccontextmanager would raise "generator didn't yield" and the
            # server could not start with enable_mcp = false.
            logger.info("MCP endpoint disabled by configuration")
            yield
        elif server.mcp_server.available:
            async with server.mcp_server.lifespan():
                yield
        else:
            logger.warning("MCP SDK is not installed; MCP endpoint is unavailable")
            yield
    finally:
        # Close both stores independently: one failing close must not skip
        # the other handle. The previous shape let a raising memory-store
        # close leave the task store open for the rest of the process — the
        # opposite of what a shutdown is for (ADR 0043).
        memory_store = server.memory_store
        if memory_store is not None:
            try:
                memory_store.close()
            except Exception as error:
                logger.warning(f"Closing the long-term memory store failed: {error}")
        try:
            platform.tasks.close()
        except Exception as error:
            logger.warning(f"Closing the task store failed: {error}")
        logger.info("Server shutdown after %.1fs", time.monotonic() - server._SERVER_START_TIME)


__all__ = ["lifespan"]
