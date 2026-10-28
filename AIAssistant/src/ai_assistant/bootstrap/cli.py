"""Process entry point: warm the runtime, then serve until terminated.

Extracted from the composition root so ``bootstrap.server`` only builds the
object graph. Everything the entry point touches is read through that module
at *call* time: the composed graph is process-wide state owned there, and the
tests patch ``server.bootstrap_runtime`` / ``server._stop_server_when_requested``
to drive the two failure paths without starting uvicorn.
"""
from __future__ import annotations

import sys
import threading
from typing import Any


def stop_server_when_requested(server: Any, shutdown_requested: threading.Event) -> None:
    """Set ``uvicorn.Server.should_exit`` once the admin route asks to stop.

    ``/admin/shutdown`` used to call ``os._exit``, which skips the lifespan
    ``finally`` (task-store close, checkpoints, flush). The route now only
    sets the event; this watcher performs the graceful stop from the serving
    process. Extracted so the transition is unit-tested.
    """
    shutdown_requested.wait()
    server.should_exit = True


def main() -> None:
    """Process entry point: warm the runtime, then serve until terminated.

    Extracted from the ``__main__`` guard so the two failure paths are unit
    tests rather than coverage exclusions: a runtime that will not load exits
    1 (the Qt host reads the log), and a serve that dies is re-raised after
    being logged.
    """
    import uvicorn

    # Absolute import, not ``from . import server``: the README documents
    # ``python src/ai_assistant/bootstrap/server.py``, where this module has no
    # package context. Importing the package module also keeps the composed
    # graph in the one module object uvicorn serves, instead of a second
    # ``__main__`` copy.
    from ai_assistant.bootstrap import server

    try:
        server.compose()
        server.bootstrap_runtime()
    except Exception:
        server.logger.exception("AI server startup failed")
        sys.exit(1)
    uvicorn_server = uvicorn.Server(uvicorn.Config(
        server.app, host=server.agent_bind_host(), port=server.agent_bind_port(),
        log_config=None, access_log=False,
    ))
    threading.Thread(
        target=server._stop_server_when_requested,
        args=(uvicorn_server, server.runtime_modules.shutdown_requested),
        name="ai-server-shutdown", daemon=True,
    ).start()
    try:
        uvicorn_server.run()
    except BaseException:
        server.logger.exception("Uvicorn terminated during server startup")
        raise


__all__ = ["main", "stop_server_when_requested"]
