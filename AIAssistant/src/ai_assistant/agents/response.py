"""Response mapping and trace lifecycle for the LangGraph runner.

Extracted from ``agents/runner.py`` (ADR 0013 follow-up): the parked-action
response shapes, the terminal response, the task record lifecycle and the
LangSmith trace close all live here.  The runner now reads as
guard → wiring → trace → run → map with no helper indirection.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ..orchestration.state import Step, has_step_type
from .deps import RunDeps
from .ids import generate_action_id

logger = logging.getLogger("ai_assistant.agents.response")


def _persist_pending(deps: RunDeps, save_pending_fn: Callable[[], None] | None) -> None:
    """Flush the pending store, via the injected hook when one is wired."""
    if save_pending_fn:
        save_pending_fn()
    else:
        deps.pending_actions.save()


def _pending_status(state: dict[str, Any]) -> str:
    """Map the parked-action state onto the response status."""
    pending_state = state.get("pending_tool") or {}
    status = ("pending_ui_action" if pending_state.get("ui_ack")
              else "pending_approval" if pending_state
              else "completed")
    return "cancelled" if state.get("cancelled") else status


def _finish_task(deps: RunDeps, session_id: str, state: dict[str, Any]) -> None:
    """Tell the coordinator whether the run is parked waiting or finished."""
    pending_state = state.get("pending_tool") or {}
    if pending_state:
        deps.task_coordinator.update(
            session_id,
            status="waiting_ui_ack" if pending_state.get("ui_ack") else "waiting_approval",
            iterations=state.get("iteration", 0),
        )
        return
    deps.task_coordinator.finish(session_id, success=not state.get("cancelled"),
                                 iterations=state.get("iteration", 0))


def _close_trace(ctx: dict[str, Any], ctx_mgr: Any, pending_status: str,
                 state: dict[str, Any], total_ms: int) -> None:
    """Record the run outcome on the LangSmith trace and close it."""
    ctx["outputs"] = {"status": pending_status, "iterations": state.get("iteration", 0),
                      "duration_ms": total_ms, "step_count": len(state.get("steps", []))}
    ctx_mgr.__exit__(None, None, None)


def _store_pending_action(
    *,
    deps: RunDeps,
    save_pending_fn: Callable[[], None] | None,
    session_id: str,
    task: str,
    state: dict[str, Any],
    temperature: float,
    language: str,
    action_id: str,
    request_id: str,
    is_ui_ack: bool,
    owner: str,
) -> None:
    """Persist the parked action so approval/UI-ack resume survives a restart."""
    pending = state["pending_tool"]
    with deps.pending_lock:
        deps.pending_actions[request_id if is_ui_ack else action_id] = {
            "tool": pending["tool"],
            "params": pending["params"],
            "session_id": session_id,
            "task": task,
            # Snapshot the live state lists: the graph keeps appending to them
            # after this entry is parked, and the store re-serialises whatever it
            # holds on the next save — aliasing made the persisted entry change
            # after the fact (finding 9).
            "messages": list(state["messages"]),
            "steps": list(state["steps"]),
            "iteration": state["iteration"],
            "temperature": temperature,
            "language": language,
            "created_at": time.time(),
            "ui_ack": is_ui_ack,
            # The run's memory namespace travels with the parked action: the
            # resume may happen in another request (or after a restart), so it
            # cannot read the owner off the caller's stack (ADR 0043).
            "owner": owner,
            "approval_scope": pending.get("approval_scope", ""),
            "approval_preview": pending.get("approval_preview"),
        }
        _persist_pending(deps, save_pending_fn)


def _pending_response(
    *,
    deps: RunDeps,
    session_id: str,
    task: str,
    state: dict[str, Any],
    temperature: float,
    language: str,
    request_started: float,
    prior_step_count: int,
    action_id_fn: Callable[[], str],
    owner: str,
    save_pending_fn: Callable[[], None] | None,
) -> dict[str, Any] | None:
    """The two shapes a parked action returns, or ``None`` when nothing is parked."""
    pending = state.get("pending_tool")
    if not pending:
        return None
    steps = state["steps"]
    # A resumed run swaps in a no-op id generator (RESUME_ACTION_ID), but the
    # action parked *here* is new and has no id yet. Returning "" would hand the
    # client an id the approval boundary refuses (an empty action_id is
    # rejected on the way back in) and would key the store on "", stranding the
    # run — so mint one, the same way the application_action request id does.
    action_id = action_id_fn() or generate_action_id()
    is_ui_ack = bool(pending.get("ui_ack"))
    request_id = pending["params"].get("request_id", action_id)
    _store_pending_action(
        deps=deps, save_pending_fn=save_pending_fn, session_id=session_id, task=task,
        state=state, temperature=temperature, language=language,
        action_id=action_id, request_id=request_id, is_ui_ack=is_ui_ack,
        owner=owner,
    )
    total_ms = round((time.monotonic() - request_started) * 1000)
    if is_ui_ack:
        return {
            "status": "pending_ui_action", "session_id": session_id,
            "steps": steps, "request_id": request_id,
            "ui_action": {"request_id": request_id, "action": pending["params"]["action"],
                          "params": pending["params"]},
            "total_ms": total_ms,
        }
    steps.append({
        "type": "pending_approval",
        "action_id": action_id,
        "tool": pending["tool"],
        "params": pending["params"],
        "description": pending["params"].get("description", f"Thực thi {pending['tool']}"),
        "approval_scope": pending.get("approval_scope", ""),
        "preview": pending.get("approval_preview"),
    })
    return {
        "status": "pending_approval", "session_id": session_id,
        "steps": steps, "action_id": action_id,
        "prior_step_count": prior_step_count,
        "total_ms": total_ms,
    }


def _completed_response(*, pending_status: str, session_id: str, state: dict[str, Any],
                        steps: list[Step], prior_step_count: int, request_started: float) -> dict[str, Any]:
    """The terminal response, guaranteeing one ``final_answer`` step exists."""
    if not any(has_step_type(step, "final_answer") for step in steps):
        steps.append({"type": "final_answer", "content": "Agent đã kết thúc mà chưa có kết luận."})
    return {
        "status": pending_status, "session_id": session_id, "steps": steps,
        "prior_step_count": prior_step_count,
        "iterations": state["iteration"],
        "total_ms": round((time.monotonic() - request_started) * 1000),
    }


__all__ = [
    "_close_trace",
    "_completed_response",
    "_finish_task",
    "_pending_response",
    "_pending_status",
    "_store_pending_action",
]
