from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..application.tasks import TaskService
from ..domain.security import DataClassification
from ..domain.tasks import AgentTask


class A2AMessagePart(BaseModel):
    kind: str = "text"
    text: str = Field(min_length=1, max_length=32000)


class A2AMessage(BaseModel):
    role: str = "ROLE_USER"
    parts: list[A2AMessagePart] = Field(min_length=1)
    message_id: str | None = Field(default=None, alias="messageId")
    task_id: str | None = Field(default=None, alias="taskId")
    context_id: str | None = Field(default=None, alias="contextId")

    model_config = {"populate_by_name": True}


class SendTaskParams(BaseModel):
    message: A2AMessage
    capability: str = "supervisor"
    context_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    classification: DataClassification = DataClassification.INTERNAL


class SendMessageConfiguration(BaseModel):
    return_immediately: bool = Field(default=True, alias="returnImmediately")

    model_config = {"populate_by_name": True}


class SendMessageParams(BaseModel):
    message: A2AMessage
    configuration: SendMessageConfiguration = Field(default_factory=SendMessageConfiguration)
    metadata: dict[str, Any] = Field(default_factory=dict)
    capability: str = "supervisor"
    classification: DataClassification = DataClassification.INTERNAL


class ResumeTaskParams(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)


_A2A_STATE = {
    "submitted": "submitted", "working": "working", "input_required": "input-required",
    "completed": "completed", "failed": "failed", "canceled": "canceled", "rejected": "rejected",
}
_APPROVAL_RESUME_EXTENSION = "urn:3d-reconstruction:a2a:approval-resume:v1"


def _text_message(task: AgentTask, text: str) -> dict[str, Any]:
    return {
        "role": "ROLE_AGENT", "messageId": f"status-{task.id}-{int(task.updated_at * 1000)}",
        "contextId": task.context_id, "taskId": task.id, "parts": [{"text": text}],
    }


def _safe_value(value: Any, key: str = "") -> Any:
    """Never expose credentials or internal continuations through A2A artifacts."""
    if key.lower() in {"password", "token", "secret", "api_key", "authorization", "continuation"}:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _safe_value(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    return value


def task_payload(task: AgentTask) -> dict[str, Any]:
    status: dict[str, Any] = {
        "state": _A2A_STATE[task.status],
        "timestamp": datetime.fromtimestamp(task.updated_at, UTC).isoformat().replace("+00:00", "Z"),
    }
    if task.status.value == "input_required" and task.result:
        status["message"] = _text_message(task, str(task.result.get("content", "Additional input is required.")))
    elif task.error:
        status["message"] = _text_message(task, task.error)
    return {
        "id": task.id,
        "contextId": task.context_id,
        "status": status,
        "metadata": _safe_value({key: value for key, value in task.metadata.items() if not key.startswith("agent_run.")}),
        "artifacts": [{"artifactId": f"result-{task.id}", "name": "result", "parts": [{"data": _safe_value(task.result)}]}]
        if task.result and task.status.value == "completed" else [],
    }


def build_a2a_router(service: TaskService, agent_name: str, agent_version: str) -> APIRouter:
    router = APIRouter(tags=["a2a"])
    skills = [
        {"id": capability, "name": capability.replace("_", " ").title(),
         "description": f"Execute {capability} tasks through the 3D-Reconstruction agent platform."}
        for capability in sorted(service.capabilities)
    ]

    def require_a2a_version(request: Request) -> None:
        requested = request.headers.get("A2A-Version")
        if requested and requested != "0.3":
            raise HTTPException(
                status_code=400,
                detail={"type": "https://a2a-protocol.org/errors/version-not-supported",
                        "supportedVersions": ["0.3"]},
            )

    def submit_message(params: SendMessageParams) -> AgentTask:
        message = "\n".join(part.text for part in params.message.parts if part.kind == "text").strip()
        if params.message.task_id:
            existing = service.get(params.message.task_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="A2A task not found")
            if params.message.context_id and params.message.context_id != existing.context_id:
                raise HTTPException(status_code=409, detail="Task context does not match message context")
            raise HTTPException(
                status_code=409,
                detail={"message": "Approval and desktop acknowledgements require the declared resume extension.",
                        "extension": _APPROVAL_RESUME_EXTENSION},
            )
        return service.submit(params.capability, message, params.metadata, params.message.context_id, params.classification)

    @router.get("/.well-known/agent.json")
    def agent_card(request: Request) -> dict[str, Any]:
        base_url = str(request.base_url).rstrip("/")
        return {
            "name": agent_name,
            "description": "3D-Reconstruction AI Agent Platform",
            "version": agent_version,
            "protocolVersion": "0.3",
            "url": base_url + "/a2a",
            "supportedInterfaces": [{
                "url": base_url, "protocolBinding": "HTTP+JSON/REST", "protocolVersion": "0.3",
            }],
            "capabilities": {"streaming": True, "pushNotifications": False},
            "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"], "skills": skills,
            "extensions": [_APPROVAL_RESUME_EXTENSION],
        }

    @router.post("/message:send")
    def send_message(params: SendMessageParams, request: Request) -> dict[str, Any]:
        require_a2a_version(request)
        return {"task": task_payload(submit_message(params))}

    @router.get("/tasks/{task_id}")
    def standard_get_task(task_id: str, request: Request) -> dict[str, Any]:
        require_a2a_version(request)
        task = service.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"task": task_payload(task)}

    @router.get("/tasks")
    def standard_list_tasks(request: Request, contextId: str | None = None, limit: int = 100) -> dict[str, Any]:
        require_a2a_version(request)
        return {"tasks": [task_payload(task) for task in service.list(context_id=contextId, limit=limit)]}

    @router.post("/tasks/{task_id}:cancel")
    def standard_cancel_task(task_id: str, request: Request) -> dict[str, Any]:
        require_a2a_version(request)
        task = service.cancel(task_id)
        if task is None:
            raise HTTPException(status_code=409, detail="A2A task cannot be cancelled")
        return {"task": task_payload(task)}

    @router.post("/a2a/tasks/send")
    def send_task(params: SendTaskParams) -> dict[str, Any]:
        message = "\n".join(part.text for part in params.message.parts if part.kind == "text").strip()
        task = service.submit(params.capability, message, params.metadata, params.context_id, params.classification)
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        task = service.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="A2A task not found")
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks")
    def list_tasks(context_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        return {"tasks": [task_payload(task) for task in service.list(context_id=context_id, limit=limit)]}

    @router.post("/a2a/tasks/{task_id}:cancel")
    def cancel_task(task_id: str) -> dict[str, Any]:
        task = service.cancel(task_id)
        if task is None:
            raise HTTPException(status_code=409, detail="A2A task cannot be cancelled")
        return {"task": task_payload(task)}

    @router.post("/a2a/tasks/{task_id}:resume")
    def resume_task(task_id: str, params: ResumeTaskParams) -> dict[str, Any]:
        task = service.resume(task_id, params.input)
        if task is None:
            raise HTTPException(status_code=409, detail="A2A task cannot be resumed")
        return {"task": task_payload(task)}

    @router.post("/tasks/{task_id}:resume")
    def standard_resume_task(task_id: str, params: ResumeTaskParams, request: Request) -> dict[str, Any]:
        require_a2a_version(request)
        task = service.resume(task_id, params.input)
        if task is None:
            raise HTTPException(status_code=409, detail="A2A task cannot be resumed")
        return {"task": task_payload(task)}

    @router.get("/a2a/tasks/{task_id}/events")
    def stream_task(task_id: str):
        if service.get(task_id) is None:
            raise HTTPException(status_code=404, detail="A2A task not found")

        async def stream():
            cursor = 0
            while True:
                events = list(service.events(task_id))
                for event in events[cursor:]:
                    yield f"event: {event['kind']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                cursor = len(events)
                task = service.get(task_id)
                if task and task.status.value in {"completed", "failed", "canceled", "rejected"}:
                    yield f"event: done\ndata: {json.dumps(task_payload(task), ensure_ascii=False)}\n\n"
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.post("/tasks/{task_id}:subscribe")
    def subscribe_task(task_id: str, request: Request):
        require_a2a_version(request)
        if service.get(task_id) is None:
            raise HTTPException(status_code=404, detail="A2A task not found")

        async def stream():
            previous = None
            while True:
                task = service.get(task_id)
                if task is None:
                    return
                snapshot = task_payload(task)
                if previous is None:
                    yield f"data: {json.dumps({'task': snapshot}, ensure_ascii=False)}\n\n"
                elif snapshot["status"] != previous["status"]:
                    yield f"data: {json.dumps({'statusUpdate': {'taskId': task.id, 'status': snapshot['status']}}, ensure_ascii=False)}\n\n"
                previous = snapshot
                if task.status.value in {"completed", "failed", "canceled", "rejected", "input_required"}:
                    return
                await asyncio.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @router.post("/a2a")
    async def json_rpc(request: Request) -> dict[str, Any]:
        payload = await request.json()
        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})
        try:
            if method in {"SendMessage", "message/send"}:
                typed = SendMessageParams.model_validate(params)
                result = {"task": task_payload(submit_message(typed))}
            elif method == "tasks/send":
                typed = SendTaskParams.model_validate(params)
                message = "\n".join(part.text for part in typed.message.parts if part.kind == "text").strip()
                task = service.submit(typed.capability, message, typed.metadata, typed.context_id, typed.classification)
                result = {"task": task_payload(task)}
            elif method == "tasks/get":
                task = service.get(str(params.get("id", "")))
                if task is None:
                    raise KeyError("A2A task not found")
                result = {"task": task_payload(task)}
            elif method == "tasks/list":
                result = {
                    "tasks": [
                        task_payload(task)
                        for task in service.list(context_id=params.get("context_id"), limit=int(params.get("limit", 100)))
                    ]
                }
            elif method == "tasks/cancel":
                task = service.cancel(str(params.get("id", "")))
                if task is None:
                    raise ValueError("A2A task cannot be cancelled")
                result = {"task": task_payload(task)}
            elif method == "tasks/resume":
                task = service.resume(str(params.get("id", "")), dict(params.get("input", {})))
                if task is None:
                    raise ValueError("A2A task cannot be resumed")
                result = {"task": task_payload(task)}
            else:
                raise NotImplementedError(f"Unsupported A2A method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, ValueError, NotImplementedError) as error:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": str(error)}}

    return router
