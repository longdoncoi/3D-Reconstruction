"""HTTP coverage for the A2A 0.3 router and its legacy compatibility surface.

``adapters/a2a_router.py`` builds every A2A endpoint (ADR 0005): the agent
card, the 0.3 task routes that require the declared ``A2A-Version`` header,
the pre-0.3 ``/a2a/tasks/*`` endpoints kept for existing clients, the two SSE
streams and the JSON-RPC envelope. Route registration is grouped by surface
(ADR 0014), and each group closes over the injected ``TaskService`` — so these
tests drive a real FastAPI app rather than calling the helpers directly, and
every closure body is reached end to end.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters import a2a_protocol
from ai_assistant.adapters.a2a_protocol import A2ARouter, RemoteAgent, a2a_available
from ai_assistant.adapters.a2a_router import _task_pages, build_a2a_router
from ai_assistant.adapters.http.security import TransportPolicy
from ai_assistant.domain.security import DataClassification
from ai_assistant.domain.tasks import AgentTask, TaskStatus

_COMPLETE = TaskStatus.COMPLETED


def _task(task_id: str, *, status: TaskStatus = _COMPLETE,
          context_id: str | None = None) -> AgentTask:
    return AgentTask(
        id=task_id,
        capability="supervisor",
        message="hello",
        context_id=context_id,
        metadata={},
        classification=DataClassification.INTERNAL,
        status=status,
    )


def _state(payload: dict) -> str:
    """``task_payload`` nests the state under ``status.state`` (A2A 0.3)."""
    return payload["status"]["state"]


class _FakeService:
    """Duck-typed ``TaskService`` over an in-memory dict.

    ``cancellable``/``resumable`` are switches so the 409 branches are
    reachable without a second implementation, and ``AgentTask`` is frozen
    (ADR 0010), so state advances by replacement rather than mutation.
    """

    def __init__(self) -> None:
        self.store: dict[str, AgentTask] = {}
        self.event_log: dict[str, list[dict]] = {}
        self.submitted: list[tuple] = []
        self.list_limits: list[int] = []
        self.capabilities = frozenset({"supervisor", "code"})
        self.cancellable = True
        self.resumable = True
        self.on_poll = None  # optional per-poll hook, used by the SSE tests

    def add(self, task: AgentTask, events: list[dict] | None = None) -> AgentTask:
        self.store[task.id] = task
        self.event_log[task.id] = list(events or [])
        return task

    def submit(self, capability, message, metadata=None, context_id=None, classification=None,
               *, owner="default"):
        self.submitted.append((capability, message, context_id))
        task = replace(_task(f"t-{len(self.submitted)}", context_id=context_id),
                       status=TaskStatus.SUBMITTED, owner=owner)
        self.store[task.id] = task
        return task

    def get(self, task_id, *, owner="default"):
        task = self.store.get(task_id)
        if task is None or task.owner != owner:
            return None
        return task

    def events(self, task_id, since=0, *, owner="default"):
        if self.get(task_id, owner=owner) is None:
            return ()
        if self.on_poll is not None:
            self.on_poll()
        return tuple(self.event_log.get(task_id, ()))[since:]

    def list(self, *, context_id=None, limit=100, owner="default"):
        self.list_limits.append(limit)
        tasks = [
            task for task in self.store.values()
            if task.owner == owner and (context_id is None or task.context_id == context_id)
        ]
        return tuple(tasks[:limit])

    def cancel(self, task_id, *, owner="default"):
        task = self.get(task_id, owner=owner)
        if task is None or not self.cancellable:
            return None
        self.store[task_id] = replace(task, status=TaskStatus.CANCELED)
        return self.store[task_id]

    def resume(self, task_id, input_data, *, owner="default"):
        task = self.get(task_id, owner=owner)
        if task is None or not self.resumable:
            return None
        self.store[task_id] = replace(task, status=TaskStatus.WORKING)
        return self.store[task_id]


def _client(service: _FakeService, policy=None, max_subscribers=None,
            card_base_url: str | None = None) -> TestClient:
    """Build the app and call it as a loopback peer.

    The A2A surface is loopback-only until remote access is opted into with a
    token (ADR 0008), so the client must present a loopback address —
    Starlette's default ``testclient`` peer is refused with a 403 before any
    route handler would run.
    """
    app = FastAPI()
    kwargs = {"policy": policy, "max_subscribers": max_subscribers}
    if card_base_url is not None:
        kwargs["card_base_url"] = card_base_url
    app.include_router(build_a2a_router(service, "assistant", "3.0", **kwargs))
    return TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False)


def _send_body(text: str, *, task_id: str | None = None,
               message_context_id: str | None = None, **extra) -> dict:
    """A ``SendMessageParams`` body.

    ``taskId``/``contextId`` live on the *message*, while a bare ``context_id``
    in ``extra`` is the top-level ``SendTaskParams`` field — the two surfaces
    genuinely differ.
    """
    message: dict = {"role": "ROLE_USER", "parts": [{"kind": "text", "text": text}]}
    if task_id is not None:
        message["taskId"] = task_id
    if message_context_id is not None:
        message["contextId"] = message_context_id
    body: dict = {"message": message, "capability": "supervisor"}
    body.update(extra)
    return body


class AgentCardAndSendRoutesTests(unittest.TestCase):
    def test_agent_card_advertises_the_endpoint_and_capabilities(self):
        with _client(_FakeService()) as client:
            card = client.get("/.well-known/agent.json").json()
        self.assertEqual(card["url"], "http://127.0.0.1:8080/a2a")
        self.assertEqual(card["name"], "assistant")
        self.assertEqual(card["version"], "3.0")
        self.assertEqual(card["protocolVersion"], "0.3")
        self.assertEqual(sorted(s["id"] for s in card["skills"]), ["code", "supervisor"])

    def test_the_advertised_url_is_the_configured_one(self):
        """ADR 0032: the card publishes the *configured* base URL."""
        with _client(_FakeService(), card_base_url="https://ai.internal:8443/") as client:
            card = client.get("/.well-known/agent.json").json()
        self.assertEqual(card["url"], "https://ai.internal:8443/a2a")
        self.assertEqual(
            card["supportedInterfaces"][0]["url"], "https://ai.internal:8443"
        )

    def test_a_spoofed_host_header_cannot_rewrite_the_advertised_url(self):
        """ADR 0032: the card used to be built from ``request.base_url``, which
        is derived from the Host header. A peer sending ``Host: evil.example``
        made this server publish a card pointing at the attacker — and a peer
        that trusts the card dispatches its tasks there."""
        with _client(_FakeService(), card_base_url="https://ai.internal:8443") as client:
            card = client.get(
                "/.well-known/agent.json", headers={"Host": "evil.example"}
            ).json()
        self.assertEqual(card["url"], "https://ai.internal:8443/a2a")
        self.assertNotIn("evil.example", json.dumps(card))

    def test_message_send_submits_and_returns_the_task(self):
        service = _FakeService()
        with _client(service) as client:
            response = client.post("/message:send", json=_send_body("xin chào"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(_state(response.json()["task"]), "submitted")
        self.assertEqual(service.submitted[0][1], "xin chào")

    def test_message_send_joins_multiple_text_parts(self):
        service = _FakeService()
        body = _send_body("part one")
        body["message"]["parts"].append({"kind": "text", "text": "part two"})
        with _client(service) as client:
            client.post("/message:send", json=body)
        self.assertEqual(service.submitted[0][1], "part one\npart two")

    def test_legacy_send_task_ignores_the_version_header(self):
        """The pre-0.3 endpoint keeps working without an ``A2A-Version`` header."""
        service = _FakeService()
        with _client(service) as client:
            response = client.post("/a2a/tasks/send", json=_send_body("legacy", context_id="c1"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.submitted[0], ("supervisor", "legacy", "c1"))


class VersionGuardTests(unittest.TestCase):
    def test_wrong_version_is_rejected_with_the_supported_list(self):
        with _client(_FakeService()) as client:
            response = client.get("/tasks/t1", headers={"A2A-Version": "0.2"})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["supportedVersions"], ["0.3"])
        self.assertIn("version-not-supported", detail["type"])

    def test_absent_header_is_accepted(self):
        service = _FakeService()
        service.add(_task("t1"))
        with _client(service) as client:
            self.assertEqual(client.get("/tasks/t1").status_code, 200)

    def test_declared_version_is_accepted(self):
        service = _FakeService()
        service.add(_task("t1"))
        with _client(service) as client:
            response = client.get("/tasks/t1", headers={"A2A-Version": "0.3"})
        self.assertEqual(response.status_code, 200)


class StandardTaskRouteTests(unittest.TestCase):
    def test_get_task_round_trips(self):
        service = _FakeService()
        service.add(_task("t1"))
        with _client(service) as client:
            self.assertEqual(client.get("/tasks/t1").json()["task"]["id"], "t1")

    def test_get_unknown_task_is_404(self):
        with _client(_FakeService()) as client:
            self.assertEqual(client.get("/tasks/missing").status_code, 404)

    def test_list_tasks_filters_by_context(self):
        service = _FakeService()
        service.add(_task("a", context_id="ctx-a"))
        service.add(_task("b", context_id="ctx-b"))
        with _client(service) as client:
            listed = client.get("/tasks", params={"contextId": "ctx-a"}).json()["tasks"]
        self.assertEqual([t["id"] for t in listed], ["a"])

    def test_list_tasks_honours_the_limit(self):
        service = _FakeService()
        service.add(_task("a"))
        service.add(_task("b"))
        with _client(service) as client:
            self.assertEqual(len(client.get("/tasks", params={"limit": 1}).json()["tasks"]), 1)

    def test_cancel_task(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))
        with _client(service) as client:
            cancelled = client.post("/tasks/t1:cancel").json()["task"]
        self.assertEqual(_state(cancelled), "canceled")

    def test_cancel_refuses_a_task_that_cannot_be_cancelled(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.cancellable = False
        with _client(service) as client:
            response = client.post("/tasks/t1:cancel")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "A2A task cannot be cancelled")

    def test_resume_task(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.INPUT_REQUIRED))
        with _client(service) as client:
            resumed = client.post("/tasks/t1:resume", json={"input": {"answer": "yes"}}).json()["task"]
        self.assertEqual(_state(resumed), "working")

    def test_resume_refuses_a_task_that_cannot_be_resumed(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.resumable = False
        with _client(service) as client:
            response = client.post("/tasks/t1:resume", json={"input": {}})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "A2A task cannot be resumed")


class SubscribeStreamTests(unittest.TestCase):
    def test_subscribe_unknown_task_is_404(self):
        with _client(_FakeService()) as client:
            self.assertEqual(client.post("/tasks/missing:subscribe").status_code, 404)

    @staticmethod
    def _script(service: _FakeService, statuses: list[TaskStatus]) -> None:
        """Advance the stored task through ``statuses`` once per poll.

        ``events`` runs before ``get`` inside the polling loop, so the hook
        fires on the poll that will read the next state.
        """
        remaining = list(statuses)
        service.on_poll = lambda: service.store.__setitem__(
            "t1",
            replace(service.store["t1"], status=remaining.pop(0) if remaining else _COMPLETE),
        )

    def test_subscribe_emits_the_snapshot_then_status_updates(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))
        # Poll 1 snapshots, poll 2 reports the terminal state and ends the stream.
        self._script(service, [TaskStatus.WORKING, _COMPLETE])

        with _client(service) as client:
            body = client.post("/tasks/t1:subscribe").text

        first, second = body.split("\n\n")[:2]
        self.assertIn('"task"', first)
        self.assertIn('"statusUpdate"', second)
        self.assertIn('"completed"', second)

    def test_subscribe_stays_silent_when_the_status_is_unchanged(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))
        # Poll 2 repeats the status (no frame); poll 3 finishes the stream.
        self._script(service, [TaskStatus.WORKING, TaskStatus.WORKING, _COMPLETE])

        with _client(service) as client:
            body = client.post("/tasks/t1:subscribe").text

        # Two frames despite three polls: the unchanged one was suppressed.
        self.assertEqual(body.count("data:"), 2)
        self.assertEqual(body.count('"task"'), 1)


class DispatchClassificationTests(unittest.TestCase):
    """The production delegation shape must survive the egress classifier."""

    def test_session_and_idempotency_keys_stay_internal(self) -> None:
        from types import SimpleNamespace

        client = MagicMock()
        client.submit.return_value = {"routing": "remote_selected"}
        router = A2ARouter(client=client)
        payload = {
            "tool": "read_file",
            "parameters": {"path": "notes.txt"},
            "session_id": "s1",
            "idempotency_key": "k1",
        }
        router._dispatch(
            SimpleNamespace(url="https://peer.example"), "research", "read notes", payload,
        )
        kwargs = client.submit.call_args.kwargs
        self.assertEqual(kwargs["classification"], DataClassification.INTERNAL)
        self.assertEqual(kwargs["metadata"], payload)


class SubscribeDeadlineTests(unittest.TestCase):
    """A non-terminal task must not keep a subscriber forever."""

    def test_the_deadline_ends_a_stream_that_never_reaches_a_terminal_state(self) -> None:
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))

        async def collect() -> list[tuple[list[dict], AgentTask]]:
            pages = []
            async for page in _task_pages(
                service, "t1", terminal=frozenset({"completed"}), deadline_seconds=0.0,
            ):
                pages.append(page)
            return pages

        pages = asyncio.run(collect())
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][1].status, TaskStatus.WORKING)

    def test_a_service_without_a_cursor_falls_back_to_count_slicing(self) -> None:
        class _CursorlessService:
            def __init__(self) -> None:
                self.task = _task("t1", status=TaskStatus.WORKING)

            def events(self, task_id: str, *, owner: str = "default") -> tuple[dict, ...]:
                return ({"kind": "progress", "data": {"step": 1}},)

            def get(self, task_id: str, *, owner: str = "default") -> AgentTask | None:
                return self.task if task_id == "t1" else None

        service = _CursorlessService()

        async def collect() -> list[tuple[list[dict], AgentTask]]:
            return [
                page async for page in _task_pages(
                    service, "t1", terminal=frozenset({"completed"}), deadline_seconds=0.0,
                )
            ]

        pages = asyncio.run(collect())
        self.assertEqual(len(pages), 1)
        self.assertEqual([event["kind"] for event in pages[0][0]], ["progress"])


class SubscriberAdmissionTests(unittest.TestCase):
    """A flood of open streams is refused before it exhausts the worker."""

    def test_the_limiter_counts_and_releases_slots(self) -> None:
        from ai_assistant.adapters.a2a_router import _SubscriberLimiter

        limiter = _SubscriberLimiter(1)
        self.assertTrue(limiter.try_acquire())
        self.assertFalse(limiter.try_acquire())
        self.assertEqual(limiter.active, 1)
        limiter.release()
        self.assertEqual(limiter.active, 0)
        limiter.release()  # an extra close must not underflow

    def test_the_cap_refuses_extra_streams_and_releases_on_completion(self) -> None:
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))

        app = FastAPI()
        router = build_a2a_router(service, "assistant", "3.0", max_subscribers=1)
        app.include_router(router)
        limiter = getattr(router, "_subscriber_limiter")

        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            self.assertTrue(limiter.try_acquire())
            legacy = client.get("/a2a/tasks/t1/events")
            self.assertEqual(legacy.status_code, 503)
            self.assertEqual(legacy.headers["retry-after"], "5")
            standard = client.post("/tasks/t1:subscribe", headers={"A2A-Version": "0.3"})
            self.assertEqual(standard.status_code, 503)

            limiter.release()
            service.store["t1"] = replace(service.store["t1"], status=_COMPLETE)
            finished = client.get("/a2a/tasks/t1/events")
            self.assertEqual(finished.status_code, 200)
            self.assertEqual(limiter.active, 0)


class LegacyRouteTests(unittest.TestCase):
    def test_get_task(self):
        service = _FakeService()
        service.add(_task("t1"))
        with _client(service) as client:
            self.assertEqual(client.get("/a2a/tasks/t1").json()["task"]["id"], "t1")

    def test_get_unknown_task_is_404(self):
        with _client(_FakeService()) as client:
            self.assertEqual(client.get("/a2a/tasks/missing").status_code, 404)

    def test_list_tasks(self):
        service = _FakeService()
        service.add(_task("a"))
        service.add(_task("b"))
        with _client(service) as client:
            self.assertEqual(len(client.get("/a2a/tasks").json()["tasks"]), 2)

    def test_cancel_task(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))
        with _client(service) as client:
            cancelled = client.post("/a2a/tasks/t1:cancel").json()["task"]
        self.assertEqual(_state(cancelled), "canceled")

    def test_cancel_refuses_when_not_cancellable(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.cancellable = False
        with _client(service) as client:
            self.assertEqual(client.post("/a2a/tasks/t1:cancel").status_code, 409)

    def test_resume_task(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.INPUT_REQUIRED))
        with _client(service) as client:
            resumed = client.post("/a2a/tasks/t1:resume", json={"input": {"answer": "no"}}).json()["task"]
        self.assertEqual(_state(resumed), "working")

    def test_resume_refuses_when_not_resumable(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.resumable = False
        with _client(service) as client:
            self.assertEqual(client.post("/a2a/tasks/t1:resume", json={}).status_code, 409)

    def test_events_stream_replays_the_log_then_finishes(self):
        service = _FakeService()
        service.add(
            _task("t1"),
            events=[{"kind": "progress", "step": 1}, {"kind": "progress", "step": 2}],
        )
        with _client(service) as client:
            body = client.get("/a2a/tasks/t1/events").text
        self.assertEqual(body.count("event: progress"), 2)
        self.assertEqual(body.count("event: done"), 1)

    def test_events_stream_for_an_unknown_task_is_404(self):
        with _client(_FakeService()) as client:
            self.assertEqual(client.get("/a2a/tasks/missing/events").status_code, 404)

    def test_events_stream_polls_until_the_task_is_terminal(self):
        """A live task replays each new batch, then the terminal marker."""
        service = _FakeService()
        service.add(
            _task("t1", status=TaskStatus.WORKING),
            events=[{"kind": "progress", "step": 1}],
        )

        def _finish():
            polls["n"] += 1
            if polls["n"] >= 2:
                service.event_log["t1"].append({"kind": "progress", "step": 2})
                service.store["t1"] = replace(service.store["t1"], status=_COMPLETE)

        polls = {"n": 0}
        service.on_poll = _finish

        with _client(service) as client:
            body = client.get("/a2a/tasks/t1/events").text

        # Poll 1 emits step 1; poll 2 emits only the new step 2, then done.
        self.assertEqual(body.count("event: progress"), 2)
        self.assertEqual(body.count("event: done"), 1)

    def test_events_stream_stops_when_the_task_vanishes(self):
        """A task removed mid-poll ends the stream without a terminal marker."""
        service = _FakeService()
        service.add(
            _task("t1", status=TaskStatus.WORKING),
            events=[{"kind": "progress", "step": 1}],
        )

        polls = {"n": 0}

        def _vanish():
            polls["n"] += 1
            if polls["n"] >= 2:
                service.store.pop("t1", None)

        service.on_poll = _vanish

        with _client(service) as client:
            body = client.get("/a2a/tasks/t1/events").text

        self.assertEqual(body.count("event: progress"), 1)
        self.assertNotIn("event: done", body)


class SubmitMessageGuardTests(unittest.TestCase):
    """In-band replies are refused: approvals are out-of-band capabilities."""

    def test_reply_on_an_unknown_task_is_404(self):
        body = _send_body("reply", task_id="missing")
        with _client(_FakeService()) as client:
            self.assertEqual(client.post("/message:send", json=body).status_code, 404)

    def test_reply_with_a_mismatched_context_is_409(self):
        service = _FakeService()
        service.add(_task("t1", context_id="ctx-1"))
        body = _send_body("reply", task_id="t1", message_context_id="ctx-2")
        with _client(service) as client:
            response = client.post("/message:send", json=body)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"], "Task context does not match message context")

    def test_reply_declares_the_resume_extension(self):
        service = _FakeService()
        service.add(_task("t1", context_id="ctx-1"))
        body = _send_body("reply", task_id="t1", message_context_id="ctx-1")
        with _client(service) as client:
            response = client.post("/message:send", json=body)
        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertIn("resume extension", detail["message"])
        self.assertTrue(detail["extension"])


class JsonRpcEnvelopeTests(unittest.TestCase):
    def _post(self, method: str, params: dict, service: _FakeService | None = None):
        with _client(service or _FakeService()) as client:
            return client.post(
                "/a2a", json={"jsonrpc": "2.0", "id": 7, "method": method, "params": params},
            )

    def test_message_send_aliases(self):
        for method in ("SendMessage", "message/send"):
            with self.subTest(method=method):
                service = _FakeService()
                result = self._post(method, _send_body("hi"), service).json()["result"]
                self.assertEqual(_state(result["task"]), "submitted")

    def test_tasks_send(self):
        service = _FakeService()
        result = self._post("tasks/send", _send_body("legacy path", context_id="c9"), service).json()["result"]
        self.assertEqual(_state(result["task"]), "submitted")
        self.assertEqual(service.submitted[0], ("supervisor", "legacy path", "c9"))

    def test_tasks_get(self):
        service = _FakeService()
        service.add(_task("t1"))
        result = self._post("tasks/get", {"id": "t1"}, service).json()["result"]
        self.assertEqual(result["task"]["id"], "t1")

    def test_tasks_get_unknown_becomes_a_json_rpc_error(self):
        error = self._post("tasks/get", {"id": "missing"}).json()["error"]
        self.assertEqual(error["code"], -32602)
        self.assertIn("A2A task not found", error["message"])

    def test_tasks_list(self):
        service = _FakeService()
        service.add(_task("a"))
        service.add(_task("b"))
        result = self._post("tasks/list", {"limit": 1}, service).json()["result"]
        self.assertEqual(len(result["tasks"]), 1)

    def test_tasks_cancel(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.WORKING))
        result = self._post("tasks/cancel", {"id": "t1"}, service).json()["result"]
        self.assertEqual(_state(result["task"]), "canceled")

    def test_tasks_cancel_refusal_is_a_json_rpc_error(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.cancellable = False
        error = self._post("tasks/cancel", {"id": "t1"}, service).json()["error"]
        self.assertIn("cannot be cancelled", error["message"])

    def test_tasks_resume(self):
        service = _FakeService()
        service.add(_task("t1", status=TaskStatus.INPUT_REQUIRED))
        result = self._post("tasks/resume", {"id": "t1", "input": {"a": 1}}, service).json()["result"]
        self.assertEqual(_state(result["task"]), "working")

    def test_tasks_resume_refusal_is_a_json_rpc_error(self):
        service = _FakeService()
        service.add(_task("t1"))
        service.resumable = False
        error = self._post("tasks/resume", {"id": "t1"}, service).json()["error"]
        self.assertIn("cannot be resumed", error["message"])

    def test_unknown_method_is_reported_without_failing_the_envelope(self):
        response = self._post("tasks/explode", {})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], 7)
        self.assertIn("Unsupported A2A method", response.json()["error"]["message"])



# ── Protocol router facade (moved from test_coverage_boost4) ───

class A2ARouterFacadeTests(unittest.TestCase):
    """Coverage for the A2ARouter facade."""

    class _FakeClient:
        def __init__(self, outcome) -> None:
            self.outcome = outcome

        def submit(self, *args, **kwargs):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    def _remote(self) -> RemoteAgent:
        return RemoteAgent(url="https://r.example.com", card={}, skills={"supervisor": {}})

    def _registry(self) -> a2a_protocol.RemoteAgentRegistry:
        registry = a2a_protocol.RemoteAgentRegistry()
        registry._agents["https://r.example.com"] = self._remote()
        return registry

    def test_a2a_available_reflects_sdk_and_setting(self) -> None:
        with patch.object(a2a_protocol, "a2a_enabled", return_value=True), patch.object(
            a2a_protocol, "_sdk_available", return_value=True
        ):
            self.assertTrue(a2a_available())
        with patch.object(a2a_protocol, "a2a_enabled", return_value=False), patch.object(
            a2a_protocol, "_sdk_available", return_value=True
        ):
            self.assertFalse(a2a_available())

    def test_route_unavailable_when_transport_disabled(self) -> None:
        with patch.object(a2a_protocol, "a2a_available", return_value=False):
            result = A2ARouter(registry=self._registry()).route("supervisor", "hi")
        self.assertEqual(result["status"], "unavailable")

    def test_route_unavailable_when_no_remote(self) -> None:
        with patch.object(a2a_protocol, "a2a_available", return_value=True):
            result = A2ARouter(registry=a2a_protocol.RemoteAgentRegistry()).route("supervisor", "hi")
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("No trusted remote agent", result["error"])

    def test_route_success(self) -> None:
        client = self._FakeClient({"routing": "remote_selected", "task": {"id": "t1"}})
        with patch.object(a2a_protocol, "a2a_available", return_value=True):
            result = A2ARouter(client=client, registry=self._registry()).route(
                "supervisor", "hi", {"k": "v"},
            )
        self.assertEqual(result["source"], "remote")
        self.assertEqual(result["task"]["id"], "t1")

    def test_route_rejected_dispatch_becomes_failure(self) -> None:
        client = self._FakeClient({"routing": "rejected_policy", "error": "not allowed"})
        with patch.object(a2a_protocol, "a2a_available", return_value=True):
            result = A2ARouter(client=client, registry=self._registry()).route("supervisor", "hi")
        self.assertEqual(result["status"], "failed")
        self.assertIn("not allowed", result["error"])

    def test_route_transport_exception_becomes_failure(self) -> None:
        client = self._FakeClient(RuntimeError("socket closed"))
        with patch.object(a2a_protocol, "a2a_available", return_value=True):
            result = A2ARouter(client=client, registry=self._registry()).route("supervisor", "hi")
        self.assertEqual(result["status"], "failed")
        self.assertIn("socket closed", result["error"])

    def test_find_remote_none_when_skill_absent(self) -> None:
        self.assertIsNone(A2ARouter(registry=self._registry()).find_remote("nonexistent"))

class TaskListLimitClampTests(unittest.TestCase):
    """``limit`` is a caller value: the router clamps it, never trusts it.

    SQLite reads ``LIMIT -1`` as "no limit", so an unclamped parameter would
    hand the whole task table to any loopback caller.
    """

    def _client(self, service: _FakeService) -> TestClient:
        app = FastAPI()
        app.include_router(build_a2a_router(service, "assistant", "3.0"))
        return TestClient(app, client=("127.0.0.1", 50000), raise_server_exceptions=False)

    def test_a_negative_limit_becomes_one_page_entry(self) -> None:
        service = _FakeService()
        service.add(_task("t-1"))
        service.add(_task("t-2"))
        response = self._client(service).get("/a2a/tasks", params={"limit": -1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.list_limits, [1])
        self.assertEqual(len(response.json()["tasks"]), 1)

    def test_an_oversized_limit_is_capped(self) -> None:
        service = _FakeService()
        service.add(_task("t-1"))
        response = self._client(service).get("/tasks", params={"limit": 100000})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(service.list_limits, [500])

    def test_the_json_rpc_dispatch_clamps_too(self) -> None:
        from ai_assistant.adapters.a2a_router import _dispatch_json_rpc

        service = _FakeService()
        service.add(_task("t-1"))
        result = _dispatch_json_rpc(service, "tasks/list", {"limit": -1}, "default")
        self.assertEqual(service.list_limits, [1])
        self.assertEqual(len(result["tasks"]), 1)


class MultiTenantOwnershipTests(unittest.TestCase):
    """One A2A bearer token cannot reach another tenant's task (ADR 0008)."""

    def setUp(self) -> None:
        self.service = _FakeService()
        self.policy = TransportPolicy(a2a_tenants={"tok-a": "alpha", "tok-b": "beta"})
        self.client = _client(self.service, policy=self.policy)

    def _headers(self, token: str) -> dict[str, str]:
        return {"A2A-Version": "0.3", "Authorization": f"Bearer {token}"}

    def _send(self, token: str) -> dict:
        response = self.client.post(
            "/message:send", json=_send_body("hello"), headers=self._headers(token),
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def test_each_token_only_sees_its_own_tasks(self) -> None:
        task_id = self._send("tok-a")["task"]["id"]
        h_a, h_b = self._headers("tok-a"), self._headers("tok-b")

        self.assertEqual(self.client.get(f"/tasks/{task_id}", headers=h_a).status_code, 200)
        self.assertEqual(self.client.get(f"/tasks/{task_id}", headers=h_b).status_code, 404)
        self.assertEqual(self.client.post(f"/tasks/{task_id}:cancel", headers=h_b).status_code, 409)
        self.assertEqual(
            self.client.post(f"/tasks/{task_id}:resume", json={"input": {}}, headers=h_b).status_code,
            409,
        )
        self.assertEqual(self.client.post(f"/tasks/{task_id}:subscribe", headers=h_b).status_code, 404)
        self.assertEqual(self.client.get("/tasks", headers=h_b).json()["tasks"], [])
        self.assertEqual(len(self.client.get("/tasks", headers=h_a).json()["tasks"]), 1)

    def test_an_unknown_token_is_rejected(self) -> None:
        response = self.client.post(
            "/message:send", json=_send_body("hello"), headers=self._headers("not-a-tenant"),
        )
        self.assertEqual(response.status_code, 401)

    def test_single_token_mode_still_requires_that_token(self) -> None:
        client = _client(_FakeService(), policy=TransportPolicy(a2a_token="secret"))
        body = _send_body("hello")
        self.assertEqual(client.post("/message:send", json=body, headers={"A2A-Version": "0.3"}).status_code, 401)
        self.assertEqual(
            client.post("/message:send", json=body,
                        headers={"A2A-Version": "0.3", "Authorization": "Bearer secret"}).status_code,
            200,
        )


if __name__ == "__main__":
    unittest.main()
