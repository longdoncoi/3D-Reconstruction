"""Unit tests for the /v1/chat/completions HTTP router.

The router talks to ``llm_module`` and ``rag_module`` through injected fakes and
renders the model stream through a fake chatbot agent, so no real model is ever
loaded — matching the offline CI constraint.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.http.chat_routes import (
    _attachment_may_be_opened,
    build_chat_router,
)
from ai_assistant.ports import SEALED_BINARY_PREFIX

LLM_CTX = 4096


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


class _FailLock:
    def __enter__(self):
        msg = "lock failed"
        raise RuntimeError(msg)

    def __exit__(self, *exc) -> bool:
        return False


class _Lease:
    """The drain contract (ADR 0040): records entry, exit and the live span."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.active = False
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.active = True
        self.entered += 1
        return self.backend

    def __exit__(self, *exc) -> bool:
        self.active = False
        self.exited += 1
        return False


class _FakeChatbotAgent:
    def __init__(self, messages=None) -> None:
        self.messages = messages or [{"role": "user", "content": "hello"}]
        self.last_kwargs: dict = {}

    def build_messages(self, messages_raw, user_query, query_image_b64, language):
        self.last_kwargs = {
            "user_query": user_query,
            "query_image_b64": query_image_b64,
            "language": language,
        }
        return self.messages, {"agent": "fake"}

    def clean_answer(self, answer, chatbot_metadata, finish_reason):
        return answer.strip()


class _LeaseAwareChatbotAgent(_FakeChatbotAgent):
    """Records whether the model lease was held while prompts were built."""

    def __init__(self, llm_module) -> None:
        super().__init__()
        self.llm_module = llm_module
        self.lease_active_during_build: bool | None = None

    def build_messages(self, *args, **kwargs):
        self.lease_active_during_build = self.llm_module.lease.active
        return super().build_messages(*args, **kwargs)


_UNSET = object()


def _fake_generate(messages, max_tokens, temperature, repeat_penalty, stream=False):
    yield {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]}
    yield {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}
    yield {"choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]}


def _make_llm_module(
    llm=_UNSET,
    is_vision: bool = False,
    estimate_tokens=10,
    lock: object = _NullLock(),
) -> SimpleNamespace:
    if llm is _UNSET:
        llm = SimpleNamespace(generate=_fake_generate)
    module = SimpleNamespace(
        llm=llm,
        is_vision_model=is_vision,
        llm_lock=lock,
        estimate_tokens=lambda _text: estimate_tokens,
    )
    module.lease = _Lease(llm)
    module.model_lease = lambda: module.lease
    return module


def _make_rag_module(**kwargs) -> SimpleNamespace:
    attrs = {
        "reranker_active": False,
        "knowledge_chunks": [],
        "CHUNK_CHARS": 1200,
        "MAX_CONTEXT_CHARS": LLM_CTX,
        "_is_image_file": lambda _p: False,
        "_image_to_data_uri": lambda _p: "",
    }
    attrs.update(kwargs)
    return SimpleNamespace(**attrs)


def _build_client(
    llm_module: SimpleNamespace | None = None,
    rag_module: SimpleNamespace | None = None,
    agent: _FakeChatbotAgent | None = None,
) -> tuple[TestClient, _FakeChatbotAgent]:
    agent = agent or _FakeChatbotAgent()
    app = FastAPI()
    app.include_router(
        build_chat_router(
            llm_module or _make_llm_module(),
            rag_module or _make_rag_module(),
            lambda: agent,
            LLM_CTX,
            SimpleNamespace(
                info=lambda *a, **k: None,
                warning=lambda *a, **k: None,
                exception=lambda *a, **k: None,
            ),
        )
    )
    return TestClient(app, client=("127.0.0.1", 50000)), agent


class ChatRoutesTest(unittest.TestCase):
    def test_returns_503_when_llm_not_initialized(self) -> None:
        client, _ = _build_client(llm_module=_make_llm_module(llm=None))
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 503)

    def test_rejects_invalid_role(self) -> None:
        client, _ = _build_client()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "moderator", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 422)

    def test_rejects_empty_messages(self) -> None:
        client, _ = _build_client()
        response = client.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(response.status_code, 422)

    def test_rejects_temperature_out_of_range(self) -> None:
        client, _ = _build_client()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "temperature": 3.0},
        )
        self.assertEqual(response.status_code, 422)

    def test_422_when_only_agent_messages_supplied(self) -> None:
        client, _ = _build_client()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "assistant_agent", "content": "tool payload"}]},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("No conversational messages", response.text)

    def test_streams_and_cleans_inference_text(self) -> None:
        client, agent = _build_client()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertFalse(payload["x_meta"]["vision_model"])
        self.assertEqual(agent.last_kwargs["language"], "vi")
        self.assertIsNone(agent.last_kwargs["query_image_b64"])

    def test_generation_holds_the_model_lease_end_to_end(self) -> None:
        """ADR 0040: one chat request sees one model from prompt to answer.

        The route used to snapshot the handle under ``llm_lock`` and generate
        outside it, so ``/admin/reload-model`` could free the backend under
        the stream. The lease now spans prompt assembly (which reads
        ``is_vision_model``), token estimation and generation.
        """
        llm_module = _make_llm_module()
        observed: list[bool] = []

        def _generate(messages, max_tokens, temperature, repeat_penalty, stream=False):
            observed.append(llm_module.lease.active)
            yield {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}

        llm_module.llm.generate = _generate
        agent = _LeaseAwareChatbotAgent(llm_module)
        client, _ = _build_client(llm_module=llm_module, agent=agent)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(agent.lease_active_during_build)
        self.assertEqual(observed, [True])
        self.assertEqual(llm_module.lease.entered, 1)
        self.assertEqual(llm_module.lease.exited, 1)
        self.assertFalse(llm_module.lease.active)

    def test_a_backend_released_before_the_lease_is_refused(self) -> None:
        """The early ``llm is None`` read is a fast path, not the guarantee.

        A reload can win the race between that read and the lease; the route
        must answer 503 instead of generating against a freed model.
        """
        llm_module = _make_llm_module()
        llm_module.lease.backend = None
        client, _ = _build_client(llm_module=llm_module)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(llm_module.lease.entered, 1)
        self.assertEqual(llm_module.lease.exited, 1)
        self.assertFalse(llm_module.lease.active)

    def test_uses_create_chat_completion_when_generate_missing(self) -> None:
        llm = SimpleNamespace(create_chat_completion=_fake_generate)
        client, _ = _build_client(llm_module=_make_llm_module(llm=llm))
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 200)

    def test_attachments_scan_until_the_first_image(self) -> None:
        """A non-image attachment moves the scan on; the first real image wins."""
        client, agent = _build_client()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{
                "role": "user",
                "content": "see these",
                "attachments": ["notes.txt", "photo.png"],
            }]},
        )
        self.assertEqual(response.status_code, 200)
        # photo.png does not exist on disk, so the encoder fails and no image
        # is attached — but the scan itself ran through both attachments.
        self.assertIsNone(agent.last_kwargs["query_image_b64"])

    def test_missing_image_utils_falls_back_to_no_images(self) -> None:
        """Without the RAG extras and without legacy helpers, nothing is an image."""
        import sys
        from unittest.mock import patch

        rag = SimpleNamespace(reranker_active=False)
        with patch.dict(sys.modules, {"ai_assistant.image_utils": None}):
            client, agent = _build_client(rag_module=rag)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{
                    "role": "user",
                    "content": "see",
                    "attachments": ["photo.png"],
                }]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(agent.last_kwargs["query_image_b64"])

    def test_legacy_data_uri_helper_is_used_as_a_last_resort(self) -> None:
        """A legacy ``_image_to_data_uri`` is missing, so the empty fallback runs."""
        import sys
        from unittest.mock import patch

        rag = SimpleNamespace(reranker_active=False, _is_image_file=lambda _p: True)
        with patch.dict(sys.modules, {"ai_assistant.image_utils": None}):
            client, agent = _build_client(rag_module=rag)
            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{
                    "role": "user",
                    "content": "see",
                    "attachments": ["photo.png"],
                }]},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(agent.last_kwargs["query_image_b64"], "")

    def test_vision_model_counts_tokens_per_part(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "caption"},
                    {"type": "image_url", "image_url": {"url": "data:img"}},
                ],
            }
        ]
        agent = _FakeChatbotAgent(messages=messages)
        client, _ = _build_client(
            llm_module=_make_llm_module(is_vision=True),
            agent=agent,
        )
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "describe"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["x_meta"]["vision_model"])

    def test_rejects_context_overflow(self) -> None:
        llm_module = _make_llm_module(estimate_tokens=3 * LLM_CTX)
        client, _ = _build_client(llm_module=llm_module)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("context window", response.text)

    def test_inference_error_returns_500(self) -> None:
        def _explode(messages, max_tokens, temperature, repeat_penalty, stream=False):
            msg = "inference boom"
            raise RuntimeError(msg)
            yield  # pragma: no cover

        llm = SimpleNamespace(generate=_explode)
        llm_module = _make_llm_module(llm=llm)
        client, _ = _build_client(llm_module=llm_module)
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(response.status_code, 500)
        self.assertIn("inference boom", response.text)
        # The failure path must release the lease too, or one bad request
        # blocks every reload for the life of the process.
        self.assertEqual(llm_module.lease.exited, 1)
        self.assertFalse(llm_module.lease.active)

    def test_attachment_image_is_used_for_retrieval(self) -> None:
        client, agent = _build_client()
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                return_value="data:image/png;base64,AAA",
            ),
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": ["photo.png"],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            agent.last_kwargs["query_image_b64"], "data:image/png;base64,AAA"
        )

    def test_an_attachment_outside_the_upload_roots_is_refused(self) -> None:
        """The request body is untrusted and the route holds the at-rest key.

        A client-named path must not become "read any file on this host":
        the file is only opened when it resolves inside a directory the
        desktop client stores attachments in, or when it is a sealed
        envelope the platform itself produced. Before the gate, a caller
        could name any image-extension file on the machine and have its
        bytes reach the model.
        """
        client, _agent = _build_client()
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        secret = os.path.join(outside, "credential.png")
        with open(secret, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n")
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                return_value="data:image/png;base64,AAA",
            ) as reader,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": [secret],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        reader.assert_not_called()

    def test_a_directory_is_not_opened_as_an_attachment(self) -> None:
        """Only a regular file is opened.

        A FIFO or a device would block the worker thread on a path the
        request body chose — a denial of service the client could trigger
        with one line of JSON.
        """
        client, _agent = _build_client()
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                return_value="data:image/png;base64,AAA",
            ) as reader,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": [directory],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        reader.assert_not_called()

    def test_a_non_regular_file_is_not_even_opened(self) -> None:
        """The guard is asserted on the **open**, not on the outcome.

        A directory raises on open anyway, so the refusal looks the same
        with and without the check — the difference is *that the file was
        opened at all*, and on a FIFO that open blocks.
        """
        directory = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(directory, ignore_errors=True))
        with mock.patch("builtins.open", side_effect=AssertionError("opened")) as opener:
            self.assertFalse(_attachment_may_be_opened(directory))
        opener.assert_not_called()

    def test_an_attachment_that_does_not_exist_is_refused(self) -> None:
        """A name that resolves to nothing is refused, not raised."""
        client, _agent = _build_client()
        missing = os.path.join(tempfile.gettempdir(), "no-such-attachment.png")
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                return_value="data:image/png;base64,AAA",
            ) as reader,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": [missing],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        reader.assert_not_called()

    def test_a_sealed_attachment_is_opened_wherever_it_lives(self) -> None:
        """A sealed envelope is the platform's own convention.

        The marker is a *hint*, not a proof of provenance — anyone can write
        those six bytes — but the route *unseals* the file afterwards, and
        the AEAD refuses anything the platform did not seal. So a planted
        marker yields an error, never a document. The branch exists because
        the desktop client may have stored the file before the upload roots
        were known to the service.
        """
        client, agent = _build_client()
        outside = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(outside, ignore_errors=True))
        sealed = os.path.join(outside, "attachment.png")
        with open(sealed, "wb") as handle:
            handle.write(SEALED_BINARY_PREFIX + b"payload")
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                return_value="data:image/png;base64,AAA",
            ) as reader,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": [sealed],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        reader.assert_called_once()
        self.assertEqual(agent.last_kwargs["query_image_b64"], "data:image/png;base64,AAA")

    def test_attachment_read_failure_falls_back_and_warns(self) -> None:
        client, agent = _build_client()
        with (
            mock.patch("ai_assistant.image_utils.is_image_file", return_value=True),
            mock.patch(
                "ai_assistant.image_utils.image_to_data_uri",
                side_effect=[OSError("bad file"), "data:image/png;base64,BBB"],
            ),
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": ["broken.png", "photo.png"],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            agent.last_kwargs["query_image_b64"], "data:image/png;base64,BBB"
        )

    def test_falls_back_to_rag_module_helpers_when_image_utils_missing(self) -> None:
        rag_module = _make_rag_module(
            _is_image_file=lambda _p: True,
            _image_to_data_uri=lambda _p: "b64fallback",
        )
        client, agent = _build_client(rag_module=rag_module)
        with mock.patch.dict(sys.modules, {"ai_assistant.image_utils": None}):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "what is this?",
                            "attachments": ["photo.png"],
                        }
                    ]
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(agent.last_kwargs["query_image_b64"], "b64fallback")


if __name__ == "__main__":
    unittest.main()
