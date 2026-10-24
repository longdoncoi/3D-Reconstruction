"""Unit tests for the HTTP health/metrics/models router.

The router receives every dependency through ``build_health_router``, so the
tests inject lightweight fakes for ``llm_module``, ``rag_module`` and
``platform`` instead of touching the legacy singletons.
"""
from __future__ import annotations

import time
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_assistant.adapters.http.health_routes import build_health_router


def _build_client(
    llm: object | None = None,
    reranker: object | None = None,
    chunks: list | None = None,
) -> TestClient:
    llm_module = SimpleNamespace(
        llm=llm,
        active_model_desc="fake-model",
        is_vision_model=False,
    )
    rag_module = SimpleNamespace(
        knowledge_chunks=["c1", "c2"] if chunks is None else chunks,
        reranker_active=reranker is not None,
        CHUNK_CHARS=1200,
        MAX_CONTEXT_CHARS=4096,
    )
    platform = SimpleNamespace(
        plugins=SimpleNamespace(specs=lambda: [SimpleNamespace(name="t1")]),
        settings=SimpleNamespace(
            enable_mcp=True,
            enable_a2a=False,
            allow_remote_a2a=False,
        ),
    )
    app = FastAPI()
    app.include_router(
        build_health_router(
            llm_module,
            rag_module,
            platform,
            server_start_time=time.monotonic(),
            embed_model_name="all-MiniLM-L6-v2",
            chars_per_token=4,
            model_idx=1,
            app=app,
        )
    )
    return TestClient(app, client=("127.0.0.1", 50000))


class HealthRoutesTest(unittest.TestCase):
    def test_health_reports_degraded_without_llm(self) -> None:
        """A missing model is not ``ok`` — probes must see the truth."""
        client = _build_client(llm=None)
        data = client.get("/health").json()
        self.assertEqual(data["status"], "degraded")
        self.assertEqual(data["reasons"], ["llm_not_loaded"])
        self.assertFalse(data["llm_loaded"])
        self.assertEqual(data["rag_chunks"], 2)
        self.assertFalse(data["reranker"])
        self.assertEqual(data["embed_model"], "all-MiniLM-L6-v2")
        # No backend ⇒ no model is advertised (ADR 0041: one source of truth).
        self.assertEqual(data["model"], "")
        self.assertFalse(data["is_vision"])
        self.assertEqual(data["chars_per_token"], 4)
        self.assertEqual(data["chunk_chars"], 1200)
        self.assertEqual(data["max_context"], 4096)
        self.assertEqual(data["model_idx"], 1)
        self.assertTrue(data["platform"]["mcp_enabled"])
        self.assertEqual(data["platform"]["registered_tools"], 1)

    def test_health_degrades_when_the_rag_index_is_empty(self) -> None:
        """P0 #1: an empty corpus used to be reported as a healthy server."""
        client = _build_client(llm=object(), chunks=[])
        data = client.get("/health").json()
        self.assertEqual(data["status"], "degraded")
        self.assertEqual(data["reasons"], ["rag_index_empty"])
        self.assertEqual(data["rag_chunks"], 0)

    def test_health_ok_when_everything_is_loaded(self) -> None:
        client = _build_client(llm=object(), reranker=object())
        data = client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["reasons"], [])
        self.assertTrue(data["llm_loaded"])
        self.assertTrue(data["reranker"])
        self.assertIn("uptime_sec", data)

    def test_livez_always_answers(self) -> None:
        client = _build_client(llm=None, chunks=[])
        response = client.get("/livez")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "alive")

    def test_readyz_is_not_ready_without_a_model(self) -> None:
        client = _build_client(llm=None, chunks=[])
        response = client.get("/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertIn("llm_not_loaded", response.json()["reasons"])

    def test_readyz_is_degraded_with_a_model_but_no_corpus(self) -> None:
        """Ready to serve traffic, but the retrieval quality signal is visible."""
        client = _build_client(llm=object(), chunks=[])
        response = client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "degraded", "reasons": ["rag_index_empty"]})

    def test_readyz_is_ready_when_model_and_index_are_present(self) -> None:
        client = _build_client(llm=object())
        response = client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready", "reasons": []})

    def test_metrics_404_when_observability_disabled(self) -> None:
        client = _build_client()
        with mock.patch("ai_assistant.observability.prometheus_payload", return_value=None):
            response = client.get("/metrics")
        self.assertEqual(response.status_code, 404)

    def test_metrics_exposes_prometheus_payload(self) -> None:
        client = _build_client()
        with mock.patch(
            "ai_assistant.observability.prometheus_payload", return_value="# HELP fake"
        ):
            response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("version=0.0.4", response.headers["content-type"])
        self.assertEqual(response.text, "# HELP fake")

    def test_models_lists_current_model(self) -> None:
        client = _build_client(llm=object())
        data = client.get("/v1/models").json()
        self.assertEqual(data["data"][0]["id"], "fake-model")
        self.assertEqual(data["data"][0]["object"], "model")

    def test_models_is_empty_after_the_model_is_released(self) -> None:
        """A released model must not be advertised to OpenAI-compatible clients."""
        client = _build_client(llm=None)
        self.assertEqual(client.get("/v1/models").json(), {"data": []})


if __name__ == "__main__":
    unittest.main()
