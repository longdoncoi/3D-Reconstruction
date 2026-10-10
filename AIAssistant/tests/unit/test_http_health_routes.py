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
) -> TestClient:
    llm_module = SimpleNamespace(
        llm=llm,
        active_model_desc="fake-model",
        is_vision_model=False,
    )
    rag_module = SimpleNamespace(
        knowledge_chunks=["c1", "c2"],
        _reranker=reranker,
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
    def test_health_reports_runtime_state_without_llm(self) -> None:
        client = _build_client(llm=None)
        data = client.get("/health").json()
        self.assertEqual(data["status"], "ok")
        self.assertFalse(data["llm_loaded"])
        self.assertEqual(data["rag_chunks"], 2)
        self.assertFalse(data["reranker"])
        self.assertEqual(data["embed_model"], "all-MiniLM-L6-v2")
        self.assertEqual(data["model"], "fake-model")
        self.assertEqual(data["chars_per_token"], 4)
        self.assertEqual(data["chunk_chars"], 1200)
        self.assertEqual(data["max_context"], 4096)
        self.assertEqual(data["model_idx"], 1)
        self.assertTrue(data["platform"]["mcp_enabled"])
        self.assertEqual(data["platform"]["registered_tools"], 1)

    def test_health_reports_loaded_llm_and_reranker(self) -> None:
        client = _build_client(llm=object(), reranker=object())
        data = client.get("/health").json()
        self.assertTrue(data["llm_loaded"])
        self.assertTrue(data["reranker"])
        self.assertIn("uptime_sec", data)

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
        client = _build_client()
        data = client.get("/v1/models").json()
        self.assertEqual(data["data"][0]["id"], "fake-model")
        self.assertEqual(data["data"][0]["object"], "model")


if __name__ == "__main__":
    unittest.main()
