"""Unit tests for the /admin/* HTTP router.

The admin endpoints mutate module-level legacy singletons, so the tests replace
``llm_module``/``rag_module`` with fakes and stub ``threading.Thread``
and ``os._exit`` to keep the suite process-safe.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import ai_assistant.adapters.http.admin_routes as admin_routes
from ai_assistant.adapters.http.admin_routes import build_admin_router


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


class _FakeLogger:
    def info(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def exception(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


class _FakeLLMModule:
    def __init__(self, lock=None, is_vision: bool = False, load_error: Exception | None = None) -> None:
        self.llm = "loaded-llm"
        self.lock = lock or _NullLock()
        self.is_vision_model = is_vision
        self.active_model_desc = "fake-model"
        self.load_calls = 0
        self.load_error = load_error
        self.llm_lock = self.lock
        self.reload_gate_calls = 0

    def load_model(self) -> None:
        self.load_calls += 1
        if self.load_error is not None:
            raise self.load_error

    def model_reload(self):
        self.reload_gate_calls += 1
        return _NullLock()


class _FakeRAGModule:
    def __init__(self, init_error: Exception | None = None) -> None:
        self.rag_lock = _NullLock()
        self.init_error = init_error
        self.release_embedding_calls = 0
        self.released_memory = False
        self.knowledge_index = None
        self.knowledge_chunks = []
        self.bm25_index = None
        self.embed_model_ref = None
        self._reranker = None
        self.reset_calls = 0

    def reset_index_state(self) -> None:
        """ADR 0032: the module owns the "the RAG state is empty" transition."""
        self.reset_calls += 1
        self.knowledge_index = None
        self.knowledge_chunks = []
        self.bm25_index = None
        self.embed_model_ref = None
        self._reranker = None

    def _release_ml_memory(self) -> None:
        self.released_memory = True

    def initialize_rag(self, force_rebuild: bool = False, enable_reranker: bool = True) -> int:
        if self.init_error is not None:
            raise self.init_error
        return 42

    def release_embedding_for_vision(self) -> None:
        self.release_embedding_calls += 1


class _FakeActionManifest:
    def __init__(self) -> None:
        self.reloaded = False

    def reload_manifest(self) -> None:
        self.reloaded = True


class _FakeThread:
    def __init__(self, target=None, *args, **kwargs) -> None:
        self.target = target

    def start(self) -> None:
        if self.target is not None:
            self.target()


def _build_client(
    llm: _FakeLLMModule | None = None,
    rag: _FakeRAGModule | None = None,
    manifest: _FakeActionManifest | None = None,
) -> tuple[TestClient, dict[str, object]]:
    llm = llm or _FakeLLMModule()
    rag = rag or _FakeRAGModule()
    manifest = manifest or _FakeActionManifest()
    state: dict[str, object] = {
        "reset_calls": 0,
        "refresh_calls": 0,
        "rebuild_calls": 0,
    }

    def _reset() -> None:
        state["reset_calls"] = int(state["reset_calls"]) + 1

    def _refresh() -> None:
        state["refresh_calls"] = int(state["refresh_calls"]) + 1

    def _rebuild() -> None:
        state["rebuild_calls"] = int(state["rebuild_calls"]) + 1

    app = FastAPI()
    app.include_router(
        build_admin_router(
            llm_module=llm,
            rag_module=rag,
            action_manifest_module=manifest,
            reset_agent_state_fn=_reset,
            models=["fake-model"],
            model_idx=0,
            refresh_agent_routes_fn=_refresh,
            rebuild_chatbot_agent_fn=_rebuild,
            logger=_FakeLogger(),
        )
    )
    return TestClient(app, client=("127.0.0.1", 50000)), state


class AdminRoutesTest(unittest.TestCase):
    def test_release_vram_unloads_and_gc_collects(self) -> None:
        llm = _FakeLLMModule()
        client, _ = _build_client(llm=llm)
        response = client.post("/admin/release-vram")
        self.assertEqual(response.status_code, 200)
        self.assertIn("VRAM", response.json()["message"])
        self.assertEqual(llm.reload_gate_calls, 1)

    def test_release_vram_error_returns_500(self) -> None:
        llm = _FakeLLMModule(lock=_FailLock())
        client, _ = _build_client(llm=llm)
        response = client.post("/admin/release-vram")
        self.assertEqual(response.status_code, 500)

    def test_reload_model_success(self) -> None:
        llm = _FakeLLMModule()
        client, state = _build_client(llm=llm)
        response = client.post("/admin/reload-model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "fake-model")
        self.assertEqual(llm.load_calls, 1)
        self.assertEqual(llm.reload_gate_calls, 1)
        self.assertEqual(state["rebuild_calls"], 1)

    def test_reload_model_releases_embeddings_for_vision(self) -> None:
        llm = _FakeLLMModule(is_vision=True)
        rag = _FakeRAGModule()
        client, _ = _build_client(llm=llm, rag=rag)
        response = client.post("/admin/reload-model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rag.release_embedding_calls, 1)

    def test_reload_model_error_returns_500(self) -> None:
        llm = _FakeLLMModule(load_error=OSError("driver missing"))
        client, _ = _build_client(llm=llm)
        response = client.post("/admin/reload-model")
        self.assertEqual(response.status_code, 500)
        self.assertIn("driver missing", response.text)

    def test_reload_rag_skipped_when_disabled(self) -> None:
        client, _ = _build_client()
        with mock.patch("ai_assistant.config.rag_config.RAGSettings") as mock_rag:
            mock_rag.from_env_and_dict.return_value = SimpleNamespace(enabled=False)
            response = client.post("/admin/reload-rag")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "skipped")

    def test_reload_rag_rebuilds_index(self) -> None:
        rag = _FakeRAGModule()
        client, state = _build_client(rag=rag)
        with mock.patch("ai_assistant.config.rag_config.RAGSettings") as mock_rag:
            mock_rag.from_env_and_dict.return_value = SimpleNamespace(enabled=True)
            response = client.post("/admin/reload-rag")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chunks"], 42)
        self.assertTrue(rag.released_memory)
        self.assertEqual(state["rebuild_calls"], 1)

    def test_reload_rag_resets_the_state_through_its_single_owner(self) -> None:
        """ADR 0032: the route used to clear five fields itself and leave the
        stale pipeline behind. The module owns the transition now, so the
        route asks for it instead of reaching into the handles."""
        rag = _FakeRAGModule()
        client, _ = _build_client(rag=rag)
        with mock.patch("ai_assistant.config.rag_config.RAGSettings") as mock_rag:
            mock_rag.from_env_and_dict.return_value = SimpleNamespace(enabled=True)
            client.post("/admin/reload-rag")
        self.assertEqual(rag.reset_calls, 1)

    def test_reload_rag_releases_embeddings_for_vision(self) -> None:
        llm = _FakeLLMModule(is_vision=True)
        rag = _FakeRAGModule()
        client, _ = _build_client(llm=llm, rag=rag)
        with mock.patch("ai_assistant.config.rag_config.RAGSettings") as mock_rag:
            mock_rag.from_env_and_dict.return_value = SimpleNamespace(enabled=True)
            response = client.post("/admin/reload-rag")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(rag.release_embedding_calls, 1)

    def test_reload_rag_error_returns_500(self) -> None:
        rag = _FakeRAGModule(init_error=RuntimeError("index boom"))
        client, _ = _build_client(rag=rag)
        with mock.patch("ai_assistant.config.rag_config.RAGSettings") as mock_rag:
            mock_rag.from_env_and_dict.return_value = SimpleNamespace(enabled=True)
            response = client.post("/admin/reload-rag")
        self.assertEqual(response.status_code, 500)
        self.assertIn("index boom", response.text)

    def test_reload_agent_resets_state_and_refreshes(self) -> None:
        manifest = _FakeActionManifest()
        client, state = _build_client(manifest=manifest)
        response = client.post("/admin/reload-agent")
        self.assertEqual(response.status_code, 200)
        # reset runs before and after the refresh.
        self.assertEqual(state["reset_calls"], 2)
        self.assertEqual(state["refresh_calls"], 1)
        self.assertTrue(manifest.reloaded)

    def test_reload_agent_error_returns_500(self) -> None:
        class _ExplodingManifest(_FakeActionManifest):
            def reload_manifest(self) -> None:
                msg = "manifest boom"
                raise RuntimeError(msg)

        client, _ = _build_client(manifest=_ExplodingManifest())
        response = client.post("/admin/reload-agent")
        self.assertEqual(response.status_code, 500)
        self.assertIn("manifest boom", response.text)

    def test_shutdown_requests_process_exit(self) -> None:
        client, _ = _build_client()
        fake_exit = mock.Mock()
        with (
            mock.patch.object(admin_routes, "time", SimpleNamespace(sleep=mock.Mock())),
            mock.patch.object(admin_routes, "os", SimpleNamespace(_exit=fake_exit)),
            mock.patch.object(admin_routes, "threading", SimpleNamespace(Thread=_FakeThread)),
        ):
            response = client.post("/admin/shutdown")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        fake_exit.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
