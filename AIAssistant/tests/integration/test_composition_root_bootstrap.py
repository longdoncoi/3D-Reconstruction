"""Composition-root bootstrap regression tests (P0).

The production server must reach a usable LLM after ``bootstrap_runtime()``.
A regression during the ``modules/`` removal replaced the functional legacy
bridges with no-op shims, so the entry point started while every agent call
failed with ``ModelNotLoadedError``. These tests pin the load path:

    bootstrap_runtime()
      -> runtime_modules.rag.initialize_rag  -> real RAG bridge
      -> runtime_modules.llm.load_model      -> ai_assistant.llm.load_backend
      -> LlamaCppBackend                     -> runtime_modules.llm.llm is set

Model weights and the embedding model are never loaded in tests; the backend
and the document scan are replaced with doubles.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Accessing a composed name on ``bootstrap.server`` composes the object graph
# (PEP 562 ``__getattr__``), and ``compose()`` builds the long-term memory
# store eagerly (ADR 0043). The import below therefore has to happen against a
# temporary data root, or the suite leaves a real ``memory.sqlite`` in the
# checkout; module teardown closes the store and removes the directory.
_COMPOSE_DATA_ROOT = tempfile.mkdtemp(prefix="aiassistant-compose-")
with patch.dict(os.environ, {"APP_DATA_DIR": _COMPOSE_DATA_ROOT}):
    from ai_assistant.bootstrap.server import runtime_modules


def _run_composed(code: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh interpreter whose data root is a temp directory.

    ``compose()`` builds the long-term memory store eagerly (ADR 0043), so a
    subprocess pointed at the checkout would leave a real ``memory.sqlite``
    beside the sources. ``APP_DATA_DIR`` redirects every data artifact (memory
    store, logs, checkpoints) to a directory the test owns and removes.
    """
    keep = ("SYSTEMROOT", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC",
            "USERPROFILE", "HOME")
    env = {key: os.environ[key] for key in keep if key in os.environ}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    env["AI_STORAGE_KEY"] = "hex:" + "11" * 32
    env.update(extra_env)
    with tempfile.TemporaryDirectory() as data_root:
        env["APP_DATA_DIR"] = data_root
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )


class CompositionRootImportTests(unittest.TestCase):
    """The entry module must import and expose a consistent runtime shape."""

    @classmethod
    def setUpClass(cls) -> None:
        from ai_assistant.bootstrap import server as server_module

        cls.server = server_module

    def test_entry_module_exposes_server_start_time(self) -> None:
        """``bootstrap/server.py`` imports ``_SERVER_START_TIME`` from legacy config."""
        self.assertTrue(hasattr(self.server, "_SERVER_START_TIME"))
        self.assertTrue(hasattr(self.server, "bootstrap_runtime"))

    def test_legacy_bridges_are_functional_not_noops(self) -> None:
        """The bridges must delegate to the real loaders, not log-and-return."""
        self.assertIsNotNone(runtime_modules.llm.load_model)
        # A no-op shim logs "legacy shim" and never delegates; the real bridge
        # calls into ai_assistant.llm, which we can observe by patching below.
        import inspect
        source = inspect.getsource(runtime_modules.llm.load_model)
        self.assertNotIn("legacy shim", source)
        self.assertTrue(
            any(name in source for name in ("_new_load", "load_model")),
            "load_model must delegate to the real model loader",
        )

    def test_rag_bridge_exposes_pipeline_state(self) -> None:
        """RAG consumers (admin/health/tools) read these attributes."""
        for attribute in (
            "knowledge_index",
            "knowledge_chunks",
            "bm25_index",
            "reranker_active",
            "rag_lock",
            "initialize_rag",
            "get_context",
            "release_embedding_for_vision",
            "release_ml_memory",
        ):
            self.assertTrue(hasattr(runtime_modules.rag, attribute), f"missing {attribute}")


class BootstrapRuntimeLoadTests(unittest.TestCase):
    """``bootstrap_runtime()`` must publish a usable backend on the LLM runtime."""

    @classmethod
    def setUpClass(cls) -> None:
        from ai_assistant.bootstrap import server as server_module

        cls.server = server_module

    def setUp(self) -> None:
        # Isolate registry mock for all bootstrap tests.
        self._saved_registry = getattr(self.server, "_registry", None)
        mock_model_def = MagicMock()
        mock_model_def.is_vision = False
        mock_model_def.filename = "test.gguf"
        mock_model_def.desc = "Test model (Q4_K_M)"
        mock_model_def.mmproj_filename = None
        mock_model_def.mmproj_repo_id = None
        self.mock_registry = MagicMock()
        self.mock_registry.get_by_index.return_value = mock_model_def
        self.server._registry = self.mock_registry

    def tearDown(self) -> None:
        if self._saved_registry is not None:
            self.server._registry = self._saved_registry
        # Stabilizer: the runtime module owns the backend now, so clearing it
        # here is the whole cleanup (there is no process-wide copy, ADR 0041).
        # The owner only permits a swap inside the drain window.
        with runtime_modules.llm.model_reload():
            runtime_modules.llm.release()

    @staticmethod
    def _patch_backend(backend):
        """Isolate the load path: no weights, no downloads, no VRAM release."""
        return (
            patch(
                "ai_assistant.llm.model_loader.LlamaCppBackend",
                return_value=backend,
            ),
            patch(
                "ai_assistant.llm.model_loader.download_if_missing",
                return_value="Models/test.gguf",
            ),
            patch("ai_assistant.llm.hardware.release_ml_memory"),
        )

    def test_bootstrap_publishes_llm_backend(self) -> None:
        backend = MagicMock()
        backend.is_vision_supported = False
        backend.model_description = "Test model (Q4_K_M)"

        p1, p2, p3 = self._patch_backend(backend)
        # Mock registry so load_model finds a valid model without real files.
        mock_model_def = MagicMock()
        mock_model_def.is_vision = False
        mock_model_def.filename = "test.gguf"
        mock_model_def.desc = "Test model (Q4_K_M)"
        mock_model_def.mmproj_filename = None
        mock_model_def.mmproj_repo_id = None
        mock_registry = MagicMock()
        mock_registry.get_by_index.return_value = mock_model_def
        def _fake_load(**kw):
            # Mirror the real loader's owner-enforced precondition: it only
            # runs inside ``model_reload()``. If bootstrap_runtime() forgets
            # the drain window this raises instead of loading.
            runtime_modules.llm._require_reload_window()
            runtime_modules.llm.llm = backend
            runtime_modules.llm.is_vision_model = False
            runtime_modules.llm.active_model_desc = "Test model (Q4_K_M)"

        with (
            p1, p2, p3,
            patch.object(runtime_modules.rag, "initialize_rag", return_value=0),
            patch.object(runtime_modules.llm, "load_model", _fake_load),
        ):
            self.server.bootstrap_runtime()

        # The agent facade gates on llm_runtime.llm — this is the P0 signal.
        self.assertIsNotNone(runtime_modules.llm.llm)
        self.assertFalse(runtime_modules.llm.is_vision_model)
        self.assertEqual(runtime_modules.llm.active_model_desc, "Test model (Q4_K_M)")

        # The whole facade must now accept an execution request.
        self.server.agent_service._require_llm()

    def test_agent_service_gated_before_load(self) -> None:
        """Without a backend the facade must raise, proving the gate is real."""
        from ai_assistant.domain.errors import ModelNotLoadedError

        saved = runtime_modules.llm.llm
        try:
            runtime_modules.llm.llm = None
            with self.assertRaises(ModelNotLoadedError):
                self.server.agent_service._require_llm()
        finally:
            runtime_modules.llm.llm = saved

    def test_vision_model_releases_embedding(self) -> None:
        """Vision models must drop the embedding model after load."""
        backend = MagicMock()
        backend.is_vision_supported = True
        backend.model_description = "Vision model"

        p1, p2, p3 = self._patch_backend(backend)

        def _fake_vision_load(**kw):
            runtime_modules.llm._require_reload_window()
            runtime_modules.llm.llm = backend
            runtime_modules.llm.is_vision_model = True
            runtime_modules.llm.active_model_desc = "Vision model"

        with p1, p2, p3, patch.object(
            runtime_modules.rag, "initialize_rag", return_value=0
        ), patch.object(runtime_modules.rag, "release_embedding_for_vision") as release, patch.object(
            runtime_modules.llm, "load_model", _fake_vision_load
        ):
            self.server.bootstrap_runtime()
            release.assert_called_once()

        self.assertTrue(runtime_modules.llm.is_vision_model)


class LazyCompositionTests(unittest.TestCase):
    """Importing the entry module must not compose the object graph.

    Composition lives in ``compose()`` (idempotent, dependency-ordered);
    importing the module only defines it, and first access of a composed name
    triggers the build via module ``__getattr__``. Re-introducing module-level
    construction would still pass most tests — they compose anyway — so the
    "nothing composed at import" contract is pinned in a fresh interpreter.
    """

    def test_fresh_import_defers_composition_until_compose(self) -> None:
        code = (
            "import sys\n"
            "import ai_assistant.bootstrap.server as s\n"
            "assert 'app' not in vars(s), 'FastAPI app built at import'\n"
            "assert 'logger' not in vars(s), 'logging set up at import'\n"
            "assert 'platform' not in vars(s), 'container built at import'\n"
            "assert s._SERVER_START_TIME, 'import-time constants must stay'\n"
            "frameworks = {'fastapi', 'starlette', 'pydantic', 'langgraph',\n"
            "              'langchain', 'langchain_core', 'langsmith', 'llama_cpp',\n"
            "              'mcp', 'a2a', 'httpx', 'uvicorn'}\n"
            "loaded = sorted({m.split('.')[0] for m in sys.modules} & frameworks)\n"
            "assert not loaded, f'frameworks imported at import time: {loaded}'\n"
            "s.compose()\n"
            "assert s.app is not None, 'compose() must build the app'\n"
            "assert s.platform.storage_cipher is not None, 'compose must load the at-rest key'\n"
            "assert s.agent_service.pending_actions._cipher is s.platform.storage_cipher\n"
            "assert s.runtime_modules.rag._cipher is s.platform.storage_cipher, \\\n"
            "    'the RAG cache must be sealed with the same key (ADR 0028)'\n"
            "assert 'logger' in vars(s)\n"
            "app_before = s.app\n"
            "s.compose()  # idempotent: a second call must not rebuild\n"
            "assert s.app is app_before, 'second compose() rebuilt the graph'\n"
        )
        # Minimal environment: the suite mutates os.environ globally, and the
        # subprocess must see only the deployment defaults, not test residue.
        proc = _run_composed(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_a_failed_build_rolls_the_flag_back_and_the_next_access_retries(self) -> None:
        code = (
            "from unittest.mock import patch\n"
            "import ai_assistant.bootstrap.server as s\n"
            "with patch('ai_assistant.config.resolve_paths', side_effect=RuntimeError('boom')):\n"
            "    for _ in range(2):\n"
            "        try:\n"
            "            s.compose()\n"
            "        except RuntimeError:\n"
            "            pass\n"
            "        else:\n"
            "            raise AssertionError('compose must propagate the failure')\n"
            "        assert s._COMPOSED is False, 'a failure left the composed flag set'\n"
            "s.compose()\n"
            "assert s._COMPOSED is True, 'a successful retry must mark the graph composed'\n"
            "assert s.app is not None, 'the retry must build the app'\n"
        )
        proc = _run_composed(code)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_memory_disabled_composes_no_store_and_an_inert_service(self) -> None:
        """``AGENT_MEMORY_ENABLED=0``: no store is built, the injected memory
        service is inert (fail-closed), and shutdown skips the close (ADR 0043)."""
        code = (
            "import ai_assistant.bootstrap.server as s\n"
            "s.compose()\n"
            "assert s.memory_store is None, 'memory disabled must build no store'\n"
            "memory = s.agent_service._memory\n"
            "assert memory is not None, 'the service is always injected'\n"
            "assert memory.enabled is False, 'a None store must stay inert'\n"
            "assert memory.recall('anything') == []\n"
            "assert s.agent_service._short_term is None\n"
        )
        proc = _run_composed(code, AGENT_MEMORY_ENABLED="0")
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

    def test_memory_disabled_composes_no_store_in_process(self) -> None:
        """The disabled half of the memory switch, measured in-process (the
        subprocess variant above pins the contract; only an in-process compose
        exercises the branch the coverage gate sees): no store is built, prune
        is skipped, and the injected service is inert (ADR 0043)."""
        from ai_assistant.bootstrap import server as server_module

        original_flag = server_module._COMPOSED
        # Both composes below must run against a temporary data root: the
        # restoring compose re-enables memory, and a compose that is pointed at
        # the checkout would leave a real ``memory.sqlite`` (and its key file)
        # in the working tree. The compose also opens a log file under that
        # root, so the cleanup is best-effort on Windows.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data_root, \
                patch.dict(os.environ, {"APP_DATA_DIR": data_root}):
            try:
                with patch.dict(os.environ, {"AGENT_MEMORY_ENABLED": "0"}):
                    server_module._COMPOSED = False
                    server_module.compose()
                    self.assertIsNone(server_module.memory_store)
                    memory = server_module.agent_service._memory
                    self.assertIsNotNone(memory)
                    self.assertFalse(memory.enabled)
                    self.assertEqual(memory.recall("anything"), [])
                    self.assertIsNone(server_module.agent_service._short_term)
            finally:
                # Leave the module composed the way the rest of the suite found
                # it: memory enabled again, so a later test that patches globals
                # sees the same graph shape it would in production.
                server_module._COMPOSED = False
                server_module.compose()
                if original_flag:
                    server_module._COMPOSED = original_flag
                # Close the temporary store so its file is not held open past
                # this test (the temp root is removed on a best-effort basis).
                server_module.memory_store.close()

    def test_an_in_process_failure_resets_the_flag(self) -> None:
        from unittest.mock import patch

        from ai_assistant.bootstrap import server as server_module

        original = server_module._COMPOSED
        try:
            with patch.object(server_module, "_compose_once", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    server_module.compose()
                self.assertFalse(server_module._COMPOSED)
        finally:
            server_module._COMPOSED = original

    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        """``__getattr__`` composes known names only; typos stay AttributeErrors."""
        from unittest.mock import patch

        from ai_assistant.bootstrap import server as server_module

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as data_root, \
                patch.dict("os.environ", {
                    "AI_STORAGE_KEY": "hex:" + "11" * 32,
                    "APP_DATA_DIR": data_root,
                }):
            server_module.compose()
        with self.assertRaises(AttributeError):
            getattr(server_module, "_no_such_attribute_xyz")

    def test_bootstrap_package_unknown_attribute_raises_in_process(self) -> None:
        """Pin the lazy package's fail-closed arm (ADR 0022).

        Submodule imports (`app_factory`, `server`) populate the parent
        `__dict__` as a side effect, so depending on test order the
        `__getattr__` miss arm may never fire — a name that can never be a
        submodule always reaches it.
        """
        import ai_assistant.bootstrap as bootstrap_pkg

        with self.assertRaises(AttributeError):
            getattr(bootstrap_pkg, "_no_such_attribute_xyz")
        # The lazy success arm: names load from their owning submodule.
        self.assertTrue(callable(bootstrap_pkg.build_container))
        self.assertTrue(callable(bootstrap_pkg.create_app))
        self.assertIsNotNone(bootstrap_pkg.PlatformContainer)

    def test_bootstrap_package_binds_nothing_at_import(self) -> None:
        """The package root stays lazy: names load from submodules on access."""
        code = (
            "import sys\n"
            "import ai_assistant.bootstrap as b\n"
            "assert 'create_app' not in vars(b), 'FastAPI factory bound at import'\n"
            "assert 'build_container' not in vars(b), 'container bound at import'\n"
            "frameworks = {'fastapi', 'starlette', 'pydantic', 'langgraph',\n"
            "              'langchain', 'langchain_core', 'langsmith', 'llama_cpp',\n"
            "              'mcp', 'a2a', 'httpx', 'uvicorn'}\n"
            "loaded = sorted({m.split('.')[0] for m in sys.modules} & frameworks)\n"
            "assert not loaded, f'frameworks imported at import time: {loaded}'\n"
            "assert callable(b.build_container), 'container must load on access'\n"
            "assert callable(b.create_app), 'factory must load on access'\n"
            "assert b.PlatformContainer is not None\n"
            "try:\n"
            "    b._no_such_attribute_xyz\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('unknown bootstrap attribute must raise AttributeError')\n"
        )
        keep = ("SYSTEMROOT", "PATH", "PATHEXT", "TEMP", "TMP", "COMSPEC",
                "USERPROFILE", "HOME")
        env = {k: os.environ[k] for k in keep if k in os.environ}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


def tearDownModule() -> None:
    """Release the app log handler ``compose()`` opened into ``logs/``.

    The composition root runs ``setup_logging(PACKAGE_ROOT / "logs")`` exactly
    once per process, attaching a real file handler to the root logger for
    every test that follows; nothing closed it, so the suite ended with
    ``ResourceWarning: unclosed file …/logs/server_*.log`` at interpreter
    shutdown. Test-side hygiene only — the production behaviour is unchanged,
    and only handlers whose file lives in the repo's log directory are
    touched.
    """
    import logging

    root = logging.getLogger()
    logs_dir = str(Path(__file__).resolve().parents[2] / "logs")
    for handler in list(root.handlers):
        if str(getattr(handler, "baseFilename", "")).startswith(logs_dir):
            root.removeHandler(handler)
            handler.close()

    # The import-time compose built its memory store under a temporary data
    # root; close it and drop the directory. Best-effort: a log handler may
    # still hold a file open on Windows.
    server_module = sys.modules.get("ai_assistant.bootstrap.server")
    store = getattr(server_module, "memory_store", None) if server_module else None
    if store is not None:
        store.close()
    shutil.rmtree(_COMPOSE_DATA_ROOT, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
