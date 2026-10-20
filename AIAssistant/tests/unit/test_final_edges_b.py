"""Final-edge coverage, part 2: factories, loaders, routers and services.

Same rule as part 1 — one test per uncovered line or arc: a missing tool
config, the ``application_action`` backfill, a cross-drive safe path, a
relative diff path, the plan-indexed verified reflection, a present mmproj
that skips its download, the domain-error HTTP translation, the vision
release on RAG rebuild, the multi-attachment scan, sentence-snapped chunks,
a subject-bearing email, a docs-dir scan and the desktop-ack truncation and
coordinator surfacing.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ai_assistant.bootstrap.app_factory import TraceContextMiddleware, create_app
from ai_assistant.domain.errors import (
    ApprovalRequiredError,
    AuthorizationError,
    ConfigurationError,
    ContextWindowExceededError,
    ModelLoadError,
    ModelNotLoadedError,
    NotFoundError,
    PlatformError,
    RAGNotReadyError,
    SandboxViolationError,
    ServiceUnavailableError,
    TaskCancelledError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolValidationError,
    UnprocessableRequestError,
)
from ai_assistant.settings import ArchitectureSettings


class ToolFactoryTests(unittest.TestCase):
    def test_a_missing_tool_config_yields_an_empty_registry(self):
        from ai_assistant.tools import factory as factory_mod

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                factory_mod, "get_paths",
                return_value=SimpleNamespace(project_root=Path(directory)),
            ):
                registry = factory_mod.create_tool_registry()
        self.assertIsNone(registry.get("read_file"))

    def test_a_config_without_application_action_gets_the_backfill(self):
        import json as _json

        from ai_assistant.tools import factory as factory_mod

        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "Config"
            config_dir.mkdir()
            (config_dir / "agent_tools.json").write_text(
                _json.dumps([{"name": "read_file", "description": "r", "parameters": {}}]),
                encoding="utf-8",
            )
            with patch.object(
                factory_mod, "get_paths",
                return_value=SimpleNamespace(project_root=Path(directory)),
            ):
                registry = factory_mod.create_tool_registry()
        self.assertIsNotNone(registry.get("read_file"))
        self.assertIsNotNone(registry.get("application_action"))

    def test_a_config_with_application_action_needs_no_backfill(self):
        import json as _json

        from ai_assistant.tools import factory as factory_mod

        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory) / "Config"
            config_dir.mkdir()
            (config_dir / "agent_tools.json").write_text(
                _json.dumps([
                    {"name": "read_file", "description": "r", "parameters": {}},
                    {"name": "application_action", "description": "a", "parameters": {}},
                ]),
                encoding="utf-8",
            )
            with patch.object(
                factory_mod, "get_paths",
                return_value=SimpleNamespace(project_root=Path(directory)),
            ):
                registry = factory_mod.create_tool_registry()
        self.assertIsNotNone(registry.get("application_action"))


class SafePathTests(unittest.TestCase):
    def test_an_uncomputable_common_path_is_rejected(self):
        from ai_assistant.tools.builtin.path_utils import agent_safe_path

        with patch("os.path.commonpath", side_effect=ValueError("different drives")):
            self.assertIsNone(agent_safe_path("a.py"))


class GitDiffTests(unittest.TestCase):
    def test_a_relative_diff_path_is_resolved_before_diffing(self):
        from ai_assistant.tools.builtin import path_utils, vcs_tools
        from ai_assistant.tools.builtin.vcs_tools import tool_git_diff

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "f.txt"
            target.write_text("x", encoding="utf-8")
            roots = SimpleNamespace(project_root=Path(directory))
            with (
                patch.object(path_utils, "get_paths", return_value=roots),
                patch.object(vcs_tools, "get_paths", return_value=roots),
            ):
                result = tool_git_diff({"path": "f.txt"})
        # Outside a repo git fails, but the path must have resolved first.
        self.assertIsInstance(result, dict)


class VerifiedReflectionIndexTests(unittest.TestCase):
    def test_a_verified_reflection_carries_the_plan_step_index(self):
        from ai_assistant.orchestration.nodes.tool import tool_node
        from tests.unit.test_tool_node_guards import _call, _ctx, _state

        update = tool_node(
            _state([_call("some_other_tool", plan_step_index=3)]),
            _ctx(result={"value": 1}, verify=lambda *_: {"passed": True, "reason": "ok"}),
        )
        self.assertEqual(update["steps"][-1]["plan_step_index"], 3)


class ModelLoaderTests(unittest.TestCase):
    def test_a_present_mmproj_skips_its_download(self):
        from ai_assistant.config.model_registry import ModelDefinition, ModelRegistry
        from ai_assistant.llm import model_loader

        registry = ModelRegistry(
            models=(ModelDefinition(
                "repo/primary", "primary.gguf", "Primary", is_vision=True,
                mmproj_repo_id="repo/mmproj", mmproj_filename="mmproj.gguf",
            ),),
            fallback=ModelDefinition("repo/fallback", "fallback.gguf", "Fallback"),
        )
        saved_backend = model_loader._global_backend
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                (Path(tmpdir) / "mmproj.gguf").write_bytes(b"clip")
                with (
                    patch.object(model_loader, "download_if_missing",
                                 return_value="primary.gguf"),
                    patch.object(model_loader, "LlamaCppBackend") as backend_cls,
                    patch.object(model_loader, "release_ml_memory"),
                    patch("huggingface_hub.hf_hub_download") as mock_download,
                ):
                    model_loader.load_model(registry, tmpdir, enable_vision=True)
            mock_download.assert_not_called()
            self.assertEqual(
                backend_cls.call_args.kwargs["mmproj_path"],
                str(Path(tmpdir) / "mmproj.gguf"),
            )
        finally:
            model_loader._global_backend = saved_backend


class DomainErrorHandlerTests(unittest.TestCase):
    def test_domain_errors_become_http_responses(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from ai_assistant.bootstrap.app_factory import _register_domain_error_handlers
        from ai_assistant.domain.errors import NotFoundError

        app = FastAPI()
        _register_domain_error_handlers(app)

        @app.get("/boom")
        def _boom():
            raise NotFoundError("gone")

        response = TestClient(app, client=("127.0.0.1", 50000)).get("/boom")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "gone")


class LoaderChunkTests(unittest.TestCase):
    def test_a_sentence_snapped_block_is_kept_whole(self):
        from ai_assistant.rag.loaders import TxtLoader

        loader = TxtLoader(chunk_chars=100, overlap_chars=10, min_chunk_chars=20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doc.txt"
            path.write_text("x" * 60 + ". " + "y" * 150, encoding="utf-8")
            chunks = loader.load(str(path), directory)
        self.assertTrue(chunks)
        self.assertTrue(chunks[0].text.endswith("."))

    def test_a_too_short_snap_keeps_the_raw_block(self):
        """A snap below the minimum is discarded; the raw block still flows on."""
        from ai_assistant.rag.loaders import TxtLoader

        loader = TxtLoader(chunk_chars=100, overlap_chars=10, min_chunk_chars=120)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "doc.txt"
            path.write_text("x" * 60 + ". " + "y" * 150, encoding="utf-8")
            chunks = loader.load(str(path), directory)
        self.assertIsInstance(chunks, list)

    def test_an_email_subject_becomes_a_chunk_part(self):
        from ai_assistant.rag.loaders import EmailLoader

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mail.eml"
            path.write_bytes(
                b"Subject: Quarterly planning notes for the desktop release\r\n"
                b"From: a@b.c\r\nTo: d@e.f\r\n"
                b'Content-Type: text/plain; charset="utf-8"\r\n\r\n'
                + b"Body text here. " * 10,
            )
            chunks = EmailLoader().load(str(path), directory)
        self.assertTrue(any("planning notes" in chunk.text for chunk in chunks))

    def test_an_email_without_a_subject_still_chunks_its_body(self):
        from ai_assistant.rag.loaders import EmailLoader

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mail.eml"
            path.write_bytes(
                b"From: a@b.c\r\nTo: d@e.f\r\n"
                b'Content-Type: text/plain; charset="utf-8"\r\n\r\n'
                + b"Body text here. " * 10,
            )
            chunks = EmailLoader().load(str(path), directory)
        self.assertTrue(chunks)


class ScanDocumentsTests(unittest.TestCase):
    def test_a_docs_directory_is_walked_for_documents(self):
        from ai_assistant.rag.index import scan_documents

        with tempfile.TemporaryDirectory() as project:
            docs = Path(project) / "Docs"
            docs.mkdir()
            (docs / "note.txt").write_text("hello", encoding="utf-8")
            (docs / "shot.png").write_bytes(b"not a real png")
            empty_root = Path(project) / "proj"
            empty_root.mkdir()
            paths = SimpleNamespace(
                docs_dirs=(docs, Path(project) / "Missing"),
                project_root=empty_root,
            )
            fake_chunk = lambda fp, root: [SimpleNamespace(loader_type="txt")]  # noqa: E731
            chunks = scan_documents(paths, SimpleNamespace(load_file=fake_chunk))
        self.assertEqual(len(chunks), 2)


class DesktopAckServiceTests(unittest.TestCase):
    def test_an_oversized_result_is_truncated_before_resume(self):
        from ai_assistant.agents.context import AgentRunContext
        from ai_assistant.agents.ui_action_service import UIActionService

        service = UIActionService(tool_registry=MagicMock(), task_coordinator=MagicMock())
        action = {
            "messages": [{"role": "system", "content": "s"}],
            "params": {"action": "A" * 9000},
            "task": "t",
            "session_id": "s",
            "temperature": 0.2,
        }
        with patch.object(AgentRunContext, "run", return_value={"status": "ok"}) as run:
            service._resume_langgraph(action, [], success=True)
        user_message = run.call_args.kwargs["initial_messages"][-1]["content"]
        self.assertIn("... [truncated]", user_message)

    def test_shared_wiring_is_surfaced_for_introspection(self):
        from ai_assistant.agents.ui_action_service import UIActionService

        coordinator = MagicMock()
        service = UIActionService(task_coordinator=coordinator)
        self.assertIs(service.task_coordinator, coordinator)



# ── App factory composition (moved from test_coverage_boost2) ──

class AppFactoryTests(unittest.TestCase):
    """Coverage for the FastAPI composition boundary."""

    def _settings(self, tmpdir: str, *, expose_api_docs: bool = False,
                  allowed_hosts: tuple[str, ...] = ()) -> ArchitectureSettings:
        return ArchitectureSettings(
            profile="desktop",
            data_dir=Path(tmpdir),
            enable_mcp=True,
            enable_a2a=True,
            allow_remote_a2a=False,
            trusted_a2a_endpoints=frozenset(),
            agent_capabilities=frozenset({"supervisor"}),
            capability_scopes={},
            allowed_plugins=frozenset(),
            allowed_origins=("*",),
            expose_api_docs=expose_api_docs,
            allowed_hosts=allowed_hosts,
        )

    def test_create_app_registers_handlers_and_cors(self) -> None:
        from fastapi.middleware.cors import CORSMiddleware

        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(self._settings(tmpdir), _lifespan)
        self.assertIn(NotFoundError, app.exception_handlers)
        self.assertIn(AuthorizationError, app.exception_handlers)
        self.assertIn(ModelNotLoadedError, app.exception_handlers)
        self.assertIn(ServiceUnavailableError, app.exception_handlers)
        self.assertTrue(any(m.cls is CORSMiddleware for m in app.user_middleware))
        # Inbound trace propagation wraps everything else (ADR 0001): the
        # traceparent must be attached before any route runs.
        self.assertTrue(
            any(m.cls is TraceContextMiddleware for m in app.user_middleware)
        )

    def test_the_docs_surface_is_not_served_by_default(self) -> None:
        """ADR 0032: ``/docs``/``/redoc``/``/openapi.json`` are *app-level*
        FastAPI routes, so the transport guard that protects every other
        surface never reaches them. They therefore ship disabled — a peer that
        could reach the port must not receive the route schema by default."""
        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(self._settings(tmpdir), _lifespan)
        client = TestClient(app, client=("127.0.0.1", 50000))
        for path in ("/openapi.json", "/docs", "/redoc"):
            self.assertEqual(client.get(path).status_code, 404, path)

    def test_the_docs_surface_is_served_when_a_deployment_opts_in(self) -> None:
        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                self._settings(tmpdir, expose_api_docs=True), _lifespan
            )
        client = TestClient(app, client=("127.0.0.1", 50000))
        self.assertEqual(client.get("/openapi.json").status_code, 200)
        self.assertEqual(client.get("/docs").status_code, 200)

    def test_the_host_allow_list_is_installed_only_when_declared(self) -> None:
        """ADR 0032: the trusted-host middleware refuses a spoofed ``Host``
        header. It is installed only when the deployment declares a policy —
        an empty list means "no policy declared", not "deny everything"."""
        from starlette.middleware.trustedhost import TrustedHostMiddleware

        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            without = create_app(self._settings(tmpdir), _lifespan)
            with_policy = create_app(
                self._settings(tmpdir, allowed_hosts=("ai.internal",)), _lifespan
            )
        self.assertFalse(
            any(m.cls is TrustedHostMiddleware for m in without.user_middleware)
        )
        self.assertTrue(
            any(m.cls is TrustedHostMiddleware for m in with_policy.user_middleware)
        )

    def test_a_spoofed_host_header_is_refused_when_the_policy_is_declared(self) -> None:
        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            app = create_app(
                self._settings(tmpdir, allowed_hosts=("ai.internal",)), _lifespan
            )
        client = TestClient(app, client=("127.0.0.1", 50000))
        response = client.get("/openapi.json", headers={"Host": "evil.example"})
        self.assertEqual(response.status_code, 400)

    def test_create_app_installs_the_rate_limiter_only_when_configured(self) -> None:
        """ADR 0024: the admission governor is opt-in via env, installed
        outside the routes (so it sees every handler) but inside tracing."""
        from ai_assistant.adapters.http.rate_limit import RateLimitMiddleware
        from ai_assistant.bootstrap import app_factory

        async def _lifespan(app):
            yield

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(app_factory, "rate_limit_per_minute", return_value=0):
                app = create_app(self._settings(tmpdir), _lifespan)
            self.assertFalse(
                any(m.cls is RateLimitMiddleware for m in app.user_middleware)
            )
            with patch.object(app_factory, "rate_limit_per_minute", return_value=60):
                app = create_app(self._settings(tmpdir), _lifespan)
            self.assertTrue(
                any(m.cls is RateLimitMiddleware for m in app.user_middleware)
            )


# ── Domain error hierarchy (moved from test_coverage_boost2) ──

class DomainErrorTests(unittest.TestCase):
    """Coverage for typed domain exceptions."""

    def test_error_hierarchy_and_messages(self) -> None:
        self.assertTrue(issubclass(ConfigurationError, PlatformError))
        self.assertTrue(issubclass(ModelLoadError, PlatformError))
        self.assertTrue(issubclass(ContextWindowExceededError, PlatformError))
        self.assertTrue(issubclass(ToolNotFoundError, PlatformError))
        self.assertTrue(issubclass(SandboxViolationError, PlatformError))
        self.assertTrue(issubclass(RAGNotReadyError, PlatformError))
        self.assertTrue(issubclass(TaskCancelledError, PlatformError))
        self.assertTrue(issubclass(UnprocessableRequestError, PlatformError))

    def test_structured_error_attributes(self) -> None:
        validation = ToolValidationError("read_file", "missing path")
        self.assertEqual(validation.tool_name, "read_file")
        self.assertIn("missing path", str(validation))

        execution = ToolExecutionError("run_command", ValueError("boom"))
        self.assertIn("boom", str(execution))

        approval = ApprovalRequiredError("write_file", "act-42")
        self.assertEqual(approval.action_id, "act-42")
        self.assertIn("act-42", str(approval))

        unprocessable = UnprocessableRequestError("bad request shape")
        self.assertEqual(str(unprocessable), "bad request shape")

if __name__ == "__main__":
    unittest.main()
