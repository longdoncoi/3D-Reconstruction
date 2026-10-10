"""Unit tests for the observability adapters.

``ai_assistant.observability.metrics``, ``.tracing`` and
``.langsmith_integration`` build their exporters lazily and must degrade to no-ops
when the optional SDKs are missing (they are absent from the offline CI
dependency set). These tests fake the SDK modules and the settings accessors to
pin the enabled, disabled and failure paths without importing Prometheus, OTel
or LangSmith for real.
"""
from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from ai_assistant.observability import langsmith_integration as ls_module
from ai_assistant.observability import metrics as metrics_module
from ai_assistant.observability import tracing as tracing_module


class _FakeMetric:
    def __init__(self, name: str) -> None:
        self.name = name
        self.observations: list = []

    def labels(self, *labels):
        self.observations.append(("labels", labels))
        return self

    def inc(self, amount: float = 1) -> None:
        self.observations.append(("inc", amount))

    def observe(self, value: float) -> None:
        self.observations.append(("observe", value))


class _FakePromClient:
    def __init__(self, fail: bool = False) -> None:
        self.created: dict[str, _FakeMetric] = {}
        self.fail = fail

    def _make(self, name, *args, **kwargs):
        metric = _FakeMetric(name)
        self.created[name] = metric
        return metric

    Counter = Histogram = _make

    def generate_latest(self) -> bytes:
        if self.fail:
            raise RuntimeError("boom")
        return b"prometheus-payload"


class MetricsTests(unittest.TestCase):
    def setUp(self):
        metrics_module._metrics = None
        self.addCleanup(setattr, metrics_module, "_metrics", None)

    def test_disabled_returns_empty(self):
        with mock.patch.object(metrics_module, "observability_enabled", return_value=False):
            self.assertEqual(metrics_module.get_metrics(), {})
        # Recorders and payload are all no-ops when disabled.
        with mock.patch.object(metrics_module, "observability_enabled", return_value=False):
            metrics_module.record_tool("t", True)
            metrics_module.record_approval("approved")
            metrics_module.record_schema_error("t")
            metrics_module.record_token_usage(1, 2)
            metrics_module.record_step_guard("ok")
            self.assertIsNone(metrics_module.prometheus_payload())

    def test_missing_sdk_falls_back_to_empty(self):
        with mock.patch.object(metrics_module, "observability_enabled", return_value=True):
            with mock.patch.dict(sys.modules, {"prometheus_client": None}):
                self.assertEqual(metrics_module.get_metrics(), {})

    def test_caches_registry(self):
        fake = _FakePromClient()
        with mock.patch.object(metrics_module, "observability_enabled", return_value=True):
            with mock.patch.dict(sys.modules, {"prometheus_client": fake}):
                first = metrics_module.get_metrics()
                second = metrics_module.get_metrics()
        self.assertIs(first, second)
        self.assertEqual(first["tool"].name, "agent_tool_calls_total")

    def test_recorders_emit_to_metrics(self):
        fake = _FakePromClient()
        with mock.patch.object(metrics_module, "observability_enabled", return_value=True):
            with mock.patch.dict(sys.modules, {"prometheus_client": fake}):
                metrics = metrics_module.get_metrics()
                metrics_module.record_tool("read", True, duration_seconds=1.5)
                metrics_module.record_tool("read", False, duration_seconds=-1)
                metrics_module.record_approval("rejected")
                metrics_module.record_schema_error("write")
                metrics_module.record_token_usage(3, 4)
                metrics_module.record_step_guard("blocked")
                payload = metrics_module.prometheus_payload()
        self.assertEqual(payload, b"prometheus-payload")
        self.assertIn(("labels", ("read", "success")), metrics["tool"].observations)
        self.assertIn(("labels", ("read", "error")), metrics["tool"].observations)
        self.assertIn(("observe", 1.5), metrics["tool_latency"].observations)
        # Negative durations are clamped to zero.
        self.assertIn(("observe", 0.0), metrics["tool_latency"].observations)
        self.assertIn(("inc", 3), metrics["tokens"].observations)
        self.assertIn(("inc", 4), metrics["tokens"].observations)


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: list = []
        self.entered = False
        self.exited = None

    def __enter__(self):
        self.entered = True
        return self

    def set_attribute(self, key, value):
        self.attributes.append((key, value))

    def __exit__(self, *exc):
        self.exited = exc


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_as_current_span(self, name):
        span = _FakeSpan()
        self.spans.append(span)
        return span


class TracingTests(unittest.TestCase):
    def setUp(self):
        tracing_module._tracer = None
        self.addCleanup(setattr, tracing_module, "_tracer", None)

    def test_tracer_disabled_is_falsy(self):
        with mock.patch.object(tracing_module, "observability_enabled", return_value=False):
            self.assertFalse(tracing_module._get_tracer())

    def test_span_without_backends_is_a_noop(self):
        with mock.patch.object(tracing_module, "_get_tracer", return_value=None):
            with mock.patch.object(tracing_module, "get_metrics", return_value={}):
                with mock.patch.object(tracing_module, "start_langsmith_run", return_value=None):
                    with tracing_module.span("op", ignored=None, kept="v"):
                        pass

    def test_span_enabled_records_success(self):
        tracer = _FakeTracer()
        metrics = {"requests": _FakeMetric("req"), "latency": _FakeMetric("lat")}
        finished: list = []
        with mock.patch.object(tracing_module, "_get_tracer", return_value=tracer):
            with mock.patch.object(tracing_module, "get_metrics", return_value=metrics):
                with mock.patch.object(tracing_module, "start_langsmith_run", return_value="run-1"):
                    with mock.patch.object(
                        tracing_module, "finish_langsmith_run",
                        side_effect=lambda *a: finished.append(a),
                    ):
                        with tracing_module.span("op", kept="v", dropped=None):
                            pass
        span = tracer.spans[0]
        self.assertTrue(span.entered)
        self.assertEqual(span.attributes, [("kept", "v")])
        self.assertIn(("labels", ("op", "success")), metrics["requests"].observations)
        self.assertEqual(finished[0][0], "run-1")
        self.assertEqual(finished[0][1]["outcome"], "success")

    def test_span_enabled_records_error(self):
        tracer = _FakeTracer()
        metrics = {"requests": _FakeMetric("req"), "latency": _FakeMetric("lat")}
        finished: list = []
        with mock.patch.object(tracing_module, "_get_tracer", return_value=tracer):
            with mock.patch.object(tracing_module, "get_metrics", return_value=metrics):
                with mock.patch.object(tracing_module, "start_langsmith_run", return_value="run-2"):
                    with mock.patch.object(
                        tracing_module, "finish_langsmith_run",
                        side_effect=lambda *a: finished.append(a),
                    ):
                        with self.assertRaises(ValueError):
                            with tracing_module.span("op"):
                                raise ValueError("boom")
        span = tracer.spans[0]
        self.assertIsInstance(span.exited[1], ValueError)
        self.assertIn(("labels", ("op", "error")), metrics["requests"].observations)
        self.assertIsInstance(finished[0][2], ValueError)

    def test_langsmith_trace_disabled_yields_empty_context(self):
        with mock.patch.object(tracing_module, "start_langsmith_run", return_value=None):
            with tracing_module.langsmith_trace("run") as context:
                context["outputs"]["x"] = 1
        self.assertIsNone(context["run_id"])

    def test_langsmith_trace_enabled_finishes_run(self):
        finished: list = []
        with mock.patch.object(tracing_module, "start_langsmith_run", return_value="run-3"):
            with mock.patch.object(
                tracing_module, "finish_langsmith_run",
                side_effect=lambda *a: finished.append(a),
            ):
                with tracing_module.langsmith_trace("run") as context:
                    self.assertEqual(context["run_id"], "run-3")
                    context["outputs"]["answer"] = 42
        self.assertEqual(finished[0][0], "run-3")
        self.assertEqual(finished[0][1], {"answer": 42})

    def test_langsmith_trace_propagates_error_and_finishes(self):
        finished: list = []
        with mock.patch.object(tracing_module, "start_langsmith_run", return_value="run-4"):
            with mock.patch.object(
                tracing_module, "finish_langsmith_run",
                side_effect=lambda *a: finished.append(a),
            ):
                with self.assertRaises(ValueError):
                    with tracing_module.langsmith_trace("run"):
                        raise ValueError("boom")
        self.assertIsInstance(finished[0][2], ValueError)


class _FakeLsClient:
    """Models the current SDK (``create_run(id=…)`` / ``update_run(run_id=…)``).

    ``create_signature``/``update_signature`` let a test simulate the older
    clients that used the opposite keyword, exercising the fallback branches.
    """

    def __init__(self, *, create_signature="id", update_signature="run_id",
                 fail=False, client_fail=False):
        self.create_signature = create_signature
        self.update_signature = update_signature
        self.fail = fail
        self.client_fail = client_fail
        self.runs: list[dict] = []
        self.updated: list[dict] = []
        self.feedback: list[dict] = []

    def create_run(self, **payload):
        if self.client_fail:
            raise RuntimeError("nope")
        if self.create_signature not in payload:
            raise TypeError(f"{self.create_signature} required")
        self.runs.append(payload)

    def update_run(self, **payload):
        if self.client_fail:
            raise RuntimeError("nope")
        if self.update_signature not in payload:
            raise TypeError(f"{self.update_signature} required")
        self.updated.append(payload)

    def create_feedback(self, **payload):
        if self.fail:
            raise RuntimeError("nope")
        self.feedback.append(payload)


class LangsmithIntegrationTests(unittest.TestCase):
    def setUp(self):
        ls_module._initialized = False
        ls_module._client_settings = None
        ls_module._langsmith_client = None
        token = ls_module._active_langsmith_run.set(None)
        # Disable real settings for every test; specific tests override this.
        settings_patcher = mock.patch.object(ls_module, "langsmith_settings", return_value=None)
        settings_patcher.start()

        def _restore():
            ls_module._initialized = False
            ls_module._client_settings = None
            ls_module._langsmith_client = None
            ls_module._active_langsmith_run.reset(token)
            settings_patcher.stop()

        self.addCleanup(_restore)

    def _enable_client(self, client) -> None:
        ls_module._initialized = True
        ls_module._client_settings = {"project": "proj"}
        ls_module._langsmith_client = client

    def test_disabled_client_is_none(self):
        with mock.patch.object(ls_module, "langsmith_settings", return_value=None):
            self.assertIsNone(ls_module._get_client())
            self.assertFalse(ls_module.langsmith_available())
            self.assertIsNone(ls_module.get_langsmith_client())

    def test_client_built_from_settings(self):
        created: list = []

        class _Client:
            def __init__(self, **kwargs):
                created.append(kwargs)

        fake_module = SimpleNamespace(Client=_Client)
        with mock.patch.object(
            ls_module, "langsmith_settings",
            return_value={"api_key": "k", "endpoint": "http://ls"},
        ):
            with mock.patch.dict(sys.modules, {"langsmith": fake_module}):
                client = ls_module._get_client()
        self.assertIsInstance(client, _Client)
        self.assertEqual(created, [{"api_key": "k", "api_url": "http://ls"}])
        self.assertTrue(ls_module.langsmith_available())

    def test_client_build_failure_is_swallowed(self):
        class _Client:
            def __init__(self, **kwargs):
                raise RuntimeError("cannot connect")

        with mock.patch.object(
            ls_module, "langsmith_settings",
            return_value={"api_key": "k", "endpoint": "http://ls"},
        ):
            with mock.patch.dict(sys.modules, {"langsmith": SimpleNamespace(Client=_Client)}):
                self.assertIsNone(ls_module._get_client())

    def test_start_run_disabled(self):
        self.assertIsNone(ls_module.start_langsmith_run("n", "chain", {}))

    def test_start_run_success_and_metadata(self):
        client = _FakeLsClient()
        self._enable_client(client)
        run_id = ls_module.start_langsmith_run("n", "chain", {"q": 1}, {"m": True})
        self.assertIsNotNone(run_id)
        self.assertEqual(client.runs[0]["name"], "n")
        self.assertEqual(client.runs[0]["project_name"], "proj")
        self.assertEqual(client.runs[0]["extra"], {"metadata": {"m": True}})

    def test_start_run_adds_parent_when_active(self):
        client = _FakeLsClient()
        self._enable_client(client)
        ls_module._active_langsmith_run.set("parent-1")
        ls_module.start_langsmith_run("n", "chain", {})
        self.assertEqual(client.runs[0]["parent_run_id"], "parent-1")

    def test_start_run_legacy_signature_fallback(self):
        client = _FakeLsClient(create_signature="run_id")
        self._enable_client(client)
        run_id = ls_module.start_langsmith_run("n", "chain", {})
        self.assertIsNotNone(run_id)
        self.assertIn("run_id", client.runs[0])

    def test_start_run_generic_failure_returns_none(self):
        client = _FakeLsClient(client_fail=True)
        self._enable_client(client)
        self.assertIsNone(ls_module.start_langsmith_run("n", "chain", {}))

    def test_finish_run_disabled(self):
        ls_module.finish_langsmith_run("run", {})  # must not raise

    def test_finish_run_success_and_error(self):
        client = _FakeLsClient()
        self._enable_client(client)
        ls_module.finish_langsmith_run("run-1", {"ok": True})
        ls_module.finish_langsmith_run("run-2", {}, ValueError("boom"))
        self.assertEqual(client.updated[0]["run_id"], "run-1")
        self.assertEqual(client.updated[1]["error"], "boom")

    def test_finish_run_legacy_signature_fallback(self):
        client = _FakeLsClient(update_signature="id")
        self._enable_client(client)
        ls_module.finish_langsmith_run("run-1", {})
        self.assertIn("id", client.updated[0])

    def test_finish_run_generic_failure_is_swallowed(self):
        self._enable_client(_FakeLsClient(client_fail=True))
        ls_module.finish_langsmith_run("run", {})

    def test_feedback_disabled(self):
        self.assertFalse(ls_module.record_langsmith_feedback("k"))

    def test_feedback_needs_active_run(self):
        self._enable_client(_FakeLsClient())
        self.assertFalse(ls_module.record_langsmith_feedback("k"))

    def test_feedback_success_with_optional_fields(self):
        client = _FakeLsClient()
        self._enable_client(client)
        ls_module._active_langsmith_run.set("run-x")
        ok = ls_module.record_langsmith_feedback("quality", score=0.9, value="v", comment="c")
        self.assertTrue(ok)
        self.assertEqual(client.feedback[0]["score"], 0.9)
        self.assertEqual(client.feedback[0]["value"], "v")
        self.assertEqual(client.feedback[0]["comment"], "c")

    def test_feedback_explicit_run_id(self):
        client = _FakeLsClient()
        self._enable_client(client)
        self.assertTrue(ls_module.record_langsmith_feedback("k", run_id="explicit"))
        self.assertEqual(client.feedback[0]["run_id"], "explicit")

    def test_feedback_failure_returns_false(self):
        self._enable_client(_FakeLsClient(fail=True))
        self.assertFalse(ls_module.record_langsmith_feedback("k", run_id="x"))


if __name__ == "__main__":
    unittest.main()
