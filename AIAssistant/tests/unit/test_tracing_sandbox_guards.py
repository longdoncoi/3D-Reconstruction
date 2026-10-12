"""Guard-path coverage for the tracer cache and the sandbox front door.

The OTel tracer resolves once and is cached: the second call must take the
cache-hit arm, and the first call must handle both an enabled SDK and a
missing one (simulated by halting the import, which raises ``ImportError``
exactly like an uninstalled package). Module-global tracer state is saved
and restored so test order cannot matter.

The sandbox arms are its two cheapest rejections: an interpreter flag the
policy has never heard of, and a command line ``shlex`` itself refuses.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import ai_assistant.observability.tracing as tracing
from ai_assistant.tools.sandbox import _inline_code_requested, run


class TracerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = tracing._tracer

    def tearDown(self) -> None:
        tracing._tracer = self._saved

    def test_a_cached_tracer_is_returned_without_rechecking(self):
        tracing._tracer = False
        self.assertIs(tracing._get_tracer(), False)

    def test_a_disabled_sdk_caches_the_disabled_sentinel(self):
        tracing._tracer = None
        with patch.object(tracing, "observability_enabled", return_value=False):
            self.assertIs(tracing._get_tracer(), False)
        self.assertIs(tracing._tracer, False)

    def test_a_missing_opentelemetry_package_caches_disabled(self):
        """``None`` in ``sys.modules`` makes the import raise ``ImportError``."""
        tracing._tracer = None
        with (
            patch.object(tracing, "observability_enabled", return_value=True),
            patch.dict(sys.modules, {"opentelemetry": None}),
        ):
            self.assertIs(tracing._get_tracer(), False)

    def test_an_installed_sdk_yields_a_cached_tracer(self):
        """A fake SDK proves the success arm without needing the package."""
        from types import SimpleNamespace

        fake_trace = SimpleNamespace(get_tracer=lambda name: "tracer!")
        fake_otel = SimpleNamespace(trace=fake_trace)
        tracing._tracer = None
        with (
            patch.object(tracing, "observability_enabled", return_value=True),
            patch.dict(sys.modules, {"opentelemetry": fake_otel,
                                     "opentelemetry.trace": fake_trace}),
        ):
            self.assertEqual(tracing._get_tracer(), "tracer!")
        self.assertEqual(tracing._tracer, "tracer!")

    def test_a_lost_lock_race_returns_the_cached_sentinel(self):
        """``None`` at the outer check but set on entry means another thread
        won the race — the inner check must take the cache-hit exit."""

        class _PreemptingLock:
            def __enter__(self) -> None:
                tracing._tracer = False

            def __exit__(self, *args) -> bool:
                return False

        tracing._tracer = None
        with (
            patch.object(tracing, "_tracer_lock", _PreemptingLock()),
            patch.object(tracing, "observability_enabled", return_value=True),
        ):
            self.assertIs(tracing._get_tracer(), False)


class InlineCodeFlagTests(unittest.TestCase):
    def test_an_unknown_double_dash_flag_is_inline_code(self):
        self.assertTrue(_inline_code_requested(["python", "--bogus=1"]))

    def test_a_safe_double_dash_flag_with_a_value_is_allowed(self):
        self.assertFalse(_inline_code_requested(["python", "--help=x", "job.py"]))

    def test_a_script_path_is_not_inline_code(self):
        self.assertFalse(_inline_code_requested(["python", "job.py"]))

    def test_a_non_interpreter_ignores_its_flags(self):
        self.assertFalse(_inline_code_requested(["ruff", "--bogus=1"]))

    def test_stdin_and_terminator_markers_are_inline_code(self):
        self.assertTrue(_inline_code_requested(["python", "-"]))
        self.assertTrue(_inline_code_requested(["python", "job.py", "--"]))

    def test_safe_short_flags_are_allowed(self):
        self.assertFalse(_inline_code_requested(["python", "-B", "job.py"]))

    def test_short_options_are_inline_code(self):
        """``-c``/``-m`` (attached or not) turn the interpreter into ``eval``."""
        self.assertTrue(_inline_code_requested(["python", "-c", "print(1)"]))
        self.assertTrue(_inline_code_requested(["python", "-mmod"]))


class SandboxFrontDoorTests(unittest.TestCase):
    def test_an_unparseable_command_is_rejected_without_running(self):
        outcome = run("git 'abc", ".", 10)
        self.assertIn("error", outcome)
        self.assertIn("Invalid command", outcome["error"])


if __name__ == "__main__":
    unittest.main()
