"""Guard-path coverage for the versioned desktop-action manifest.

The manifest is file-backed and cached, so its failure modes only appear at
the edges: a missing file (test/partial installs), a malformed ``actions``
value, the cache-clear entry point itself and a required parameter the model
omitted. The cache-aware tests patch the path getter with a real
``lru_cache`` function so ``reload_manifest`` keeps working, and always
reload after themselves so no fake entry leaks into other tests.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import ai_assistant.tools.action_manifest as manifest_mod
from ai_assistant.tools.action_manifest import (
    looks_like_ui_action,
    validate_action_params,
)


def _run_with_manifest_file(path: Path, test):
    """Run ``test`` with the manifest reader pointed at ``path``."""
    fake_getter = lru_cache(maxsize=1)(lambda: path)
    with patch.object(manifest_mod, "_get_manifest_path", fake_getter):
        manifest_mod.reload_manifest()
        try:
            return test()
        finally:
            manifest_mod.reload_manifest()


class ManifestFileTests(unittest.TestCase):
    def test_a_missing_manifest_file_falls_back_to_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "no-such-manifest.json"
            result = _run_with_manifest_file(
                missing, lambda: manifest_mod.manifest(),
            )
            self.assertEqual(result, {"actions": []})

    def test_a_manifest_without_an_actions_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "agent_action_manifest.json"
            bad.write_text(json.dumps({"actions": {"open": {}}}), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                _run_with_manifest_file(bad, lambda: manifest_mod.manifest())
            self.assertIn("actions array", str(caught.exception))

    def test_a_valid_manifest_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "agent_action_manifest.json"
            good.write_text(json.dumps({"actions": [
                {"id": "open_app", "aliases": ["open"], "intents": ["open app"],
                 "parameters": {"app": {"required": True}}},
            ]}), encoding="utf-8")
            def _check():
                self.assertEqual(manifest_mod.manifest()["actions"][0]["id"], "open_app")
                self.assertEqual(manifest_mod.canonical_action("open"), "open_app")
            _run_with_manifest_file(good, _check)


class ParameterContractTests(unittest.TestCase):
    _INDEX = {
        "open_app": {
            "id": "open_app",
            "parameters": {"app": {"required": True}},
            "intents": [],
        },
    }

    def test_a_missing_required_parameter_is_rejected(self):
        with patch.object(manifest_mod, "_index", lambda: self._INDEX):
            result, error = validate_action_params({"action": "open_app"})
            self.assertIsNone(result)
            self.assertEqual(error, "open_app requires parameter 'app'")

    def test_a_satisfied_required_parameter_passes(self):
        with patch.object(manifest_mod, "_index", lambda: self._INDEX):
            result, error = validate_action_params({"action": "open_app", "app": "calc"})
            self.assertIsNone(error)
            self.assertEqual(result["action"], "open_app")


class DeprecatedProbeTests(unittest.TestCase):
    def test_the_deprecated_ui_action_probe_always_says_no(self):
        self.assertFalse(looks_like_ui_action("open the browser"))
        self.assertFalse(looks_like_ui_action(""))


if __name__ == "__main__":
    unittest.main()
