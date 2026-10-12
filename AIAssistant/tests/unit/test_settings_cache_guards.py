"""Guard-path coverage for the cached platform settings accessors.

``settings.py`` is the single TOML parse point (ADR 0001), but three of its
lines are only reachable when the cache happens to be cold — a test-order
accident, not a guarantee. ``a2a_trusted_endpoints`` is the operator-facing
allowlist the remote A2A adapter enforces, so its merge of the environment
override with the config file is pinned explicitly, including the trailing
slash normalisation that makes the two sources comparable.
"""
from __future__ import annotations

import unittest
from typing import Mapping

from ai_assistant import settings as platform_settings


class TrustedEndpointMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        platform_settings.invalidate_settings_cache()

    def test_the_environment_override_is_normalised_and_merged(self):
        env: Mapping[str, str] = {
            "A2A_REMOTE_AGENTS": " http://a.test/ ,, http://b.test/ , ",
        }
        endpoints = platform_settings.a2a_trusted_endpoints(env)
        self.assertIsInstance(endpoints, frozenset)
        # Trailing slashes are stripped so config and env entries compare equal.
        self.assertEqual(endpoints, frozenset({"http://a.test", "http://b.test"}))

    def test_an_unconfigured_allowlist_is_empty(self):
        self.assertEqual(platform_settings.a2a_trusted_endpoints({}), frozenset())


class SettingsCacheTests(unittest.TestCase):
    def test_invalidate_settings_cache_forces_a_reparse(self):
        first = platform_settings.load_settings()
        self.assertIsNotNone(platform_settings._settings_cache)

        platform_settings.invalidate_settings_cache()
        self.assertIsNone(platform_settings._settings_cache)

        second = platform_settings.load_settings()
        self.assertIsNot(first, second)

    def test_a_second_load_reuses_the_cached_instance(self):
        first = platform_settings.load_settings()
        self.assertIs(first, platform_settings.load_settings())


if __name__ == "__main__":
    unittest.main()
