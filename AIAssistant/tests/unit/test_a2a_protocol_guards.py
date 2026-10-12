"""Guard-path coverage for the A2A protocol adapter's local-only edges.

Three seams here have no runtime test: ``AgentSkill.__repr__`` (the debugging
aid a reviewer actually reads when a card looks wrong), the module-scope
``a2a-sdk`` probe (it runs exactly once at import, so the ``ImportError`` arm
is invisible unless the module is re-imported with the SDK masked) and the
remote-lookup loop behind the supervisor's dependency-inverted port (ADR 0002).

The probe tests reload ``a2a_protocol`` on purpose: reloading re-executes the
real ``try``/``except`` block against a controlled ``sys.modules``. Because a
reload also rebinds the module's classes, every test that reloads restores the
whole captured module dictionary afterwards — otherwise ``isinstance`` checks
in unrelated tests would compare against a stale class object.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ai_assistant.adapters import a2a_protocol
from ai_assistant.adapters.a2a_payloads import build_agent_card as build_agent_card_payload
from ai_assistant.adapters.a2a_protocol import (
    AgentCard,
    AgentSkill,
    RemoteAgent,
    build_agent_card,
    discover_remote_agents,
    get_remote_registry,
)

# Captured before any test reloads the module; used to put it back exactly as
# it was found, class objects and mutable registry included.
_MODULE_STATE = dict(a2a_protocol.__dict__)


def _install_fake_a2a() -> None:
    """Put a minimal ``a2a`` package in ``sys.modules`` so the probe succeeds."""
    package = types.ModuleType("a2a")
    types_module = types.ModuleType("a2a.types")
    types_module.AgentCard = object
    package.types = types_module
    sys.modules["a2a"] = package
    sys.modules["a2a.types"] = types_module


def _remove_fake_a2a() -> None:
    sys.modules.pop("a2a.types", None)
    sys.modules.pop("a2a", None)


class _ReloadableAdapterTest(unittest.TestCase):
    """Base class: run with the SDK probe enabled, restore the module after."""

    def setUp(self) -> None:
        self._saved_enabled = os.environ.get("A2A_ENABLED")
        os.environ["A2A_ENABLED"] = "1"

    def tearDown(self) -> None:
        if self._saved_enabled is None:
            os.environ.pop("A2A_ENABLED", None)
        else:
            os.environ["A2A_ENABLED"] = self._saved_enabled
        _remove_fake_a2a()
        a2a_protocol.__dict__.clear()
        a2a_protocol.__dict__.update(_MODULE_STATE)


class AgentSkillTests(unittest.TestCase):
    def test_repr_shows_the_identity_not_the_body(self):
        skill = a2a_protocol.AgentSkill("code", "Code Specialist", "writes code", ["py"])
        self.assertEqual(repr(skill), "AgentSkill(id='code', name='Code Specialist')")

    def test_a_skill_without_tags_advertises_none(self):
        skill = a2a_protocol.AgentSkill("chatbot", "Chatbot", "chats")
        self.assertEqual(skill.tags, [])


class SdkProbeTests(_ReloadableAdapterTest):
    """Re-run the import-time probe under both installation states."""

    def test_the_probe_marks_the_transport_available_when_the_sdk_imports(self):
        _install_fake_a2a()
        importlib.reload(a2a_protocol)
        self.assertTrue(a2a_protocol._sdk_available)
        self.assertTrue(a2a_protocol.a2a_available())

    def test_the_probe_falls_back_to_local_only_mode_when_the_sdk_is_missing(self):
        _remove_fake_a2a()
        importlib.reload(a2a_protocol)
        self.assertFalse(a2a_protocol._sdk_available)
        self.assertFalse(a2a_protocol.a2a_available())


class RemoteSpecialistLookupTests(_ReloadableAdapterTest):
    """The composition root's resolver returns a URL only for an advertised skill."""

    def setUp(self) -> None:
        super().setUp()
        _install_fake_a2a()
        importlib.reload(a2a_protocol)

    def test_a_discovered_advertisement_resolves_to_its_endpoint(self):
        registry = a2a_protocol._remote_registry
        registry["http://remote.test/a2a"] = SimpleNamespace(skills=["chatbot", "code"])
        registry["http://other.test/a2a"] = SimpleNamespace(skills=["research"])
        self.assertEqual(
            a2a_protocol.resolve_remote_specialist("research"), "http://other.test/a2a",
        )

    def test_an_unadvertised_specialist_delegates_locally(self):
        a2a_protocol._remote_registry["http://remote.test/a2a"] = SimpleNamespace(
            skills=["chatbot"],
        )
        self.assertIsNone(a2a_protocol.resolve_remote_specialist("verification"))

    def test_an_empty_registry_delegates_locally(self):
        self.assertIsNone(a2a_protocol.resolve_remote_specialist("chatbot"))



# ── Card construction pins (moved from test_coverage_boost4) ───

class AgentCardTests(unittest.TestCase):
    """Coverage for Agent Card construction and serialisation."""

    def test_build_agent_card_has_skills(self) -> None:
        card = build_agent_card(base_url="https://example.com")
        self.assertIsInstance(card, AgentCard)
        self.assertEqual(card.url, "https://example.com/a2a")
        self.assertGreater(len(card.skills), 0)
        self.assertIsInstance(card.skills[0], AgentSkill)

    def test_dataclass_card_matches_served_payload(self) -> None:
        """The dataclass view and the served wire dict share one source."""
        capabilities = ["research", "supervisor"]
        card = build_agent_card(base_url="https://example.com", capabilities=capabilities)
        payload = build_agent_card_payload(
            agent_name="3D-Reconstruction AI Agent Platform",
            agent_version="1.0.0",
            base_url="https://example.com",
            capabilities=capabilities,
        )
        self.assertEqual(card.url, payload["url"])
        self.assertEqual(card.protocolVersion, payload["protocolVersion"])
        self.assertEqual(
            [skill.id for skill in card.skills],
            [skill["id"] for skill in payload["skills"]],
        )

    def test_agent_card_serialisation_round_trip(self) -> None:
        card = AgentCard(
            name="Test", description="d", url="https://x",
            skills=[AgentSkill(id="s", name="S", description="desc", tags=["t"])],
        )
        payload = json.loads(card.to_json())
        self.assertEqual(payload["name"], "Test")
        self.assertEqual(card.to_dict()["url"], "https://x")


# ── Remote discovery paths (moved from test_coverage_boost4) ───

class RemoteDiscoveryTests(unittest.TestCase):
    """Coverage for remote agent discovery."""

    def test_discover_no_urls_returns_registry(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {}), patch.object(
            a2a_protocol, "a2a_remote_agent_urls", return_value=[]
        ):
            result = discover_remote_agents()
        self.assertEqual(result, {})

    def test_discover_success_populates_registry(self) -> None:
        payload = {
            "name": "Remote",
            "url": "https://remote.example.com",
            "skills": [{"id": "supervisor", "name": "Supervisor"}, {"id": "", "name": "ignored"}],
        }
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
            mock_urlopen.return_value = mock_resp
            result = discover_remote_agents(["https://remote.example.com"])

        self.assertIn("https://remote.example.com", result)
        self.assertIn("supervisor", result["https://remote.example.com"].skills)

    def test_discover_failure_is_swallowed(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen", side_effect=OSError("unreachable")
        ):
            result = discover_remote_agents(["https://remote.example.com"])
        self.assertEqual(result, {})

    def test_get_remote_registry_returns_copy(self) -> None:
        with patch.object(a2a_protocol, "_remote_registry", {"u": RemoteAgent(url="u", card={}, skills={})}):
            snapshot = get_remote_registry()
        self.assertIn("u", snapshot)

    def test_discover_uses_well_known_url(self) -> None:
        base = "https://remote.example.com"
        with patch.object(a2a_protocol, "_remote_registry", {}), patch(
            "urllib.request.urlopen"
        ) as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__.return_value.read.return_value = b'{"name":"R","skills":[]}'
            mock_urlopen.return_value = mock_resp
            discover_remote_agents([base])
        request = mock_urlopen.call_args.args[0]
        self.assertIn(".well-known/agent.json", request.full_url)

if __name__ == "__main__":
    unittest.main()
