"""Import-cycle regression: orchestration must never import the A2A adapter.

History: ``orchestration/supervisor`` used to import ``adapters.a2a_protocol``
inside a module-scope ``try/except`` (graceful no-op), while ``a2a_protocol``
imported ``Specialist`` from the supervisor inside a function body to advertise
Agent-Card skills.  That cycle was import-order dependent: importing the
supervisor first made the ``try/except`` swallow an ``ImportError`` and silently
disable A2A (``supervisor.delegate()`` then never consulted the remote
registry).

The fix moved ``Specialist`` into ``ai_assistant.domain.specialists`` and made
``delegate`` accept the remote resolver as an argument. The composition root
binds its registry into that argument, so the orchestration layer has neither an
adapter import nor a process-wide installed lookup. These tests pin that
invariant so the cycle can never be reintroduced.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_assistant.adapters import a2a_protocol
from ai_assistant.agents.service import AgentService
from ai_assistant.domain.specialists import Specialist
from ai_assistant.orchestration import supervisor

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "ai_assistant"


def _imported_modules(path: Path) -> list[str]:
    """Return every dotted module imported anywhere in ``path`` (AST walk)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            level = node.level
            if level:
                # Resolve relative imports against the file's package path.
                parts = path.relative_to(PACKAGE).with_suffix("").parts[:-1]
                base = ["ai_assistant", *parts[: len(parts) - (level - 1)]]
                modules.append(".".join(base + node.module.split(".")))
            else:
                modules.append(node.module)
    return modules


class OrchestrationDoesNotImportAdaptersTest(unittest.TestCase):
    def test_no_orchestration_module_imports_any_adapter(self) -> None:
        """Any adapter import from orchestration is a latent cycle (ADR 0002)."""
        offenders: list[str] = []
        for path in sorted((PACKAGE / "orchestration").rglob("*.py")):
            for module in _imported_modules(path):
                if module.startswith("ai_assistant.adapters"):
                    offenders.append(f"{path.relative_to(PACKAGE)} -> {module}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_specialist_lives_in_the_domain(self) -> None:
        """The shared role enum is domain-owned so every layer may use it."""
        self.assertIs(supervisor.Specialist, Specialist)
        self.assertEqual(
            [member.value for member in Specialist],
            ["supervisor", "chatbot", "toolapp", "research", "desktop_workflow", "code", "verification"],
        )

    def test_delegate_stays_local_without_a_lookup(self) -> None:
        """No resolver argument → delegation never reaches for a remote."""
        delegation = supervisor.delegate("read the config", "s1", tool="read_file")
        self.assertIsNone(delegation.remote_endpoint)
        self.assertNotIn("A2A remote", delegation.reason)

    def test_delegate_consults_the_injected_lookup(self) -> None:
        """The caller-supplied resolver drives remote routing."""
        delegation = supervisor.delegate(
            "read the config", "s2", tool="read_file",
            remote_lookup=lambda _sid: "http://remote.test",
        )
        self.assertEqual(delegation.remote_endpoint, "http://remote.test")
        self.assertIn("A2A remote: http://remote.test", delegation.reason)

    def test_delegate_fails_closed_when_the_lookup_raises(self) -> None:
        """A broken remote lookup must never break local delegation."""

        def _boom(_sid: str) -> str:
            msg = "registry offline"
            raise RuntimeError(msg)

        delegation = supervisor.delegate(
            "read the config", "s3", tool="read_file", remote_lookup=_boom,
        )
        self.assertIsNone(delegation.remote_endpoint)

    def test_registry_resolver_matches_the_supervisor_port(self) -> None:
        """``RemoteAgentRegistry.resolve`` satisfies the supervisor's port shape."""
        registry = a2a_protocol.RemoteAgentRegistry()
        self.assertTrue(callable(registry.resolve))
        with patch.object(a2a_protocol, "a2a_available", return_value=False):
            self.assertIsNone(registry.resolve("code"))

    def test_agent_service_exposes_the_injected_delegate_to_the_run_context(self) -> None:
        """A facade built without wiring keeps the local default; an injected
        delegate reaches every run path through ``AgentRunContext``."""
        sentinel = object()
        service = AgentService(delegate_fn=sentinel)
        self.assertIs(service._ctx.delegate_fn, sentinel)
        self.assertIsNot(AgentService()._ctx.delegate_fn, sentinel)

    def test_agent_card_capabilities_still_cover_every_specialist(self) -> None:
        """The module-level domain import keeps the default card complete."""
        card = a2a_protocol.build_agent_card()
        advertised = {skill.id for skill in card.skills}
        self.assertEqual(advertised, {member.value for member in Specialist})


class CompositionRootWiresTheRemoteLookupTest(unittest.TestCase):
    """Dependency injection only helps if the composition root supplies it.

    Nothing else in the suite reads the composition root's A2A wiring, so
    dropping the ``delegate_fn`` argument would leave every test here green while
    remote delegation silently stopped working.
    """

    SERVER = Path(__file__).resolve().parents[2] / "src" / "ai_assistant" / "bootstrap" / "server.py"

    def setUp(self) -> None:
        self.source = self.SERVER.read_text(encoding="utf-8")

    def test_the_server_binds_the_registry_into_the_supervisor_delegate(self) -> None:
        self.assertIn("remote_lookup=a2a_registry.resolve", self.source)

    def test_the_server_injects_the_delegate_into_the_agent_service(self) -> None:
        tree = ast.parse(self.source)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AgentService"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one AgentService construction")
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertIn("delegate_fn", keywords)

    def test_the_router_and_resolver_share_one_registry_instance(self) -> None:
        self.assertIn("A2ARouter(registry=a2a_registry)", self.source)
        self.assertIn("a2a_registry = RemoteAgentRegistry()", self.source)

    def test_no_process_wide_lookup_remains(self) -> None:
        self.assertNotIn("set_remote_specialist_lookup", self.source)

    def test_the_composition_root_injects_the_at_rest_cipher(self) -> None:
        self.assertIn("storage_cipher = load_storage_cipher()", self.source)
        self.assertIn("storage_cipher=storage_cipher", self.source)
        self.assertIn("cipher=platform.storage_cipher", self.source)


if __name__ == "__main__":
    unittest.main()
