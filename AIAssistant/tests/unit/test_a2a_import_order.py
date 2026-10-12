"""Import-cycle regression: orchestration must never import the A2A adapter.

History: ``orchestration/supervisor`` used to import ``adapters.a2a_protocol``
inside a module-scope ``try/except`` (graceful no-op), while ``a2a_protocol``
imported ``Specialist`` from the supervisor inside a function body to advertise
Agent-Card skills.  That cycle was import-order dependent: importing the
supervisor first made the ``try/except`` swallow an ``ImportError`` and silently
disable A2A (``supervisor.delegate()`` then never consulted the remote
registry).

The fix moved ``Specialist`` into ``ai_assistant.domain.specialists`` and made
the supervisor take the remote lookup through dependency inversion
(:func:`orchestration.supervisor.set_remote_specialist_lookup`), with the
composition root installing the adapter's resolver.  These tests pin that
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

    def test_delegate_stays_local_without_an_installed_lookup(self) -> None:
        """No resolver installed → delegation never reaches for a remote."""
        with patch.object(supervisor, "_remote_specialist_lookup", None):
            delegation = supervisor.delegate("read the config", "s1", tool="read_file")
        self.assertIsNone(delegation.remote_endpoint)
        self.assertNotIn("A2A remote", delegation.reason)

    def test_delegate_consults_the_installed_lookup(self) -> None:
        """The composition-root-installed resolver drives remote routing."""
        with patch.object(supervisor, "_remote_specialist_lookup", lambda _sid: "http://remote.test"):
            delegation = supervisor.delegate("read the config", "s2", tool="read_file")
        self.assertEqual(delegation.remote_endpoint, "http://remote.test")
        self.assertIn("A2A remote: http://remote.test", delegation.reason)

    def test_delegate_fails_closed_when_the_lookup_raises(self) -> None:
        """A broken remote lookup must never break local delegation."""

        def _boom(_sid: str) -> str:
            raise RuntimeError("registry offline")

        with patch.object(supervisor, "_remote_specialist_lookup", _boom):
            delegation = supervisor.delegate("read the config", "s3", tool="read_file")
        self.assertIsNone(delegation.remote_endpoint)

    def test_adapter_resolver_matches_the_supervisor_port(self) -> None:
        """``resolve_remote_specialist`` satisfies the supervisor's port shape."""
        self.assertTrue(callable(a2a_protocol.resolve_remote_specialist))
        with patch.object(a2a_protocol, "a2a_available", return_value=False):
            self.assertIsNone(a2a_protocol.resolve_remote_specialist("code"))

    def test_agent_card_capabilities_still_cover_every_specialist(self) -> None:
        """The module-level domain import keeps the default card complete."""
        card = a2a_protocol.build_agent_card()
        advertised = {skill.id for skill in card.skills}
        self.assertEqual(advertised, {member.value for member in Specialist})


class CompositionRootWiresTheRemoteLookupTest(unittest.TestCase):
    """Dependency inversion only helps if the composition root installs it.

    Nothing else in the suite reads ``StartChatbotServer.py``, so deleting the
    install call would leave every test here green while A2A remote delegation
    silently stopped working — the resolver would exist, satisfy the port test
    above, and never be reachable.
    """

    SERVER = Path(__file__).resolve().parents[2] / "StartChatbotServer.py"

    def _install_calls(self) -> list[ast.Call]:
        tree = ast.parse(self.SERVER.read_text(encoding="utf-8"))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "set_remote_specialist_lookup"
        ]

    def test_startup_installs_the_adapter_resolver(self) -> None:
        """The server wires ``resolve_remote_specialist`` at startup."""
        calls = self._install_calls()
        self.assertTrue(calls, "StartChatbotServer.py never installs the remote-specialist lookup")
        installed = {
            arg.id
            for call in calls
            for arg in call.args
            if isinstance(arg, ast.Name)
        }
        self.assertIn("resolve_remote_specialist", installed)

    def test_shutdown_clears_the_lookup(self) -> None:
        """A restart must not reuse a resolver bound to a torn-down adapter."""
        self.assertTrue(
            any(
                len(call.args) == 1
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value is None
                for call in self._install_calls()
            ),
            "the server must clear the lookup on shutdown",
        )


if __name__ == "__main__":
    unittest.main()
