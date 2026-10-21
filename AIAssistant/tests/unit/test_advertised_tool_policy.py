"""Drift guard: an advertised tool must have a policy — or be declared.

The supervisor's ``ALLOWED_TOOLS`` table is the least-privilege chokepoint:
a tool the model may call has to appear in the row of the specialist that
may run it, and ``authorise`` refuses everything else, fail-closed.

Two LSP tools were registered and advertised to the model without such a row
(ADR 0013 §6, corrected by ADR 0035), so every call to them was refused —
legitimate, but invisible to the next reader, and the refusal is what masked
a real defect in ``lsp_client`` for as long as it existed.

This guard makes the state explicit: every tool the model may see must
**either** have a policy row **or** be declared in
``supervisor.REGISTERED_NOT_ENABLED_TOOLS``. Enabling a tool therefore means
adding a row *and* removing the declaration, and forgetting the second half
fails the build.
"""
from __future__ import annotations

import json
import unittest

from ai_assistant.agents.completion import _TOOL_NAME_ALIASES
from ai_assistant.config.paths import get_paths
from ai_assistant.orchestration.supervisor import (
    ALLOWED_TOOLS,
    REGISTERED_NOT_ENABLED_TOOLS,
    Delegation,
    Specialist,
    authorise,
)


def _advertised_tools() -> set[str]:
    """The tool names the model is shown."""
    catalog = get_paths().project_root / "Config" / "agent_tools.json"
    payload = json.loads(catalog.read_text(encoding="utf-8"))
    entries = payload if isinstance(payload, list) else payload.get("tools", [])
    return {str(entry["name"]) for entry in entries if "name" in entry}


def _canonical(name: str) -> str:
    """The name the supervisor actually sees.

    A few advertised names are *aliases* — they are rewritten to their
    canonical tool before the call reaches the policy table
    (``agents/completion._TOOL_NAME_ALIASES``), so it is the canonical name
    that needs a row.
    """
    return _TOOL_NAME_ALIASES.get(name, name)


class AdvertisedToolPolicyTests(unittest.TestCase):
    def test_the_catalog_is_readable(self) -> None:
        self.assertGreater(len(_advertised_tools()), 10)

    def test_every_advertised_tool_has_a_policy_or_a_declaration(self) -> None:
        permitted = {tool for tools in ALLOWED_TOOLS.values() for tool in tools}
        undeclared = {
            _canonical(name)
            for name in _advertised_tools()
        } - permitted - REGISTERED_NOT_ENABLED_TOOLS
        self.assertEqual(
            undeclared, set(),
            "an advertised tool has no policy row and no entry in "
            "REGISTERED_NOT_ENABLED_TOOLS — either give it a row or declare it",
        )

    def test_a_declared_tool_is_really_refused(self) -> None:
        """A declaration must match the behaviour: the tool is refused.

        Otherwise the declaration would be a comment in the code rather than
        a statement about it.
        """
        for tool in sorted(REGISTERED_NOT_ENABLED_TOOLS):
            with self.subTest(tool=tool):
                delegation = Delegation(
                    specialist=Specialist.CODE,
                    tool=tool,
                    reason="drift guard",
                    idempotency_key="k1",
                )
                allowed, reason = authorise(delegation, needs_approval=False)
                self.assertFalse(allowed)
                self.assertTrue(reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
