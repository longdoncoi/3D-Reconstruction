"""Desktop workflow specialist for Qt UI actions."""
from __future__ import annotations

from collections.abc import Callable


class ToolAppAgent:
    """Specialist responsible for desktop application action matching and dispatch."""

    def __init__(
        self,
        match_action: Callable[[str], dict | None],
        match_sequence: Callable[[str], list[dict] | None],
    ) -> None:
        self._match_action = match_action
        self._match_sequence = match_sequence

    def match(self, task: str) -> tuple[dict | None, list[dict] | None]:
        sequence = self._match_sequence(task)
        return (sequence[0] if sequence else self._match_action(task), sequence)

    @staticmethod
    def instruction() -> str:
        return (
            "Use application_action only after the planner selected a canonical action. "
            "Wait for the desktop acknowledgement before reporting completion."
        )


__all__ = ["ToolAppAgent"]
