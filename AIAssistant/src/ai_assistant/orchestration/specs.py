"""Specialist roles and instructions (no adapter dependencies)."""
from __future__ import annotations

from enum import StrEnum


class Specialist(StrEnum):
    SUPERVISOR = "supervisor"
    CHATBOT = "chatbot"
    TOOLAPP = "toolapp"
    RESEARCH = "research"
    WORKFLOW = "desktop_workflow"
    CODE = "code"
    VERIFICATION = "verification"


_SPECIALIST_INSTRUCTIONS = {
    Specialist.CHATBOT: "Answer conversationally using the Chatbot Agent and its RAG context only.",
    Specialist.TOOLAPP: "Dispatch only canonical Qt application_action calls and wait for acknowledgement.",
    Specialist.RESEARCH: "Inspect only the smallest relevant evidence; cite files and do not change state.",
    Specialist.WORKFLOW: "Dispatch only a canonical desktop action and wait for the desktop acknowledgement.",
    Specialist.CODE: "Describe the smallest safe change and require explicit approval before changing project state.",
    Specialist.VERIFICATION: "Independently check observable output; report a failed check instead of assuming success.",
    Specialist.SUPERVISOR: "Clarify intent, enforce policy, and coordinate the next specialist handoff.",
}

__all__ = ["_SPECIALIST_INSTRUCTIONS", "Specialist"]
