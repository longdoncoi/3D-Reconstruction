"""Specialist role vocabulary shared by orchestration, adapters and evals.

``Specialist`` used to live in ``orchestration.supervisor``, which forced the
A2A adapter to import an orchestration module (and the supervisor to import the
adapter back) to enumerate the advertised skills — an import cycle that only
worked because both sides hid the import in a ``try/except`` or a function
body.  Moving the enum into the domain makes it a framework-free contract every
layer may depend on inward (ADR 0001/0002).
"""
from __future__ import annotations

from enum import StrEnum


class Specialist(StrEnum):
    """One worker role the supervisor can hand a task to."""

    SUPERVISOR = "supervisor"
    CHATBOT = "chatbot"
    TOOLAPP = "toolapp"
    RESEARCH = "research"
    WORKFLOW = "desktop_workflow"
    CODE = "code"
    VERIFICATION = "verification"


__all__ = ["Specialist"]
