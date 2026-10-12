"""Shared vocabulary of the fail-fast validation layer.

Every check the pipeline runs before work starts — environment, dataset,
hyper-parameters — is expressed as a :class:`Check`, and the checks of one run
are collected into a :class:`PreflightReport`. Keeping the type in its own leaf
module is what lets :mod:`cvtrain.dataset` and :mod:`cvtrain.preflight` both
contribute checks without importing each other (which would be circular).

A check never raises: it records whether it passed, how fatal a failure is, and
a detail written in terms of what the operator can change.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

__all__ = ["Check", "PreflightReport", "module_available"]


@dataclass
class Check:
    """One validation result: a name, a verdict, and an actionable detail."""

    name: str
    ok: bool
    detail: str
    fatal: bool = True
    """A failing non-fatal check is a warning: the run may proceed anyway."""

    def line(self) -> str:
        if self.ok:
            status = "OK  "
        elif self.fatal:
            status = "FAIL"
        else:
            status = "WARN"
        return f"[{status}] {self.name}: {self.detail}"


@dataclass
class PreflightReport:
    """The aggregate of every check run for one invocation."""

    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok or not check.fatal for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.ok and check.fatal]

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def lines(self) -> list[str]:
        return [check.line() for check in self.checks]


def module_available(name: str) -> bool:
    """Whether ``name`` can be imported, without importing it.

    ``find_spec`` raises ``ImportError``/``ValueError`` for half-initialised
    ``sys.modules`` entries; a probe that crashed would take the whole report
    down with it, so those are reported as "missing".
    """

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
