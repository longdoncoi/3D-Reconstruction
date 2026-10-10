"""Preflight checks: fail fast with an actionable message instead of a traceback."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .config import BASE_DIR, MODEL_SPECS, TrainConfig

MIN_PYTHON = (3, 11)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = True

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


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_python() -> Check:
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    detail = f"{current[0]}.{current[1]} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)"
    return Check("python", ok, detail)


def check_dependencies(*, fatal: bool, modules: Sequence[str]) -> list[Check]:
    """Verify every module in ``modules`` is importable.

    ``modules`` is supplied by the caller from the active
    :class:`~cvtrain.backend.TrainingBackend`: preflight must not hard-code
    which ML framework a run is going to use, and a test double legitimately
    needs nothing installed at all. There is deliberately no default — an
    omitted argument would silently skip the whole dependency check.
    """

    checks: list[Check] = []
    for module in modules:
        available = _module_available(module)
        detail = "installed" if available else "missing"
        if not available and fatal:
            detail = f"missing - run: pip install -r {BASE_DIR / 'requirements.txt'}"
        checks.append(Check(f"dependency.{module}", available, detail, fatal=fatal))
    return checks


def check_data_yaml(path: Path) -> Check:
    if not path.exists():
        return Check(
            "dataset",
            False,
            f"not found: {path} - create it with 'train', 'val' and 'names' keys",
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return Check("dataset", False, f"unreadable: {path} ({exc})")

    if not text.strip():
        return Check("dataset", False, f"empty file: {path}")

    if not _module_available("yaml"):
        return Check("dataset", True, f"{path} found (PyYAML missing, content not validated)")

    import yaml

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return Check("dataset", False, f"invalid YAML: {exc}")

    if not isinstance(parsed, dict):
        return Check("dataset", False, "top level of data.yaml must be a mapping")
    missing = [key for key in ("names",) if key not in parsed]
    if "train" not in parsed and "val" not in parsed:
        missing.append("train or val")
    if missing:
        return Check("dataset", False, f"missing required key(s): {', '.join(missing)} in {path}")
    return Check("dataset", True, f"{path} ({len(parsed.get('names') or [])} classes)")


def check_weights(models: tuple[str, ...]) -> list[Check]:
    checks: list[Check] = []
    for key in models:
        spec = MODEL_SPECS[key]
        weights = spec.weights_path
        present = weights.exists()
        detail = "present" if present else f"{spec.ckpt} missing — ultralytics will download it (needs network)"
        # Missing base weights are recoverable (auto-download), so never fatal.
        checks.append(Check(f"weights.{key}", present, detail, fatal=False))
    return checks


def run(cfg: TrainConfig, *, deps_fatal: bool, modules: Sequence[str]) -> PreflightReport:
    """Run every preflight check for ``cfg`` and return the aggregate report.

    ``modules`` is the dependency set declared by the backend that will perform
    the run (see :func:`check_dependencies`).
    """

    report = PreflightReport()
    report.add(check_python())
    for check in check_dependencies(fatal=deps_fatal, modules=modules):
        report.add(check)
    report.add(check_data_yaml(cfg.data_yaml))
    for check in check_weights(cfg.models):
        report.add(check)
    return report
