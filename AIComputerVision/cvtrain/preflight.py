"""Preflight checks: fail fast with an actionable message instead of a traceback.

Everything the pipeline needs to be true *before* work starts is validated
here, whichever entry point produced the configuration: the environment and the
hyper-parameters themselves. A run that reaches the framework has already been
told its inputs are sane.

The module is deliberately an *aggregator* over two concerns kept elsewhere:

* :mod:`cvtrain.dataset` — the ``data.yaml`` document and the paths it names;
* :mod:`cvtrain.checks` — the :class:`Check`/:class:`PreflightReport` vocabulary
  every contributor shares.

It does not import them through each other, so a new kind of check can be added
without touching either neighbour.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Sequence

from .checks import Check, PreflightReport, module_available
from .config import (
    AUTO_BATCH,
    BASE_DIR,
    IMGSZ_STRIDE,
    MAX_EPOCHS,
    MAX_IMGSZ,
    MAX_OPSET,
    MAX_WORKERS,
    MODEL_REGISTRY,
    SUPPORTED_ROLES,
    TrainConfig,
)
from .dataset import check_data_yaml

__all__ = [
    "MAX_EPOCHS",
    "MAX_IMGSZ",
    "MAX_WORKERS",
    "MIN_IMGSZ",
    "MIN_OPSET",
    "MIN_PYTHON",
    "Check",
    "check_dependencies",
    "check_device",
    "check_hyperparameters",
    "check_metric_thresholds",
    "check_model_roles",
    "check_python",
    "check_weights",
    "run",
]

MIN_PYTHON = (3, 11)

MIN_IMGSZ = IMGSZ_STRIDE
"""Smallest image size YOLO can consume — its coarsest stride."""

MIN_OPSET = 11
"""Oldest opset this pipeline will request.

The floor is policy rather than a runtime limit: the shipped models were
exported at opset 12 (see ``Models/manifest.json``), and asking for something
far older produces a graph nobody has validated. The ceiling
(:data:`cvtrain.config.MAX_OPSET`) is the runtime contract.
"""

_DEVICE_TOKEN = re.compile(r"^(?:cpu|mps|metal|cuda(?::\d+)?|\d+)$")
"""One acceptable comma-separated component of ``--device``."""


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
        available = module_available(module)
        detail = "installed" if available else "missing"
        if not available and fatal:
            detail = f"missing - run: pip install -r {BASE_DIR / 'requirements.txt'}"
        checks.append(Check(f"dependency.{module}", available, detail, fatal=fatal))
    return checks


def check_model_roles(cfg: TrainConfig) -> list[Check]:
    """Every selected model must have a role the ONNX contract can export.

    ``onnx_contract.contract_for`` rejects an unknown role when the export is
    validated — which is *after* training, i.e. the most expensive possible
    moment to learn that a newly registered model family has no C++ parser and
    no tensor contract. The registry accepts custom roles by design; preflight
    is where "custom" becomes a fast, actionable failure instead of a wasted
    run. Unknown keys are reported by ``param.models`` and skipped here so the
    report stays complete.
    """

    supported = ", ".join(sorted(SUPPORTED_ROLES))
    checks: list[Check] = []
    for key in cfg.models:
        if key not in MODEL_REGISTRY:
            continue
        role = MODEL_REGISTRY[key].role
        ok = role in SUPPORTED_ROLES
        detail = role if ok else f"{role!r} is not supported (choose one of: {supported})"
        checks.append(Check(f"param.role.{key}", ok, detail))
    return checks


def check_hyperparameters(cfg: TrainConfig) -> list[Check]:
    """Validate the resolved hyper-parameters before any work starts.

    The dataset is checked exhaustively, so leaving the numbers unchecked would
    be an odd asymmetry: ``--epochs 0`` or ``--opset 99`` otherwise travels into
    the framework and dies there with a traceback, which is exactly what
    preflight exists to prevent. Every rule is stated in terms of the flag the
    operator used, so a ``FAIL`` line says what to change.
    """

    available = ", ".join(MODEL_REGISTRY.keys())
    unknown = [key for key in cfg.models if key not in MODEL_REGISTRY]
    checks = [
        Check(
            "param.models",
            bool(cfg.models) and not unknown,
            f"{', '.join(cfg.models) or 'none'} (at least one of {available} must be selected)"
            if not unknown
            else f"unknown model(s): {', '.join(unknown)} (choose from {available})",
        ),
        Check(
            "param.epochs",
            1 <= cfg.epochs <= MAX_EPOCHS,
            f"{cfg.epochs} (must be 1..{MAX_EPOCHS}; pass --epochs N)",
        ),
        Check(
            "param.imgsz",
            MIN_IMGSZ <= cfg.imgsz <= MAX_IMGSZ and cfg.imgsz % IMGSZ_STRIDE == 0,
            f"{cfg.imgsz} (must be {MIN_IMGSZ}..{MAX_IMGSZ} in steps of {IMGSZ_STRIDE}; pass --imgsz N)",
        ),
        Check(
            "param.batch",
            cfg.batch == AUTO_BATCH or cfg.batch >= 1,
            f"{cfg.batch} (must be >= 1, or {AUTO_BATCH} for the automatic batch size)",
        ),
        Check(
            "param.workers",
            0 <= cfg.workers <= MAX_WORKERS,
            f"{cfg.workers} (must be 0..{MAX_WORKERS}; pass --workers N)",
        ),
        Check(
            "param.opset",
            MIN_OPSET <= cfg.opset <= MAX_OPSET,
            f"{cfg.opset} (must be {MIN_OPSET}..{MAX_OPSET}; pass --opset N)",
        ),
        check_device(cfg),
    ]
    checks.extend(check_model_roles(cfg))
    checks.extend(check_metric_thresholds(cfg.min_metrics))
    return checks


def check_device(cfg: TrainConfig) -> Check:
    """Validate ``--device`` with the same fail-fast policy as every number.

    Left unchecked, a typo like ``--device gpu`` is only rejected deep inside
    the framework's device selector — after the environment checks have all
    passed and a traceback has replaced the actionable message preflight exists
    to deliver. The accepted grammar is the one the backend forwards verbatim:
    ``cpu``, ``mps``/``metal``, ``cuda``/``cuda:N``, a GPU index, any of those
    comma-separated, or empty/``auto`` for "let the framework choose".
    """

    raw = cfg.device
    if raw is None or not raw.strip():
        return Check("param.device", True, "auto (let the framework choose)")
    if raw.strip().lower() == "auto":
        return Check("param.device", True, "auto (let the framework choose)")
    tokens = [token.strip() for token in raw.split(",")]
    bad = [token for token in tokens if not _DEVICE_TOKEN.match(token.lower())]
    if bad:
        return Check(
            "param.device",
            False,
            f"{raw!r}: unsupported component(s) {', '.join(repr(token) for token in bad)}; "
            "use cpu, mps, cuda:N, a GPU index such as 0, or a list like 0,1",
        )
    return Check("param.device", True, raw)


def check_metric_thresholds(min_metrics: tuple[tuple[str, float], ...]) -> list[Check]:
    """One check per ``--min-metric`` threshold.

    Thresholds are validated here rather than only at the command line so that
    every entry point is held to the same rule: a threshold the pipeline cannot
    evaluate (an empty name, a ``nan``, a value outside the metric's range) must
    fail as a *configuration* error, not as a surprise halfway through training.
    Whether the backend actually reports that metric can only be known once it
    has run — that is the gate's job in :mod:`cvtrain.gate`. ``0..1`` covers
    the metrics the framework reports — precision, recall, mAP and fitness are
    all ratios; this gate is for higher-is-better metrics, not for losses.
    """

    checks: list[Check] = []
    for name, threshold in min_metrics:
        ok = bool(name) and math.isfinite(threshold) and 0.0 <= threshold <= 1.0
        detail = f"{name} >= {threshold} (finite, within 0..1)"
        if not ok:
            detail = f"{name!r}={threshold}: pass a metric name and a finite threshold within 0..1"
        checks.append(Check(f"param.min_metric.{name or '<unnamed>'}", ok, detail))
    return checks


def check_weights(cfg: TrainConfig) -> list[Check]:
    checks: list[Check] = []
    for key in cfg.models:
        if key not in MODEL_REGISTRY:
            # Unknown keys are reported by ``param.models``; skipping here keeps
            # the report complete instead of raising ``KeyError`` mid-preflight.
            continue
        spec = MODEL_REGISTRY[key]
        weights = cfg.weights_path(spec)
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
    for check in check_hyperparameters(cfg):
        report.add(check)
    for check in check_dependencies(fatal=deps_fatal, modules=modules):
        report.add(check)
    report.add(check_data_yaml(cfg.data_yaml))
    for check in check_weights(cfg):
        report.add(check)
    return report
