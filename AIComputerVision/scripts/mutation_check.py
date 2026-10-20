"""Mutation check for ``cvtrain``'s load-bearing decisions.

100% statement/branch coverage says a guard is *executed*, not that it is
*correct*: a test asserting the wrong direction still executes the line. This
script inverts one decision at a time in the pipeline's safety rules and runs
the suite:

    KILLED   — the suite fails: the decision is pinned by a test.
    SURVIVED — the suite stays green: a regression could land unnoticed.
    HUNG     — the suite did not finish: the mutant reached real work
               (a training run) instead of failing fast. Treated as a failure.
    STALE    — the anchor no longer matches the source: fail instead of
               silently shrinking the check.

**The suite runs in a sandbox copy of the tree, never in the working
directory.** That is not tidiness — it is the difference between a gate and a
hazard. A mutant that makes a command fall through to the training path would
otherwise train against the machine's real dataset and checkpoints, writing
into the real ``Models/`` directory and burning a GPU. In the sandbox there is
no dataset, no checkpoint and no ``Models/``, so such a mutant fails fast in
preflight instead. The working tree is never written to at all; the sandbox is
discarded when the run ends.

A lock file refuses overlapping runs and, after a crash, blocks the next run
until the tree is verified and the lock is removed by hand. Run it *after* the
test/coverage step.

Usage (from ``AIComputerVision/``)::

    python scripts/mutation_check.py

Scope: every decision the package's own documentation claims to protect — the
fail-closed provenance rules, the quality gate, the ONNX contract shared with
the Qt runtime, the publication order, the advisory lock, the progress protocol
and the exit-code taxonomy. A decision that is *not* listed here is a decision
nobody has pinned, and ``Docs/SCORECARD.md`` says so explicitly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parent
LOCK_PATH = BASE / "scripts" / ".mutation_check.lock"

#: Files the sandbox needs so the suite behaves exactly as it does in the
#: repository: the package, its tests, the entry point, the README the
#: architecture test reads, and the C++ sources the cross-language contract
#: tests compare against.
SANDBOX_PACKAGE = ("cvtrain", "tests")
SANDBOX_FILES = ("TrainModel.py", "pyproject.toml", "README.md")

_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak", ".mypy_cache", ".ruff_cache")


@dataclass(frozen=True)
class Mutant:
    """One deliberate logic inversion, with the behaviour it should break."""

    file: str
    original: str
    mutated: str
    label: str
    timeout_seconds: int = 120


MUTANTS: tuple[Mutant, ...] = (
    # ── Provenance: the skip rule is fail-closed, field by field ───────────
    Mutant(
        "cvtrain/gate.py",
        "if not isinstance(entry_content, str) or not entry_content or entry_content != recorded_content:",
        "if False:",
        "missing dataset_content_sha256 accepted by --skip-unchanged (fail-open)",
    ),
    Mutant(
        "cvtrain/gate.py",
        'return weights_sha is not None and entry.get("source_weights_sha256") == weights_sha',
        "return True",
        "base-checkpoint digest removed from --skip-unchanged",
    ),
    Mutant(
        "cvtrain/gate.py",
        "if recorded != requested:",
        "if False:",
        "hyper-parameter comparison removed from --skip-unchanged",
    ),
    Mutant(
        "cvtrain/gate.py",
        'if recorded_dataset is None or entry.get("dataset_sha256") != recorded_dataset:',
        "if False:",
        "dataset definition digest removed from --skip-unchanged",
    ),
    Mutant(
        "cvtrain/gate.py",
        'if entry.get("sha256") != sha256_file(target):',
        "if False:",
        "published artifact digest removed from --skip-unchanged",
    ),
    # ── Quality gate: a bar that is not met is a refusal ───────────────────
    Mutant(
        "cvtrain/gate.py",
        "if actual < minimum:",
        "if actual > minimum:",
        "quality gate direction inverted",
    ),
    Mutant(
        "cvtrain/gate.py",
        "if actual is None:",
        "if False:",
        "a metric the backend never reported counts as a pass",
    ),
    Mutant(
        "cvtrain/gate.py",
        "if isinstance(value, bool) or not isinstance(value, numbers.Real):",
        "if not isinstance(value, numbers.Real):",
        "a boolean metric counts as a measurement",
    ),
    # ── Model selection: a typo must not silently train the defaults ───────
    Mutant(
        "cvtrain/config.py",
        "if unknown:",
        "if False:",
        "unknown model key silently resolves to an empty/default selection",
    ),
    Mutant(
        "cvtrain/config.py",
        "defaults = tuple(key for key in DEFAULT_MODELS if key in reg)",
        "defaults = tuple(reg.keys())",
        "an empty selection trains every registered model",
    ),
    Mutant(
        "cvtrain/config.py",
        "if not isinstance(spec, ModelSpec):",
        "if False:",
        "the registry accepts a non-ModelSpec",
    ),
    # ── Preflight: fail before the expensive work starts ────────────────────
    Mutant(
        "cvtrain/preflight.py",
        "ok = role in SUPPORTED_ROLES",
        "ok = True",
        "unsupported model role reaches training/export",
    ),
    Mutant(
        "cvtrain/preflight.py",
        "if not available and fatal:",
        "if False:",
        "a missing dependency is never fatal",
    ),
    Mutant(
        "cvtrain/preflight.py",
        "MIN_OPSET <= cfg.opset <= MAX_OPSET",
        "True",
        "the ONNX opset window is not enforced",
    ),
    Mutant(
        "cvtrain/preflight.py",
        "1 <= cfg.epochs <= MAX_EPOCHS",
        "True",
        "the epoch ceiling is not enforced",
    ),
    Mutant(
        "cvtrain/preflight.py",
        "ok = bool(name) and math.isfinite(threshold) and 0.0 <= threshold <= 1.0",
        "ok = bool(name)",
        "a NaN or out-of-range metric threshold is accepted",
    ),
    # ── Dataset: the fingerprint covers everything the YAML reaches ────────
    Mutant(
        "cvtrain/dataset.py",
        '    raw = Path(value).expanduser()\n    target = raw if raw.is_absolute() else base / raw\n    if any(char in value for char in "*?["):',
        '    raw = Path(value).expanduser()\n    target = raw if raw.is_absolute() else base / raw\n    if False:',
        "glob split treated as a literal path (matching file changes invisible)",
    ),
    Mutant(
        "cvtrain/dataset.py",
        'if target.suffix.lower() == ".txt":',
        "if False:",
        "txt listing contents not fingerprinted (listed images invisible)",
    ),
    Mutant(
        "cvtrain/dataset.py",
        '    root = parsed.get("path")',
        "    root = None",
        "the top-level 'path:' key of data.yaml is ignored",
    ),
    Mutant(
        "cvtrain/dataset.py",
        'missing = [key for key in ("train", "val", "names") if key not in parsed]',
        "missing = []",
        "the required data.yaml keys (train/val/names) are not enforced",
    ),
    # ── ONNX contract: the contract the Qt loader depends on ───────────────
    Mutant(
        "cvtrain/onnx_contract.py",
        "if len(dims) != INPUT_RANK:",
        "if False:",
        "the NCHW input-rank check is disabled",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        "elif extent != imgsz:",
        "elif False:",
        "the exported spatial size is not matched against --imgsz",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        "if match.channels is not None and dims[1] not in (0, match.channels):",
        "if False:",
        "the output channel-count check is disabled",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        "if value_info.type.tensor_type.elem_type != _FLOAT32:",
        "if False:",
        "the float32 dtype check is disabled",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        '    if not module_available("onnxruntime"):',
        "    if False:",
        "the ONNX Runtime smoke inference is skipped",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        'if opset.domain in ("", "ai.onnx") and opset.version > MAX_OPSET:',
        "if False:",
        "the opset ceiling of the pinned ONNX Runtime is not enforced",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        "    outputs = list(model.graph.output)\n    if len(outputs) != len(role_contract.outputs):",
        "    outputs = list(model.graph.output)\n    if False:",
        "the graph output count is not checked (segmentation needs two)",
    ),
    Mutant(
        "cvtrain/onnx_contract.py",
        '    if role in ("detection", "tracking"):',
        '    if role == "detection":',
        "the tracking role loses its tensor contract",
    ),
    # ── Manifest: provenance and integrity ─────────────────────────────────
    Mutant(
        "cvtrain/manifest.py",
        "if not (models_dir / filename).exists():",
        "if False:",
        "deleted artifact entry kept after --write-manifest (ghost provenance)",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "if only is None:",
        "if True:",
        "the verification scope filter (only=) is broken",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "if version > SCHEMA_VERSION:",
        "if False:",
        "a newer manifest schema is down-stamped instead of read as-is",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "if not all(isinstance(entry, dict) for entry in entries.values()):",
        "if False:",
        "a non-map manifest entry is accepted (no quarantine)",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "if not expected:",
        "if False:",
        "an entry without a sha256 is accepted by verification",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "if actual != expected:",
        "if False:",
        "a sha256 mismatch is accepted by verification",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "data = _MIGRATIONS[step](data)",
        "data = data",
        "the schema migration chain is skipped",
    ),
    Mutant(
        "cvtrain/manifest.py",
        "        os.replace(path, target)",
        "        pass",
        "a corrupt manifest is not quarantined (the next write erases it)",
    ),
    # ── Trainer: the order of the steps is the contract ────────────────────
    Mutant(
        "cvtrain/trainer.py",
        "onnx_contract.validate_export(generated, spec, cfg)",
        "pass  # validation skipped",
        "the export contract is not checked before publishing",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "target = atomic_replace_with_backup(generated, cfg.output_path(spec))",
        "target = cfg.output_path(spec)",
        "the model is never published to Models/",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "    manifest_mod.save_manifest(cfg.models_dir, manifest)",
        "    pass",
        "the per-model provenance flush is removed (a crash loses every entry)",
    ),
    Mutant(
        "cvtrain/trainer.py",
        'gate.enforce_quality_gate(cfg, spec, kept.get("metrics"))',
        "pass",
        "a skipped model bypasses the quality gate",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "gate.enforce_quality_gate(cfg, spec, model.metrics)",
        "pass",
        "a freshly trained model bypasses the quality gate",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "reporter.done()",
        "pass",
        "the success path never emits the 'done' marker (the Qt dock waits forever)",
    ),
    # ── Progress protocol: the wire contract with the Qt dock ──────────────
    Mutant(
        "cvtrain/progress.py",
        "if clamped < self._last_pct:",
        "if False:",
        "the monotonic progress guard is disabled (the bar can rewind)",
    ),
    # ── Filesystem: durability and lock ownership ──────────────────────────
    Mutant(
        "cvtrain/fsio.py",
        "if age is not None and age > STEAL_STALE_SECONDS:",
        "if True:",
        "fresh steal claim removed (two processes can race the same stale lock)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "if current_age <= stale_after:",
        "if False:",
        "lock that turned fresh while stealing is deleted from under its owner",
    ),
    Mutant(
        "cvtrain/fsio.py",
        'if f"token={expected_token}" not in content:',
        "if False:",
        "the lock ownership token is not checked (another run's lock is released)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        if age is not None and age > stale_after:",
        "        if False:",
        "a stale lock is never stolen (a crashed run wedges the directory forever)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "if dst.exists():",
        "if False:",
        "no .bak rollback copy is kept when an artifact is replaced",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "os.replace(tmp_name, path)",
        "pass",
        "the atomic manifest write degenerates into a non-atomic one",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "os.replace(tmp_name, dst)",
        "pass",
        "the atomic publish degenerates into a non-atomic one",
    ),
    # ── Backend: the seam and the exporter identity ────────────────────────
    Mutant(
        "cvtrain/backend.py",
        'f"onnx={_package_version(\'onnx\')}",',
        'f"onnx=unknown",',
        "the exporter identity drops the onnx version",
    ),
    Mutant(
        "cvtrain/backend.py",
        'f"ultralytics={ultralytics_version}",',
        'f"ultralytics=unknown",',
        "the exporter identity drops the ultralytics version",
    ),
    Mutant(
        "cvtrain/backend.py",
        "if isinstance(value, numbers.Real):",
        "if True:",
        "metrics_from keeps non-numeric values (the manifest becomes unreadable)",
    ),
    # ── CLI and exit codes: the contract with the desktop app ──────────────
    Mutant(
        "cvtrain/cli.py",
        "if args.write_manifest:",
        "if False:",
        "--write-manifest mode never runs",
    ),
    Mutant(
        "cvtrain/cli.py",
        "if args.verify_manifest:",
        "if False:",
        "--verify-manifest mode never runs",
    ),
    Mutant(
        "cvtrain/errors.py",
        "exit_code: ExitCode = ExitCode.QUALITY",
        "exit_code: ExitCode = ExitCode.PREFLIGHT",
        "a quality-gate failure reports the preflight exit code",
    ),
    Mutant(
        "cvtrain/errors.py",
        "exit_code: ExitCode = ExitCode.EXPORT_CONTRACT",
        "exit_code: ExitCode = ExitCode.FAILURE",
        "an export-contract failure reports the wrong exit code",
    ),
)


def _build_sandbox(root: Path) -> Path:
    """Copy the tree the suite needs into ``root`` and return the package dir.

    The sandbox has no ``Dataset/``, no checkpoints and no ``Models/``: a
    mutant that reaches the training path therefore fails in preflight
    instead of starting a real run.
    """

    package = root / "AIComputerVision"
    package.mkdir(parents=True, exist_ok=True)
    for name in SANDBOX_PACKAGE:
        shutil.copytree(BASE / name, package / name, ignore=_IGNORED)
    for name in SANDBOX_FILES:
        shutil.copy2(BASE / name, package / name)
    # The cross-language contract tests read the C++ sources; without them
    # they would skip, and a skip is not a verdict.
    shutil.copytree(REPO / "src", root / "src", ignore=_IGNORED)
    shutil.copy2(REPO / "CMakeLists.txt", root / "CMakeLists.txt")
    return package


def _run_suite(package: Path, timeout: int) -> int | None:
    """Run the suite in the sandbox.

    Returns the process exit code, or ``None`` when the suite did not finish
    within ``timeout`` seconds (a mutant that reached real work).
    """

    try:
        outcome = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=package,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    return outcome.returncode


def _acquire_lock() -> None:
    try:
        handle = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"mutation_check: {LOCK_PATH} exists — another run is active, or a previous "
            "run died mid-mutation. Verify the tree (git diff), restore the source if "
            "needed, then delete the lock. Never run this alongside the test suite."
        ) from None
    os.write(handle, str(os.getpid()).encode("ascii"))
    os.close(handle)


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def _encode(anchor: str, newline: str) -> bytes:
    """Encode an anchor with the file's own line ending.

    The mutants are written in the source with ``\\n``; the files on disk are
    CRLF on Windows. Matching on *bytes* keeps the anchor independent of the
    checkout's line endings and makes the restore byte-for-byte exact.
    """

    return anchor.replace("\n", newline).encode("utf-8")


def run_mutant(package: Path, mutant: Mutant) -> str:
    """Apply one mutant in the sandbox, run the suite, restore the file."""

    target = package / mutant.file
    original = target.read_bytes()
    newline = "\r\n" if b"\r\n" in original else "\n"
    anchor = _encode(mutant.original, newline)
    replacement = _encode(mutant.mutated, newline)
    if original.count(anchor) != 1:
        print(f"  STALE {mutant.label}: anchor not unique/absent in {mutant.file}")
        return "STALE"
    try:
        target.write_bytes(original.replace(anchor, replacement))
        code = _run_suite(package, mutant.timeout_seconds)
    finally:
        target.write_bytes(original)
        if target.read_bytes() != original:
            raise RuntimeError(f"{mutant.file} was not restored byte-for-byte")
    if code is None:
        print(f"  the suite did not finish within {mutant.timeout_seconds}s")
        return "HUNG"
    return "KILLED" if code != 0 else "SURVIVED"


def main() -> int:
    print("Baseline: the unmutated suite must be green")
    sandbox_root = Path(tempfile.mkdtemp(prefix="cvtrain-mutation-"))
    try:
        package = _build_sandbox(sandbox_root)
        baseline = _run_suite(package, 300)
        if baseline is None or baseline != 0:
            print("Baseline failed; refusing to attribute later failures to mutants.")
            return 1

        survivors: list[str] = []
        stale: list[str] = []
        hung: list[str] = []
        _acquire_lock()
        try:
            for index, mutant in enumerate(MUTANTS, start=1):
                verdict = run_mutant(package, mutant)
                print(f"[{index:2d}/{len(MUTANTS)}] {verdict:9s} {mutant.file}: {mutant.label}")
                if verdict == "SURVIVED":
                    survivors.append(f"{mutant.file}: {mutant.label}")
                elif verdict == "STALE":
                    stale.append(f"{mutant.file}: {mutant.label}")
                elif verdict == "HUNG":
                    hung.append(f"{mutant.file}: {mutant.label}")
        finally:
            _release_lock()

        killed = len(MUTANTS) - len(survivors) - len(stale) - len(hung)
        print(
            f"\nMutants: {len(MUTANTS)}  killed: {killed}  "
            f"survived: {len(survivors)}  hung: {len(hung)}  stale: {len(stale)}"
        )
        for entry in survivors:
            print(f"  SURVIVED -> {entry}")
        for entry in hung:
            print(f"  HUNG     -> {entry} (the mutant reached real work instead of failing fast)")
        for entry in stale:
            print(f"  STALE    -> {entry} (anchor no longer matches the source)")
        return 1 if survivors or stale or hung else 0
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
