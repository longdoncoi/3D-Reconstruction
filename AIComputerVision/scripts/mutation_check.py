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
    python scripts/mutation_check.py --files cvtrain/fsio.py

``--files`` runs a *subset* of the sweep — the mutants of the named files
(``fnmatch`` patterns, comma separated). It exists for the platform jobs:
the Windows runner re-runs the mutants of the files whose decisions are
platform-sensitive (the advisory lock, the atomic publish, the retention
rules) so those guarantees are checked on the platform the product ships on.
A subset run never lowers the floor: the floor is checked against the whole
declared set, and the summary always says how many of the declared mutants
the run actually covered.

Scope: every decision the package's own documentation claims to protect — the
fail-closed provenance rules, the quality gate, the ONNX contract shared with
the Qt runtime, the publication order, the advisory lock, the progress protocol
and the exit-code taxonomy. A decision that is *not* listed here is a decision
nobody has pinned, and ``Docs/SCORECARD.md`` says so explicitly.
"""

from __future__ import annotations

import argparse
import fnmatch
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
#: architecture test reads, the requirement files the toolchain tripwire reads,
#: and the C++ sources the cross-language contract tests compare against. A
#: file the suite reads but the sandbox does not copy would turn its guard into
#: a skip — and a skip is not a verdict.
SANDBOX_PACKAGE = ("cvtrain", "tests")
SANDBOX_FILES = (
    "TrainModel.py",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
    "requirements-test.txt",
    "requirements-test.lock",
)
SANDBOX_WORKFLOWS = "workflows"

_IGNORED = shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak", ".mypy_cache", ".ruff_cache")


@dataclass(frozen=True)
class Mutant:
    """One deliberate logic inversion, with the behaviour it should break."""

    file: str
    original: str
    mutated: str
    label: str
    timeout_seconds: int = 120


#: The committed floor of the mutant count.
#:
#: Without a floor, *deleting* a mutant is indistinguishable from *passing* it:
#: the suite stays green, coverage stays at 100 % and the decision the mutant
#: was pinning silently stops being pinned. A new guard therefore adds its
#: mutant in the same commit, and a removed guard lowers this number in the
#: same commit — so the diff always shows the decision. (Same rule as the
#: Python service's ``MIN_MUTANTS``, ADR 0032, and the C++ gate's.)
#:
#: Measured 2026-10: 107 mutants, 107 killed, 0 survived, 0 stale.
MIN_MUTANTS = 107

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
    # ── What counts as a metric: the two sides of the gate must agree ───────
    Mutant(
        "cvtrain/metrics.py",
        "if isinstance(value, bool) or not isinstance(value, numbers.Real):",
        "if not isinstance(value, numbers.Real):",
        "a boolean metric counts as a measurement",
    ),
    Mutant(
        "cvtrain/metrics.py",
        "return number if math.isfinite(number) else None",
        "return number",
        "a non-finite metric is kept (NaN passes every threshold, Infinity is not JSON)",
    ),
    Mutant(
        "cvtrain/metrics.py",
        "except (OverflowError, TypeError, ValueError):",
        "except (TypeError, ValueError):",
        "a metric that overflows float crashes the gate instead of being unread",
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
    Mutant(
        "cvtrain/config.py",
        'return ",".join(part.strip().lower() for part in raw.split(","))',
        "return raw",
        "the device request is not normalised (two spellings of one request look different)",
    ),
    # ── Resuming an interrupted run ─────────────────────────────────────────
    Mutant(
        "cvtrain/backend.py",
        "        if not cfg.resume:\n            return TrainPlan(",
        "        if True:\n            return TrainPlan(",
        "--resume is ignored (an interrupted run starts over)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if resume_from is None:",
        "        if False:",
        "a missing checkpoint starts a fresh run instead of failing closed",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:",
        "        if False:",
        "a completed checkpoint is resumed (the framework asserts mid-run)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if plan.dataset_content_sha256 is None:",
        "        if False:",
        "a resume without a dataset fingerprint is accepted (unprovable data)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "            if recorded_digest is None or recorded_digest != plan.dataset_content_sha256:",
        "            if False:",
        "a checkpoint trained on different data is resumed behind the same run name",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if not checkpoint_data.is_file():",
        "        if False:",
        "a checkpoint whose dataset vanished is resumed on the invocation's --data",
    ),
    Mutant(
        "cvtrain/backend.py",
        "            source_weights=resume_from,\n            resume_from=resume_from,",
        "            source_weights=cfg.weights_path(spec),\n            resume_from=resume_from,",
        "the resumed run starts from the base checkpoint (resume=True then has nothing to continue)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "results = model.train(resume=True)",
        "results = model.train()",
        "the resumed run is not actually resumed (the checkpoint is ignored)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if plan.resume_from is not None:\n            claim_run_marker(plan.resume_from.parent.parent)",
        "        if False:\n            claim_run_marker(plan.resume_from.parent.parent)",
        "a resumed run keeps the dead process's lease (--prune-runs can delete a live run)",
    ),
    Mutant(
        "cvtrain/gate.py",
        "    if entry.get(\"resume_from\"):\n        return False",
        "    if False:\n        return False",
        "a resumed artifact is treated as reproducible by --skip-unchanged",
    ),
    Mutant(
        "cvtrain/gate.py",
        "    source = source_weights if source_weights is not None else cfg.weights_path(spec)",
        "    source = cfg.weights_path(spec)",
        "the provenance hashes the base checkpoint instead of the file training started from",
    ),
    Mutant(
        "cvtrain/backend.py",
        '    if cfg.device_resolved:\n        kwargs["device"] = cfg.device_resolved',
        '    if False:\n        kwargs["device"] = cfg.device_resolved',
        "the resolved device is not forwarded to the framework (it picks for itself)",
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
    Mutant(
        "cvtrain/preflight.py",
        "if bad:",
        "if False:",
        "an unsupported --device is accepted",
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
    Mutant(
        "cvtrain/dataset.py",
        "        if on_file is not None:\n            on_file(index, total)",
        "        if False:\n            on_file(index, total)",
        "the fingerprint reports no progress (a long hash looks like a hang)",
    ),
    Mutant(
        "cvtrain/dataset.py",
        'digest.update(b"<unreadable>")',
        "pass",
        "an unreadable file contributes nothing (the fingerprint stops noticing it)",
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
    Mutant(
        "cvtrain/onnx_contract.py",
        '    raise ExportContractError(\n        f"unsupported model role {role!r}: extend the ONNX contract (and the C++ "\n        "loader) before exporting this model family"\n    )',
        "    return RoleContract(outputs=(OutputSpec(rank=DETECTION_OUTPUT_RANK, channels=None),))",
        "an unknown model role is guessed instead of rejected",
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
    Mutant(
        "cvtrain/manifest.py",
        'payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\\n"',
        'payload = json.dumps(manifest, indent=2, sort_keys=True) + "\\n"',
        "the manifest writes NaN/Infinity (not valid JSON; strict readers reject it)",
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
    Mutant(
        "cvtrain/trainer.py",
        "    importable, import_msg = backend.check_importable(cfg)",
        "    importable, import_msg = True, \"framework importable\"",
        "a stack that cannot be imported passes the preflight",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "    if facts.epochs_completed is not None:\n        entry[\"epochs_completed\"] = facts.epochs_completed",
        "    if False:\n        entry[\"epochs_completed\"] = facts.epochs_completed",
        "the completed count is not recorded (100/100 and 40/100 look the same)",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "    if facts.resumed_from is not None:\n        entry[\"resume_from\"] = facts.resumed_from",
        "    if False:\n        entry[\"resume_from\"] = facts.resumed_from",
        "a resumed artifact loses its resume_from marker (it becomes skippable)",
    ),
    # ── The dataset band of the progress protocol ──────────────────────────
    Mutant(
        "cvtrain/progress.py",
        "fraction = (done / total) if total else 1.0",
        "fraction = 1.0",
        "the dataset band never moves (the bar stands still while hashing)",
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
        "if not _is_stale(lock, current_age, stale_after):",
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
        "        if age is not None and _is_stale(lock, age, stale_after):",
        "        if age is not None and False:",
        "a stale lock is never stolen (a crashed run wedges the directory forever)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        if age is not None and _is_stale(lock, age, stale_after):",
        "        if age is not None and age > stale_after:",
        "a dead same-host owner is only detected after the whole lease window",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        if alive is True:\n            return False",
        "        if False:\n            return False",
        "a live owner's expired lock is stolen (two writers enter Models/)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        if alive is False:\n            return True",
        "        if False:\n            return True",
        "a dead owner is not detected until the lease expires",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:\n        return None",
        "    if False:\n        return None",
        "an invalid lease PID is probed (a malformed marker is read as evidence)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        while not stop.wait(interval):\n            touch_lock(lock)",
        "        while False:\n            touch_lock(lock)",
        "the lease heartbeat never runs (a long epoch looks crashed)",
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
    Mutant(
        "cvtrain/fsio.py",
        '    except OSError as exc:\n        LOG.warning("Could not refresh lock %s: %s", lock, exc)',
        "    except OSError:\n        raise",
        "a lock that cannot be refreshed stops the run instead of warning",
    ),
    # ── Retention: the rules that keep the pruner safe ─────────────────────
    Mutant(
        "cvtrain/fsio.py",
        "            if index < keep:\n                continue",
        "            if False:\n                continue",
        "the newest runs are pruned too (--resume loses its checkpoint)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "            if age < RUN_ALIVE_SECONDS or _run_owner_alive(candidate):",
        "            if age < RUN_ALIVE_SECONDS:",
        "a run whose owner is alive is deleted during a long epoch",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "    return _process_alive(pid) is True",
        "    return False",
        "a live run owner is reported dead (the pruner deletes an active run)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "        if not marker.exists():\n            marker.write_text(",
        "        if True:\n            marker.write_text(",
        "refreshing the lease rewrites its owner (a marker can change hands)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        '        run_dir.mkdir(parents=True, exist_ok=True)\n        marker.write_text(_owner_line(), encoding="utf-8")',
        '        run_dir.mkdir(parents=True, exist_ok=True)\n        if marker.exists():\n            return\n        marker.write_text(_owner_line(), encoding="utf-8")',
        "claiming a resumed run leaves the dead owner's lease in place",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "    except OSError:\n        return time.time()",
        "    except OSError:\n        return 0.0",
        "an unreadable run is treated as the oldest (unknown age becomes a deletion)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "    marker = run_dir / RUN_MARKER_NAME\n    try:\n        newest = max(newest, marker.stat().st_mtime)\n    except OSError:\n        pass",
        "    marker = run_dir / RUN_MARKER_NAME",
        "the lease's freshness no longer counts as activity (a remote run looks dead)",
    ),
    Mutant(
        "cvtrain/fsio.py",
        "candidates = [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]",
        "candidates = [path for path in model_dir.iterdir() if path.is_dir()]",
        "a directory that is not a run is deleted too",
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
        'f"onnxslim={_package_version(\'onnxslim\')}",',
        'f"onnxslim=unknown",',
        "the exporter identity drops the onnxslim version (a simplified graph looks reproducible)",
    ),
    Mutant(
        "cvtrain/backend.py",
        'f"ultralytics={ultralytics_version}",',
        'f"ultralytics=unknown",',
        "the exporter identity drops the ultralytics version",
    ),
    # ── Effective run facts: the manifest records the experiment, not the ask ─
    Mutant(
        "cvtrain/backend.py",
        "        epochs = cls._effective_epochs(model)\n        if epochs is None:",
        "        epochs = None\n        if epochs is None:",
        "the provenance falls back to the request even when the framework reported a budget",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if completed is not None and completed >= 0:\n            completed += 1",
        "        if completed is not None and completed >= 0:\n            completed = completed",
        "the completed epoch count is off by one",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        except BackendUnavailableError as exc:\n            return False, str(exc)",
        "        except BackendUnavailableError as exc:\n            return True, str(exc)",
        "a broken install is reported as importable",
    ),
    Mutant(
        "cvtrain/backend.py",
        "                import_module(name)",
        "                pass",
        "a declared dependency is never imported (preflight proves only the framework)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "            except Exception as exc:",
        "            except ImportError as exc:",
        "a native DLL load error escapes preflight instead of failing it",
    ),
    # ── Device resolution: "auto" is a request, the resolution is the fact ──
    Mutant(
        "cvtrain/backend.py",
        "        requested = cfg.device_class\n        if requested != \"auto\":",
        "        requested = cfg.device_class\n        if False:",
        "'auto' is treated like an explicit device (never resolved)",
    ),
    Mutant(
        "cvtrain/backend.py",
        "        if torch.cuda.is_available():\n            return \"cuda:0\"",
        "        if False:\n            return \"cuda:0\"",
        "'auto' never resolves to a GPU",
    ),
    Mutant(
        "cvtrain/trainer.py",
        "cfg = replace(cfg, device_resolved=backend.resolve_device(cfg))",
        "cfg = cfg  # never resolved",
        "the run's device is never resolved (the provenance keeps a request)",
    ),
    Mutant(
        "cvtrain/gate.py",
        'if entry.get("device_resolved") != cfg.device_resolved:',
        "if False:",
        "the resolved device is not compared (a CPU run passes for a GPU run)",
    ),
    Mutant(
        "cvtrain/gate.py",
        '    if cfg.device_resolved is None:\n        return False\n    if entry.get("device_resolved") != cfg.device_resolved:',
        '    if entry.get("device_resolved") != cfg.device_resolved:',
        "an unresolved configuration claims a model is current (the device fact is never compared)",
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
        "cvtrain/cli.py",
        "if not separator or not name:",
        "if False:",
        "a malformed --min-metric name is accepted",
    ),
    Mutant(
        "cvtrain/cli.py",
        "if args.prune_runs is not None:",
        "if False:",
        "--prune-runs mode never runs (old runs are never cleaned up)",
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
    # The toolchain and CI tripwires read the workflows and the requirement
    # files; same rule — the sandbox has to hold what the suite reads.
    workflows = root / ".github" / SANDBOX_WORKFLOWS
    workflows.mkdir(parents=True, exist_ok=True)
    for workflow in sorted((REPO / ".github" / SANDBOX_WORKFLOWS).glob("*.yml")):
        shutil.copy2(workflow, workflows / workflow.name)
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


def floor_violation(mutant_count: int) -> str | None:
    """The reason the declared mutant count is below the floor, if it is."""
    if mutant_count < MIN_MUTANTS:
        return (
            f"{mutant_count} mutants declared, {MIN_MUTANTS} required — "
            "a guard was removed without a decision"
        )
    return None


def select_mutants(patterns: list[str]) -> list[Mutant]:
    """The mutants of the files named by ``patterns``.

    An empty pattern list selects everything. A pattern is an ``fnmatch``
    pattern matched against the mutant's file (``cvtrain/fsio.py`` selects the
    filesystem decisions; ``cvtrain/*`` selects the whole package).
    """

    if not patterns:
        return list(MUTANTS)
    return [mutant for mutant in MUTANTS if any(fnmatch.fnmatch(mutant.file, pattern) for pattern in patterns)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Invert one decision at a time and require the suite to fail.")
    parser.add_argument(
        "--files",
        default="",
        help="comma separated fnmatch patterns of the files to sweep (default: every declared mutant)",
    )
    args = parser.parse_args(argv)
    patterns = [pattern.strip() for pattern in args.files.split(",") if pattern.strip()]

    # Checked before the (expensive) sandbox run: a deleted mutant must fail
    # immediately, not after twenty minutes of green results. The floor is
    # checked against the *declared* set even for a subset run: a subset must
    # never be a way to lower it.
    violation = floor_violation(len(MUTANTS))
    if violation is not None:
        print(f"FLOOR -> {violation}")
        return 1

    selected = select_mutants(patterns)
    if not selected:
        print(f"No mutant matches {patterns}; nothing to check.")
        return 1

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
            for index, mutant in enumerate(selected, start=1):
                verdict = run_mutant(package, mutant)
                print(f"[{index:2d}/{len(selected)}] {verdict:9s} {mutant.file}: {mutant.label}")
                if verdict == "SURVIVED":
                    survivors.append(f"{mutant.file}: {mutant.label}")
                elif verdict == "STALE":
                    stale.append(f"{mutant.file}: {mutant.label}")
                elif verdict == "HUNG":
                    hung.append(f"{mutant.file}: {mutant.label}")
        finally:
            _release_lock()

        killed = len(selected) - len(survivors) - len(stale) - len(hung)
        coverage = "" if len(selected) == len(MUTANTS) else f" of {len(MUTANTS)} declared"
        print(
            f"\nMutants: {len(selected)}{coverage}  killed: {killed}  "
            f"survived: {len(survivors)}  hung: {len(hung)}  stale: {len(stale)}"
        )
        for entry in survivors:
            print(f"  SURVIVED -> {entry}")
        for entry in hung:
            print(f"  HUNG     -> {entry} (the mutant reached real work instead of failing fast)")
        for entry in stale:
            print(f"  STALE    -> {entry} (anchor no longer matches the source)")
        if len(MUTANTS) < MIN_MUTANTS:
            print(f"  FLOOR    -> {floor_violation(len(MUTANTS))}")
            return 1
        return 1 if survivors or stale or hung else 0
    finally:
        shutil.rmtree(sandbox_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
