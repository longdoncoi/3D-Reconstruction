"""Model manifest: provenance + integrity for the exported ONNX artifacts.

Every exported model gets an entry describing which base weights, dataset and
hyper-parameters produced it, plus a SHA-256 digest so ``--verify-manifest``
(or a release auditor) can detect a truncated or replaced file. The Qt app
does not read this document — it loads the ONNX files the manifest accounts
for; checking them is this package's job.

The document shape is expressed as :class:`Manifest`/:class:`ManifestEntry`
``TypedDict``s so that a typo in an entry key is a type error rather than
provenance that silently never gets read back.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable, Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from . import generator_name
from .config import MANIFEST_NAME, MODEL_REGISTRY, TRACKER_INFO
from .errors import ManifestIntegrityError
from .fsio import atomic_write_text, file_size, sha256_file

__all__ = [
    "SCHEMA_VERSION",
    "LastRun",
    "Manifest",
    "ManifestEntry",
    "build_from_scan",
    "load_manifest",
    "migrate",
    "now_utc_iso",
    "resolve_git_commit",
    "save_manifest",
    "upsert_entry",
    "verify_manifest",
    "verify_or_raise",
]

LOG = logging.getLogger("cvtrain.manifest")

SCHEMA_VERSION = 1
"""Version of the document shape this build writes. See :func:`migrate`."""

# Adding optional entry fields (batch, workers, backend_version,
# source_weights_sha256, device, dataset_sha256, dataset_content_sha256)
# deliberately does *not* bump
# this: ``ManifestEntry`` is ``total=False``, so their absence is a shape this
# build already knows how to read, and the consumer that cares —
# ``gate.already_current`` — treats a missing value as "provenance we do not
# have" and fails closed by retraining. A bump is reserved for a change that
# makes an *old* document unreadable or actively misleading to a new reader; a
# migration then has to say what a reader of the old shape was allowed to
# assume.


class ManifestEntry(TypedDict, total=False):
    """One exported artifact: what produced it and how to verify it.

    ``total=False`` because entries created by ``--write-manifest`` (a bare
    disk scan) legitimately carry only identity + digest, while entries written
    after a real training run carry the full provenance.
    """

    role: str
    """Semantic role, mirrors ``ModelSpec.role`` (detection/segmentation/...)."""

    source_weights: str
    """Base checkpoint the model was trained from."""

    sha256: str
    """Hex digest of the exported file; the integrity contract."""

    size_bytes: int

    epochs: int
    imgsz: int
    seed: int
    opset: int
    batch: int
    workers: int

    backend_version: str
    """Composite exporter identity of the stack that produced this artifact.

    Names ``cvtrain``, ``ultralytics``, ``torch`` and ``onnx``; see
    :func:`cvtrain.backend.exporter_identity`. Recorded *per entry* (the
    document-level ``ultralytics`` field describes only the last run) because
    :func:`cvtrain.gate.already_current` has to know which exporter wrote the
    file it is being asked to trust — an Ultralytics upgrade is not the only
    dependency that can change the emitted graph."""

    device: str
    """Compute device class the artifact was trained on (``auto`` when unset).

    CPU and CUDA runs of the same hyper-parameters are different artifacts
    behind the same file name, so this is one of the factors the skip rule
    compares rather than an interesting fact nobody checks."""

    dataset_sha256: str
    """Digest of ``data.yaml`` captured *before* the artifact was trained.

    The path is not provenance: the YAML defines the classes, the splits and
    the image locations, so an edited file behind an unchanged name is a
    different experiment. Absent when the file could not be read before
    training, which makes ``--skip-unchanged`` retrain rather than assume it
    matches."""

    dataset_content_sha256: str
    """Digest of the dataset contents captured before training.

    Covers the bytes and relative names reachable from every ``train``/``val``
    reference shape (directory, glob, single file, ``.txt`` listing, list).
    ``gate.already_current`` requires it: an entry without it is unproven, not
    automatically current."""

    source_weights_sha256: str
    """Digest of the base checkpoint the model was actually trained from.

    The file name alone is not provenance: ``yolo11x.pt`` can be replaced or
    re-downloaded, and a different starting point produces a different model.
    Absent when the checkpoint was not on disk while recording, which makes
    ``--skip-unchanged`` retrain rather than assume it matches."""

    trained_at: str | None
    """UTC ISO timestamp, or ``None`` when the file was adopted from disk."""

    scanned_at: str
    """UTC ISO timestamp of the last ``--write-manifest`` scan."""

    dataset: str
    run_dir: str
    metrics: dict[str, float]

    tracker: dict[str, object]
    """Association metadata for a ``tracking`` artifact.

    The ONNX graph is a detector; the tracker (algorithm, IoU threshold,
    runtime owner) lives in C++. Recording it here makes the entry's ``role``
    honest and lets an auditor reproduce the association behaviour without
    inspecting the Qt sources. Absent for detection/segmentation entries.
    """


class LastRun(TypedDict, total=False):
    """Header describing the most recent training invocation."""

    started_at: str
    dataset: str
    models: list[str]
    epochs: int
    imgsz: int
    seed: int
    opset: int
    batch: int
    workers: int
    device: str
    python: str
    """Interpreter that drove the run (``major.minor.micro``), for audit.

    Informational only: :func:`cvtrain.gate.already_current` does not compare
    it, so a Python upgrade never triggers a retrain by itself — the backend
    version it gates on already captures export-relevant drift.
    """

    git_commit: str
    """Commit SHA of the repository when the run started ('unversioned' if unavailable)."""

    min_metrics: list[list[str | float]]
    """Quality thresholds of the invocation, as ``[[name, minimum], ...]``.

    Lists, not tuples: the manifest is JSON, and a tuple would come back as a
    list on the next load. Informational, like ``python``.
    """


class Manifest(TypedDict, total=False):
    """The ``Models/manifest.json`` document."""

    schema_version: int
    generator: str
    entries: dict[str, ManifestEntry]
    last_run: LastRun
    ultralytics: str
    """Composite exporter identity of the last run (key is historical).

    Despite the name, the value is the full ``cvtrain``/``ultralytics``/
    ``torch``/``onnx`` identity written by the backend. Per-entry values live
    in ``backend_version``; this header field only describes the latest run.
    """
    updated_at: str


def resolve_git_commit(cwd: Path) -> str:
    """Best-effort resolution of the git commit SHA for audit provenance.

    Returns the 40-character hex SHA if cwd is inside a git repository,
    or 'unversioned' otherwise. Never raises.
    """

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res.returncode == 0:
            commit = res.stdout.strip()
            if len(commit) == 40 and all(c in "0123456789abcdefABCDEF" for c in commit):
                return commit.lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


def now_utc_iso() -> str:
    """Timestamp for provenance fields: UTC, second precision, offset-suffixed."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _skeleton() -> Manifest:
    return {"schema_version": SCHEMA_VERSION, "generator": generator_name(), "entries": {}}


Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""Upgrades a document from schema ``N`` to ``N + 1``; mutates and returns it."""


def _migrate_0_to_1(data: dict[str, Any]) -> dict[str, Any]:
    """Schema 0 -> 1: documents written before the field was versioned.

    They carry no ``schema_version`` and may carry no ``generator``. The entry
    map itself is already validated by :func:`load_manifest`, so the shapes of
    ``entries`` are unchanged by this step.
    """

    data.setdefault("generator", generator_name())
    return data


#: Ordered by the schema version they consume: ``_MIGRATIONS[0]`` yields v1.
#: Adding a schema bump means adding one entry here plus one test — the chain is
#: walked by :func:`migrate`, never by ad-hoc ``setdefault`` calls at call sites.
_MIGRATIONS: dict[int, Migration] = {0: _migrate_0_to_1}


def migrate(data: dict[str, Any], path: Path) -> Manifest:
    """Bring a loaded document up to :data:`SCHEMA_VERSION`.

    Two rules, both deliberate:

    * a document **older** than this build (schema ``0``: written before the
      field existed) is upgraded step by step through :data:`_MIGRATIONS`, so a
      caller can never observe a document whose shape it has not agreed to read;
    * a document **newer** than this build is returned untouched and is
      explicitly *not* down-stamped. Overwriting its header with the old version
      number would silently advertise a schema this build does not actually
      understand, which is how a future reader loses its own migration hook.

    ``version`` is also coerced to ``>= 0``: a garbage header is treated as
    "unversioned" rather than making the chain index a negative key.
    """

    raw = data.get("schema_version")
    version = max(0, raw) if isinstance(raw, int) else 0
    if version > SCHEMA_VERSION:
        LOG.warning(
            "Manifest %s declares schema %d but this build writes %d; reading it as-is (not down-stamping)",
            path,
            version,
            SCHEMA_VERSION,
        )
        return cast(Manifest, data)
    for step in range(version, SCHEMA_VERSION):
        data = _MIGRATIONS[step](data)
    data["schema_version"] = SCHEMA_VERSION
    return cast(Manifest, data)


def _quarantine(path: Path) -> None:
    """Move an unreadable manifest aside so a later write cannot erase it.

    Best effort and non-fatal: on a read-only volume (or a directory sitting
    where the file should be) the rename itself fails, and the corruption is
    still reported by the caller's warning. The point is only that the evidence
    of *what was on disk* survives the next ``save_manifest``; the pipeline still
    fails closed by returning a skeleton.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    target = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        os.replace(path, target)
    except OSError as exc:
        LOG.warning("Could not quarantine %s: %s", path, exc)
    else:
        LOG.warning("Quarantined unreadable manifest to %s", target)


def load_manifest(models_dir: Path, *, strict: bool = False) -> Manifest:
    """Return the existing manifest migrated to the current schema, or a fresh
    skeleton when the file is absent, unreadable or structurally invalid.

    A corrupt or structurally invalid document is *quarantined* (renamed aside)
    before the skeleton is returned, so the next ``save_manifest`` cannot
    silently overwrite the only record of what was there. A missing file is not
    an error and is not quarantined.

    ``strict=True`` is the *verdict* mode ``--verify-manifest`` reads in. The
    same problems raise :class:`~cvtrain.errors.ManifestIntegrityError`
    (exit ``2``) and nothing is renamed: recovery is the right answer for a run
    that rebuilds provenance as it goes, but for a read-only verdict it would
    mean reporting "Manifest OK" over a document this very call had set aside —
    the one outcome :mod:`cvtrain.errors` promises the pipeline never produces.
    Quarantining here would also make a command documented as *reading*
    ``Models/`` silently rewrite it. A missing file is not an error even in
    strict mode: whether there is anything to verify is the caller's call.
    """

    path = models_dir / MANIFEST_NAME
    if not path.exists():
        return _skeleton()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if strict:
            raise ManifestIntegrityError(f"Manifest corrupt: {path} ({exc})") from exc
        LOG.warning("Corrupt manifest %s: %s", path, exc)
        _quarantine(path)
        return _skeleton()
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        if strict:
            raise ManifestIntegrityError(f"Manifest invalid: {path} (missing entries dict)")
        LOG.warning("Manifest %s is invalid (missing entries dict)", path)
        _quarantine(path)
        return _skeleton()
    entries = cast(dict[str, Any], data)["entries"]
    if not all(isinstance(entry, dict) for entry in entries.values()):
        # Every reader below is typed against ``ManifestEntry``: a scalar where
        # an entry is expected would otherwise surface as ``AttributeError``
        # halfway through verification (exit 1 + traceback). Same verdict as a
        # document with no entries at all — quarantined, then rebuilt.
        if strict:
            raise ManifestIntegrityError(f"Manifest invalid: {path} (an entry is not a map)")
        LOG.warning("Manifest %s is invalid (an entry is not a map)", path)
        _quarantine(path)
        return _skeleton()
    return migrate(cast(dict[str, Any], data), path)


def save_manifest(models_dir: Path, manifest: Manifest) -> Path:
    """Atomically write the manifest so a crash cannot leave it half-written."""

    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    return atomic_write_text(models_dir / MANIFEST_NAME, payload)


def upsert_entry(manifest: Manifest, filename: str, entry: ManifestEntry) -> None:
    """Insert or replace the entry for ``filename``."""

    entries = manifest.get("entries")
    if entries is None:
        entries = {}
        manifest["entries"] = entries
    entries[filename] = entry


def verify_manifest(models_dir: Path, manifest: Manifest, *, only: Collection[str] | None = None) -> list[str]:
    """Return a list of problems; an empty list means the manifest is consistent.

    Verification is bidirectional: an entry whose file is missing on disk is a
    problem, and so is an ONNX file present on disk without an entry — otherwise
    ``--verify-manifest`` would silently stop accounting for an artifact that a
    training run (or a stray copy) dropped into ``Models/``.

    ``only`` narrows the sweep to the named artifacts. A training run passes the
    models it just published, so a stray file already sitting in ``Models/``
    cannot fail an otherwise clean run; the global ``--verify-manifest`` command
    passes ``only=None`` and keeps the full bidirectional check. A name in
    ``only`` that has no entry is reported, so narrowing cannot hide a model the
    run failed to record.
    """

    problems: list[str] = []
    entries = manifest.get("entries") or {}
    for filename, entry in entries.items():
        if only is not None and filename not in only:
            continue
        path = models_dir / filename
        if not path.exists():
            problems.append(f"{filename}: missing on disk")
            continue
        expected = (entry or {}).get("sha256")
        if not expected:
            problems.append(f"{filename}: entry has no sha256")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{filename}: sha256 mismatch (expected {expected[:12]}..., got {actual[:12]}...)")
    if only is None:
        for path in sorted(models_dir.glob("*.onnx")):
            if path.name not in entries:
                problems.append(f"{path.name}: on disk but not recorded in the manifest")
    else:
        for filename in sorted(only):
            if filename not in entries:
                problems.append(f"{filename}: expected but not recorded in the manifest")
    return problems


def verify_or_raise(models_dir: Path, manifest: Manifest, *, only: Collection[str] | None = None) -> None:
    """Verify and turn any problem into a single :class:`ManifestIntegrityError`.

    Both callers of :func:`verify_manifest` — the ``--verify-manifest`` command
    and the run that just wrote a manifest — ask the same question and want the
    same verdict (exit code ``2``, every problem named). Raising instead of
    returning a code keeps that decision here, next to the rules that produce
    the problems, and lets the composition root render it like any other
    expected failure: one actionable line, no traceback.

    ``only`` is forwarded unchanged: a training run narrows verification to the
    artifacts it just published (see :func:`verify_manifest`).
    """

    problems = verify_manifest(models_dir, manifest, only=only)
    if problems:
        raise ManifestIntegrityError(
            f"Manifest verification failed: {models_dir / MANIFEST_NAME} -> " + "; ".join(problems)
        )


def build_from_scan(models_dir: Path) -> Manifest:
    """Create/refresh manifest entries from the ONNX files already on disk.

    Used by ``--write-manifest`` to adopt models that were produced elsewhere
    (for example the pre-existing artifacts shipped with the repository). Only
    the files named by a spec in :data:`cvtrain.config.MODEL_REGISTRY` are
    adopted: vouching for an unknown ONNX would hand it a clean verification
    with no provenance behind it, so ``verify_manifest`` reports a stray
    instead. Any training metadata already recorded in the manifest is
    preserved as long as the file content is unchanged, while entries whose
    file is gone are pruned — a scan describes the directory as it is now.
    """

    manifest = load_manifest(models_dir)
    # ``load_manifest`` always returns a document with an ``entries`` map.
    entries = manifest["entries"]

    scanned = now_utc_iso()
    # A scan describes the directory as it is now, so an entry whose file is
    # gone is stale evidence rather than provenance worth keeping. Without this
    # ``--write-manifest`` left a deleted model recorded forever, and the next
    # ``--verify-manifest`` reported a ghost the scan itself could have pruned.
    for filename in sorted(entries):
        if not (models_dir / filename).exists():
            LOG.info("Pruning manifest entry for missing file: %s", filename)
            del entries[filename]

    for spec in MODEL_REGISTRY.values():
        path = models_dir / spec.output
        if not path.exists():
            continue

        previous = entries.get(spec.output, {})
        new_sha = sha256_file(path)
        content_changed = bool(previous.get("sha256")) and previous.get("sha256") != new_sha

        if content_changed:
            # The file changed outside this pipeline: drop stale training
            # provenance instead of reporting misleading metadata.
            entry: ManifestEntry = {}
            kept_role = previous.get("role")
            if kept_role is not None:
                entry["role"] = kept_role
            kept_source = previous.get("source_weights")
            if kept_source is not None:
                entry["source_weights"] = kept_source
        else:
            # Unchanged on disk: adopt the existing entry (and its provenance)
            # and refresh the fields that describe this scan.
            entry = previous

        entry.setdefault("role", spec.role)
        entry.setdefault("source_weights", spec.ckpt)
        if spec.role == "tracking":
            # Association lives in the Qt runtime, not in the graph, so this
            # stays true for an artifact adopted from disk without a training
            # run behind it. ``setdefault`` preserves a richer recorded block.
            entry.setdefault("tracker", dict(TRACKER_INFO))
        entry["sha256"] = new_sha
        entry["size_bytes"] = file_size(path)
        if "trained_at" not in entry:
            entry["trained_at"] = None
        entry["scanned_at"] = scanned
        entries[spec.output] = entry

    manifest["generator"] = generator_name()
    # ``schema_version`` is deliberately *not* re-assigned here: ``load_manifest``
    # already migrated the document, and stamping our own version over a newer
    # one would down-stamp it (see :func:`migrate`).
    manifest["updated_at"] = scanned
    return manifest
