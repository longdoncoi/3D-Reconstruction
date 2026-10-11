"""Model manifest: provenance + integrity for the exported ONNX artifacts.

Every exported model gets an entry describing which base weights, dataset and
hyper-parameters produced it, plus a SHA-256 digest so the Qt app (or a release
auditor) can detect a truncated or replaced file.

The document shape is expressed as :class:`Manifest`/:class:`ManifestEntry`
``TypedDict``s so that a typo in an entry key is a type error rather than
provenance that silently never gets read back.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from . import __version__
from .config import MANIFEST_NAME, MODEL_SPECS
from .fsio import atomic_write_text, sha256_file

__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "LastRun",
    "Manifest",
    "ManifestEntry",
    "build_from_scan",
    "load_manifest",
    "migrate",
    "now_utc_iso",
    "save_manifest",
    "sha256_file",
    "upsert_entry",
    "verify_manifest",
]

LOG = logging.getLogger("cvtrain.manifest")

SCHEMA_VERSION = 1
"""Version of the document shape this build writes. See :func:`migrate`."""


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

    trained_at: str | None
    """UTC ISO timestamp, or ``None`` when the file was adopted from disk."""

    scanned_at: str
    """UTC ISO timestamp of the last ``--write-manifest`` scan."""

    dataset: str
    run_dir: str
    metrics: dict[str, float]


class LastRun(TypedDict, total=False):
    """Header describing the most recent training invocation."""

    started_at: str
    dataset: str
    models: list[str]
    epochs: int
    imgsz: int
    seed: int
    opset: int


class Manifest(TypedDict, total=False):
    """The ``Models/manifest.json`` document."""

    schema_version: int
    generator: str
    entries: dict[str, ManifestEntry]
    last_run: LastRun
    ultralytics: str
    """Version of the backend that produced the last run (key is historical)."""
    updated_at: str


def now_utc_iso() -> str:
    """Timestamp for provenance fields: UTC, second precision, offset-suffixed."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _skeleton() -> Manifest:
    return {"schema_version": SCHEMA_VERSION, "generator": f"cvtrain {__version__}", "entries": {}}


Migration = Callable[[dict[str, Any]], dict[str, Any]]
"""Upgrades a document from schema ``N`` to ``N + 1``; mutates and returns it."""


def _migrate_0_to_1(data: dict[str, Any]) -> dict[str, Any]:
    """Schema 0 -> 1: documents written before the field was versioned.

    They carry no ``schema_version`` and may carry no ``generator``. The entry
    map itself is already validated by :func:`load_manifest`, so the shapes of
    ``entries`` are unchanged by this step.
    """

    data.setdefault("generator", f"cvtrain {__version__}")
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


def load_manifest(models_dir: Path) -> Manifest:
    """Return the existing manifest migrated to the current schema, or a fresh
    skeleton when the file is absent, unreadable or structurally invalid."""

    path = models_dir / MANIFEST_NAME
    if not path.exists():
        return _skeleton()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOG.warning("Corrupt manifest %s: %s", path, exc)
        return _skeleton()
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        LOG.warning("Manifest %s is invalid (missing entries dict)", path)
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


def verify_manifest(models_dir: Path, manifest: Manifest) -> list[str]:
    """Return a list of problems; an empty list means the manifest is consistent.

    Verification is bidirectional: an entry whose file is missing on disk is a
    problem, and so is an ONNX file present on disk without an entry — otherwise
    ``--verify-manifest`` would silently stop accounting for an artifact that a
    training run (or a stray copy) dropped into ``Models/``.
    """

    problems: list[str] = []
    entries = manifest.get("entries") or {}
    for filename, entry in entries.items():
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
    for path in sorted(models_dir.glob("*.onnx")):
        if path.name not in entries:
            problems.append(f"{path.name}: on disk but not recorded in the manifest")
    return problems


def build_from_scan(models_dir: Path) -> Manifest:
    """Create/refresh manifest entries from the ONNX files already on disk.

    Used by ``--write-manifest`` to adopt models that were produced elsewhere
    (for example the pre-existing artifacts shipped with the repository). Any
    training metadata already recorded in the manifest is preserved as long as
    the file content is unchanged.
    """

    manifest = load_manifest(models_dir)
    # ``load_manifest`` always returns a document with an ``entries`` map.
    entries = manifest["entries"]

    scanned = now_utc_iso()
    for spec in MODEL_SPECS.values():
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
        entry["sha256"] = new_sha
        entry["size_bytes"] = path.stat().st_size
        if "trained_at" not in entry:
            entry["trained_at"] = None
        entry["scanned_at"] = scanned
        entries[spec.output] = entry

    manifest["generator"] = f"cvtrain {__version__}"
    # ``schema_version`` is deliberately *not* re-assigned here: ``load_manifest``
    # already migrated the document, and stamping our own version over a newer
    # one would down-stamp it (see :func:`migrate`).
    manifest["updated_at"] = scanned
    return manifest
