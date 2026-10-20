"""Dataset validation: is ``data.yaml`` present, well-formed and on disk?

Isolated from the rest of preflight because it is a different kind of work —
parsing and resolving a YAML document rather than inspecting the environment or
the hyper-parameters — and because it is the one check whose inputs come from
the user's filesystem rather than from :class:`~cvtrain.config.TrainConfig`.

Everything here answers with a single :class:`~cvtrain.checks.Check` named
``dataset``; :mod:`cvtrain.preflight` simply appends it to its report.
"""

from __future__ import annotations

import glob
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .checks import Check, module_available

__all__ = [
    "check_data_yaml",
    "dataset_base",
    "dataset_class_count",
    "dataset_content_fingerprint",
    "missing_dataset_paths",
    "ref_exists",
]


def dataset_base(parsed: dict[object, object], yaml_dir: Path) -> Path:
    """Directory that relative ``train``/``val`` references resolve against.

    Mirrors Ultralytics: an explicit top-level ``path:`` key relocates the
    dataset root (resolved relative to ``data.yaml``), and without it references
    are relative to the folder holding ``data.yaml``.
    """

    root = parsed.get("path")
    if isinstance(root, str) and root:
        candidate = Path(root).expanduser()
        return candidate if candidate.is_absolute() else yaml_dir / candidate
    return yaml_dir


def ref_exists(value: str, base: Path) -> bool:
    """Whether a single ``train``/``val`` reference points at something on disk.

    Globs (``images/*.jpg``) are checked at their static prefix: a wildcard has
    no literal path to test, so validating ``images/`` is the useful subset.
    """

    raw = Path(value)
    if raw.is_absolute():
        return raw.exists()
    if (base / raw).exists():
        return True
    if any(char in value for char in "*?["):
        head = value.split("?")[0].split("*")[0].split("[")[0].rstrip("\\/")
        return (base / head).exists() if head else base.exists()
    return False


def missing_dataset_paths(parsed: dict[object, object], yaml_dir: Path) -> list[str]:
    """Human-readable ``train``/``val`` references that are not on disk.

    Only string entries are validated: ``train`` can legitimately be a text file
    listing or a remote URL that the framework resolves, so anything that is not
    a plain path is left to the framework instead of being rejected here —
    ``ref_exists`` is *only* asked about references that name a local path.
    """

    base = dataset_base(parsed, yaml_dir)
    missing: list[str] = []
    for key in ("train", "val"):
        value = parsed.get(key)
        if not isinstance(value, str):
            continue
        if "://" in value:
            # Remote (http://, https://, s3://, ...): there is nothing on this
            # disk to check, and reporting it as "missing" would reject a
            # dataset shape Ultralytics downloads itself.
            continue
        if not ref_exists(value, base):
            missing.append(f"{key}='{value}' (resolved under {base})")
    return missing


def check_data_yaml(path: Path) -> Check:
    """Validate the dataset definition the run is going to train against."""

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

    if not module_available("yaml"):
        return Check("dataset", True, f"{path} found (PyYAML missing, content not validated)")

    import yaml

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return Check("dataset", False, f"invalid YAML: {exc}")

    if not isinstance(parsed, dict):
        return Check("dataset", False, "top level of data.yaml must be a mapping")
    # Both splits are required, exactly as README ("Setup") documents: a dataset
    # without ``val`` would pass preflight and then fail (or train without ever
    # validating) deep inside the framework, which is what this check exists to
    # prevent.
    missing = [key for key in ("train", "val", "names") if key not in parsed]
    if missing:
        return Check("dataset", False, f"missing required key(s): {', '.join(missing)} in {path}")
    class_count = _names_count(parsed.get("names"))
    if class_count is None:
        # A scalar or an empty container is not a class list: ``len()`` on it
        # used to raise a bare TypeError from inside a check whose whole job is
        # to report malformed input as a verdict, not a traceback.
        return Check(
            "dataset",
            False,
            f"'names' must define at least one class as a mapping or list in {path}",
        )
    missing_paths = missing_dataset_paths(parsed, path.parent)
    if missing_paths:
        return Check("dataset", False, "missing path(s): " + "; ".join(missing_paths))
    return Check("dataset", True, f"{path} ({class_count} classes)")


def _names_count(names: object) -> int | None:
    """Classes a ``names`` value declares, or ``None`` when the shape is unusable.

    Both shapes Ultralytics accepts are supported: a list of names and an
    ``{index: name}`` mapping. Anything else — a scalar, an empty container — is
    not a class list, and measuring it with ``len()`` is exactly the crash a
    malformed ``data.yaml`` used to cause.
    """

    if isinstance(names, (dict, list, tuple)):
        return len(names) if names else None
    return None


def dataset_class_count(path: Path) -> int | None:
    """Number of classes declared by ``names`` in ``data.yaml``, or ``None``.

    The ONNX contract needs the class count to prove that an export exposes the
    channels the Qt parser expects (``4 + classes`` for detection,
    ``4 + classes + 32`` for segmentation). This deliberately re-reads the YAML
    instead of threading a count through every layer: it is the same document
    the framework trains from, so there is no second source of truth to drift.
    ``None`` means "cannot prove" — the caller relaxes only the exact channel
    check and keeps validating rank/count/dtype.
    """

    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not module_available("yaml"):
        return None

    import yaml

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    return _names_count(parsed.get("names"))


def _digest_file(digest: Any, label: str, path: Path, *, relative_name: str | None = None) -> None:
    """Feed one file's identity and bytes into ``digest``.

    The relative name is part of the hash (moving an image between splits is a
    different dataset), and unreadable files contribute a stable marker instead
    of aborting the fingerprint: ``--skip-unchanged`` reads the filesystem
    opportunistically, and a transient sharing violation must not crash a run.
    """

    name = path.name if relative_name is None else relative_name
    digest.update(f"{label}:{name}:".encode("utf-8"))
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        digest.update(b"<unreadable>")
    digest.update(b"\n")


def _is_remote(value: str) -> bool:
    """Whether a split reference points off this machine (nothing to hash)."""

    return "://" in value


def _relative_name(path: Path, base: Path) -> str:
    """Stable digest name: relative to the dataset root when possible."""

    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _reference_paths(value: str, base: Path) -> list[Path]:
    """Local paths one split reference names, with globs expanded in sorted order.

    Ultralytics accepts a directory, a single file (including a ``.txt`` listing)
    and a glob such as ``images/train/*.jpg``. All three have to reach the
    fingerprint, or ``--skip-unchanged`` would trust a dataset whose glob now
    selects different images. Sorting keeps the digest deterministic across
    filesystems.
    """

    raw = Path(value).expanduser()
    target = raw if raw.is_absolute() else base / raw
    if any(char in value for char in "*?["):
        return [Path(match) for match in sorted(glob.glob(str(target), recursive=True))]
    return [target]


def _listing_paths(listing: Path, base: Path) -> list[Path]:
    """Files named by a ``train.txt``-style listing.

    Blank lines and ``#`` comments are ignored. Entries are resolved relative to
    the dataset root first (Ultralytics' convention) and to the listing's own
    folder second, so both shapes found in the wild work. An entry that exists
    nowhere still contributes a missing marker: creating the file later must
    change the fingerprint, because it changes what the dataset *is*.
    """

    try:
        text = listing.read_text(encoding="utf-8")
    except OSError:
        return []
    paths: list[Path] = []
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or _is_remote(entry):
            continue
        raw = Path(entry).expanduser()
        if raw.is_absolute():
            paths.append(raw)
            continue
        candidates = [base / raw]
        if listing.parent != base:
            candidates.append(listing.parent / raw)
        paths.append(next((candidate for candidate in candidates if candidate.exists()), candidates[0]))
    return paths


def _split_files(value: object, base: Path) -> list[tuple[Path, str]]:
    """``(path, digest name)`` pairs one ``train``/``val`` value contributes.

    Handles every shape Ultralytics documents: a directory tree, a glob, a
    single file, a ``.txt`` listing of paths, and a list mixing them. A value
    the framework resolves itself (a URL, a non-string) contributes nothing —
    ``missing_dataset_paths`` leaves it alone for the same reason.
    """

    if isinstance(value, (list, tuple)):
        files: list[tuple[Path, str]] = []
        for item in value:
            if isinstance(item, str) and not _is_remote(item):
                files.extend(_split_files(item, base))
        return files
    if not isinstance(value, str) or _is_remote(value):
        return []
    files = []
    for target in _reference_paths(value, base):
        if target.is_dir():
            children = sorted(
                (child for child in target.rglob("*") if child.is_file()),
                key=lambda child: child.relative_to(target).as_posix(),
            )
            files.extend((child, child.relative_to(target).as_posix()) for child in children)
        elif target.is_file():
            files.append((target, _relative_name(target, base)))
            if target.suffix.lower() == ".txt":
                files.extend((entry, _relative_name(entry, base)) for entry in _listing_paths(target, base))
        else:
            # A literal path that does not exist (or a glob that currently
            # matches nothing literal): hash a stable marker so creating and
            # deleting the file move the fingerprint in both directions.
            files.append((target, value.replace("\\", "/")))
    return files


def dataset_content_fingerprint(path: Path, on_file: Callable[[int, int], None] | None = None) -> str | None:
    """Compute a content fingerprint of the files a dataset YAML references.

    Returns ``None`` when the YAML file is absent or unreadable. Combines the
    raw hash of ``data.yaml`` with the relative path and contents of every file
    reachable from the referenced ``train``/``val`` splits — directory trees,
    globs, single files, ``.txt`` listings and list-valued splits alike.
    Hashing the bytes — not just ``(name, size)`` — is what detects an image
    replaced by different pixels under the same name and the same file size.

    ``on_file`` is called as ``on_file(done, total)`` after each file, so a
    caller can report progress: the fingerprint reads the whole dataset, and
    on a large one that is minutes during which the operator would otherwise
    see nothing at all.

    The cost is O(dataset bytes) per check. That is deliberate: the fingerprint
    exists so ``--skip-unchanged`` can prove the data, and an inventory that a
    single same-size edit can fool is not proof.
    """

    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None

    digest = hashlib.sha256(text.encode("utf-8"))
    if not module_available("yaml"):
        return digest.hexdigest()

    import yaml

    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError:
        return digest.hexdigest()

    if not isinstance(parsed, dict):
        return digest.hexdigest()

    base = dataset_base(parsed, path.parent)
    # One entry per (split, relative name): a file reachable through two routes
    # is still one fact about the dataset, and deduplicating keeps the digest
    # independent of how the YAML spells its references.
    files: dict[tuple[str, str], Path] = {}
    for key in ("train", "val"):
        for target, name in _split_files(parsed.get(key), base):
            files.setdefault((key, name), target)
    ordered = sorted(files.items())
    total = len(ordered)
    for index, ((key, name), target) in enumerate(ordered, start=1):
        _digest_file(digest, key, target, relative_name=name)
        if on_file is not None:
            on_file(index, total)

    return digest.hexdigest()

