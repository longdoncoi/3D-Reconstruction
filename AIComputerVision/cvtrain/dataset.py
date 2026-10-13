"""Dataset validation: is ``data.yaml`` present, well-formed and on disk?

Isolated from the rest of preflight because it is a different kind of work —
parsing and resolving a YAML document rather than inspecting the environment or
the hyper-parameters — and because it is the one check whose inputs come from
the user's filesystem rather than from :class:`~cvtrain.config.TrainConfig`.

Everything here answers with a single :class:`~cvtrain.checks.Check` named
``dataset``; :mod:`cvtrain.preflight` simply appends it to its report.
"""

from __future__ import annotations

from pathlib import Path

from .checks import Check, module_available

__all__ = ["check_data_yaml", "dataset_base", "missing_dataset_paths", "ref_exists"]


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
    a plain path is left to the framework instead of being rejected here.
    """

    base = dataset_base(parsed, yaml_dir)
    missing: list[str] = []
    for key in ("train", "val"):
        value = parsed.get(key)
        if not isinstance(value, str):
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
    if not parsed.get("names"):
        return Check("dataset", False, f"'names' must define at least one class in {path}")
    missing_paths = missing_dataset_paths(parsed, path.parent)
    if missing_paths:
        return Check("dataset", False, "missing path(s): " + "; ".join(missing_paths))
    return Check("dataset", True, f"{path} ({len(parsed.get('names') or [])} classes)")
