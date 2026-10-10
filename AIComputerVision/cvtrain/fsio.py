"""Filesystem primitives: hashing and crash-safe writes.

Both the manifest writer and the model publisher need "write a file without
ever leaving a half-written one behind". Keeping that logic here means there is
exactly one temp-file + rename implementation to get right (and to test), and
one hashing routine so a digest computed during training can never disagree
with one recomputed by ``--verify-manifest``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

__all__ = ["CHUNK_SIZE", "atomic_copy", "atomic_write_text", "sha256_file"]

CHUNK_SIZE = 1 << 20
"""Hashing block size (1 MiB): large enough to be fast, small enough for RAM."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, streamed in :data:`CHUNK_SIZE` chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8", newline: str = "\n") -> Path:
    """Write ``text`` to ``path`` atomically (temp file in the target directory + rename).

    ``newline="\\n"`` keeps text files byte-identical across platforms, which
    matters for a JSON manifest that is diffed by Git.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline=newline) as stream:
            stream.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return path


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` onto ``dst`` atomically; returns ``dst``.

    Readers therefore see either the previous artifact or the complete new one,
    never a truncated ONNX file mid-copy.
    """

    dst.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.", suffix=".tmp")
    os.close(handle)
    try:
        shutil.copyfile(src, tmp_name)
        os.replace(tmp_name, dst)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return dst
