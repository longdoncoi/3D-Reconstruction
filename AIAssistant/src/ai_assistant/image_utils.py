"""Image helpers shared by the vision and retrieval sides — a leaf module.

The helpers used to live in ``rag/image_utils.py``, which made ``llm`` import
``rag`` (``llm/prompts.py`` encoded vision attachments through it) while
``rag`` already imported ``llm`` (``rag/index.py`` → ``llm.hardware``): a
two-way dependency between two sibling packages. They sit at the package root
now — imported by ``llm``, ``rag`` and the HTTP adapter alike — so neither
package reaches into the other.

Nothing here imports from ``ai_assistant``: this module is a shared kernel
alongside ``ports.py`` (contracts) and ``settings.py`` (environment owner).
"""
from __future__ import annotations

import base64
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from .ports import SEALED_BINARY_PREFIX

if TYPE_CHECKING:
    from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def read_sealed_or_plain(path: str, unseal: Callable[[bytes], bytes] | None = None) -> bytes:
    """Read a file that may be sealed at rest (ADR 0029).

    The chat attachments are sealed with the platform at-rest key, so the
    AI service has to open them before it can show them to the model. The
    ``unseal`` callable is **injected**: this module is a kernel leaf and may
    not import an adapter, so the caller owns the key.

    A file that is not a sealed envelope is returned unchanged, so an
    attachment written before sealing existed still loads.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    if unseal is not None and raw.startswith(SEALED_BINARY_PREFIX):
        return unseal(raw)
    return raw


def is_image_file(filepath: str) -> bool:
    """Check if a file has a recognized image extension."""
    return os.path.splitext(filepath)[1].lower() in IMAGE_EXTS


def image_to_data_uri(filepath: str, max_dim: int = 512, *,
                      unseal: Callable[[bytes], bytes] | None = None) -> str:
    """Read an image, scale it down, and return a base64 data URI.

    ``unseal`` opens a file that was sealed at rest (see
    :func:`read_sealed_or_plain`).
    """
    import io

    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
        ".bmp": "bmp", ".gif": "gif", ".webp": "webp"
    }
    mime = mime_map.get(ext, "jpeg")
    raw = read_sealed_or_plain(filepath, unseal)

    try:
        from PIL import Image
        img: "Image.Image" = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if mime == "jpeg" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        save_fmt = "JPEG" if mime == "jpeg" else mime.upper()
        img.save(buf, format=save_fmt, quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # PIL is optional: without it (or if decoding fails) fall back to
        # base64-encoding the raw file so callers keep getting a usable URI.
        b64 = base64.b64encode(raw).decode("utf-8")

    return f"data:image/{mime};base64,{b64}"


def load_image_for_embedding(filepath: str, max_dim: int = 256) -> Image.Image:
    """Load and thumbnail an image for Clip/embedding models."""
    from PIL import Image, ImageOps
    with Image.open(filepath) as opened:
        transposed: Image.Image = ImageOps.exif_transpose(opened) or opened
        img = transposed.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return img.copy()
