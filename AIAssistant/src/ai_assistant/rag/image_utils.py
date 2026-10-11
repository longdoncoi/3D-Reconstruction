"""Image processing utilities for vision and embedding."""
from __future__ import annotations

import base64
import os

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def is_image_file(filepath: str) -> bool:
    """Check if a file has a recognized image extension."""
    return os.path.splitext(filepath)[1].lower() in IMAGE_EXTS


def image_to_data_uri(filepath: str, max_dim: int = 512) -> str:
    """Read an image, scale it down, and return a base64 data URI."""
    import io

    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        ".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png",
        ".bmp": "bmp", ".gif": "gif", ".webp": "webp"
    }
    mime = mime_map.get(ext, "jpeg")
    
    try:
        from PIL import Image
        img: "Image.Image" = Image.open(filepath)
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
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            
    return f"data:image/{mime};base64,{b64}"


def load_image_for_embedding(filepath: str, max_dim: int = 256):
    """Load and thumbnail an image for Clip/embedding models."""
    from PIL import Image, ImageOps
    with Image.open(filepath) as opened:
        transposed: Image.Image = ImageOps.exif_transpose(opened) or opened
        img = transposed.convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        return img.copy()
