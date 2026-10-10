"""Unit tests for RAG image utilities."""
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path

from ai_assistant.rag.image_utils import IMAGE_EXTS, image_to_data_uri, is_image_file, load_image_for_embedding

# Minimal 1x1 transparent PNG.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


class IsImageFileTests(unittest.TestCase):
    def test_recognised_extensions(self) -> None:
        self.assertTrue(is_image_file("photo.JPG"))
        self.assertTrue(is_image_file("photo.png"))
        self.assertTrue(is_image_file("photo.webp"))

    def test_unknown_extension(self) -> None:
        self.assertFalse(is_image_file("photo.py"))
        self.assertFalse(is_image_file("photo.tex"))

    def test_constant_extensions_present(self) -> None:
        self.assertIn(".png", IMAGE_EXTS)
        self.assertIn(".bmp", IMAGE_EXTS)


class ImageToDataUriTests(unittest.TestCase):
    def test_valid_png_returns_data_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pix.png"
            path.write_bytes(_PNG_1X1)
            uri = image_to_data_uri(str(path))
            self.assertTrue(uri.startswith("data:image/png;base64,"))
            decoded = base64.b64decode(uri.split(",", 1)[1])
            self.assertTrue(decoded)

    def test_fallback_encoding_for_unreadable_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake.jpg"
            path.write_bytes(b"not an image at all " * 3)
            uri = image_to_data_uri(str(path))
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_jpeg_rgba_converted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pix.jpg"
            path.write_bytes(_PNG_1X1)  # RGBA pixels decoded by PIL
            uri = image_to_data_uri(str(path))
            self.assertTrue(uri.startswith("data:image/jpeg;base64,"))


class LoadImageForEmbeddingTests(unittest.TestCase):
    def test_returns_rgb_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pix.png"
            path.write_bytes(_PNG_1X1)
            img = load_image_for_embedding(str(path), max_dim=128)
            self.assertEqual(img.mode, "RGB")
            self.assertLessEqual(max(img.size), 128)


if __name__ == "__main__":
    unittest.main()
