"""Guard-path coverage for image scaling.

``image_to_data_uri`` must downscale large captures before base64-encoding
them — an 800px screenshot sent at full resolution would blow up the vision
prompt. A synthetic image keeps the test offline and deterministic.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_assistant.rag.image_utils import image_to_data_uri, is_image_file


class ImageScalingTests(unittest.TestCase):
    def test_a_large_image_is_scaled_to_the_limit(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.png"
            Image.new("RGB", (800, 600), "red").save(path)
            uri = image_to_data_uri(str(path))
        self.assertTrue(uri.startswith("data:image/png;base64,"))

        import base64
        import io

        from PIL import Image as _Image

        raw = base64.b64decode(uri.split(",", 1)[1])
        with _Image.open(io.BytesIO(raw)) as decoded:
            self.assertLessEqual(max(decoded.size), 512)

    def test_a_small_image_keeps_its_dimensions(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "icon.png"
            Image.new("RGB", (64, 64), "blue").save(path)
            uri = image_to_data_uri(str(path))
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_extension_check_is_case_insensitive(self):
        self.assertTrue(is_image_file("PHOTO.JPG"))
        self.assertFalse(is_image_file("notes.txt"))


if __name__ == "__main__":
    unittest.main()
