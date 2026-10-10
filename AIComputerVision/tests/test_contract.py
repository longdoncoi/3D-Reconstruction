"""Cross-language contract test.

The ONNX file names produced here are hard-coded expectations of the Qt app
(``src/app/AppConstants.h``). This test fails if the two sides ever drift.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from cvtrain.config import MODEL_SPECS

HEADER = Path(__file__).resolve().parents[2] / "src" / "app" / "AppConstants.h"

EXPECTED = {
    "detectionModelFile": MODEL_SPECS["det"].output,
    "segmentationModelFile": MODEL_SPECS["seg"].output,
    "trackingModelFile": MODEL_SPECS["track"].output,
}


@unittest.skipUnless(HEADER.exists(), f"AppConstants.h not found at {HEADER}")
class ContractTests(unittest.TestCase):
    text: str
    """Contents of ``AppConstants.h``, loaded once in :meth:`setUpClass`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = HEADER.read_text(encoding="utf-8")

    def _returned_literal(self, function: str) -> str:
        match = re.search(rf'{function}\(\)\s*\{{[^"]*"([^"]+)"', self.text)
        self.assertIsNotNone(match, f"{function}() not found in AppConstants.h")
        assert match is not None  # narrowing for the type checker after assertIsNotNone
        return match.group(1)

    def test_exported_names_match_header(self) -> None:
        for function, expected in EXPECTED.items():
            with self.subTest(function=function):
                self.assertEqual(self._returned_literal(function), expected)


if __name__ == "__main__":
    unittest.main()
