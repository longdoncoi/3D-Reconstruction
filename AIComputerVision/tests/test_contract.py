"""Cross-language contract test.

The ONNX file names produced here are hard-coded expectations of the Qt app
(``src/app/AppConstants.h``). This test fails if the two sides ever drift.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from cvtrain.config import MODEL_SPECS
from cvtrain.preflight import MAX_OPSET

HEADER = Path(__file__).resolve().parents[2] / "src" / "app" / "AppConstants.h"
CMAKE = Path(__file__).resolve().parents[2] / "CMakeLists.txt"

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


class ContractSourcePresenceTests(unittest.TestCase):
    """A relocated contract file must not switch the contract itself off.

    ``ContractTests``/``OpsetCeilingTests`` skip themselves when the C++
    sources are absent, which is right for a Python-only checkout — but a
    *rename* has the identical effect, leaving CI green with no comparison
    running at all. This test inverts the guard: if the Qt sources are checked
    out, the files the contract reads must be there too, so a move turns the
    suite red instead of quiet.
    """

    REPO_SRC = Path(__file__).resolve().parents[2] / "src"

    def test_the_contract_sources_are_present_when_the_qt_sources_are(self) -> None:
        if not self.REPO_SRC.exists():
            self.skipTest("Python-only checkout: there is no contract to compare against")
        self.assertTrue(HEADER.exists(), f"{HEADER} moved or was renamed; ContractTests now skips silently")
        self.assertTrue(CMAKE.exists(), f"{CMAKE} missing; the opset ceiling tripwire is inactive")


ORT_OPSET_CEILING = {"1.20": 21}
"""ONNX Runtime -> highest opset its loader implements (vendor table).

A deliberate tripwire, not a proof: when ``ONNXRUNTIME_VERSION`` moves in
``CMakeLists.txt``, this test fails until a reviewer confirms the new
loader's ceiling and extends the map — so an export stamped higher than the
bundled runtime can load is never a silent surprise.
"""


@unittest.skipUnless(CMAKE.exists(), f"CMakeLists.txt not found at {CMAKE}")
class OpsetCeilingTests(unittest.TestCase):
    def test_max_opset_matches_the_pinned_onnxruntime(self) -> None:
        match = re.search(r'set\(ONNXRUNTIME_VERSION\s+"([^"]+)"', CMAKE.read_text(encoding="utf-8"))
        self.assertIsNotNone(match, "ONNXRUNTIME_VERSION not found in CMakeLists.txt")
        assert match is not None
        version = match.group(1)
        short = ".".join(version.split(".")[:2])
        self.assertIn(
            short,
            ORT_OPSET_CEILING,
            f"ONNX Runtime {version} is newer than the reviewed map; "
            "check the vendor opset table and extend ORT_OPSET_CEILING",
        )
        self.assertEqual(
            MAX_OPSET,
            ORT_OPSET_CEILING[short],
            f"preflight MAX_OPSET must be the opset ONNX Runtime {version} implements",
        )


if __name__ == "__main__":
    unittest.main()
