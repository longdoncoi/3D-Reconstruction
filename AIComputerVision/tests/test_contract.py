"""Cross-language contract test.

The ONNX file names produced here are hard-coded expectations of the Qt app
(``src/core/AIConstants.h``). This test fails if the two sides ever drift.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from cvtrain import onnx_contract
from cvtrain.config import MODEL_REGISTRY, SUPPORTED_ROLES, TRACKER_INFO
from cvtrain.preflight import MAX_OPSET

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "core" / "AIConstants.h"
MANIFEST = ROOT / "src" / "core" / "AppManifest.json"
CMAKE = ROOT / "CMakeLists.txt"
POST_PROCESSOR = ROOT / "src" / "services" / "ai" / "AIYoloPostProcessor.h"
AI_PROCESSOR = ROOT / "src" / "services" / "ai" / "AIProcessor.cpp"

EXPECTED = {
    "detectionModelFile": MODEL_REGISTRY["det"].output,
    "segmentationModelFile": MODEL_REGISTRY["seg"].output,
    "trackingModelFile": MODEL_REGISTRY["track"].output,
}


@unittest.skipUnless(HEADER.exists(), f"AIConstants.h not found at {HEADER}")
class ContractTests(unittest.TestCase):
    text: str
    """Contents of ``AIConstants.h``, loaded once in :meth:`setUpClass`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = HEADER.read_text(encoding="utf-8")

    def _returned_literal(self, function: str) -> str:
        match = re.search(rf'{function}\(\)\s*\{{[^"]*"([^"]+)"', self.text)
        self.assertIsNotNone(match, f"{function}() not found in AIConstants.h")
        assert match is not None  # narrowing for the type checker after assertIsNotNone
        return match.group(1)

    def test_exported_names_match_header(self) -> None:
        for function, expected in EXPECTED.items():
            with self.subTest(function=function):
                self.assertEqual(self._returned_literal(function), expected)

    def test_manifest_exists_and_matches_specs(self) -> None:
        import json
        self.assertTrue(MANIFEST.exists(), f"AppManifest.json not found at {MANIFEST}")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("contract", {}).get("detection_model"), EXPECTED["detectionModelFile"])
        self.assertEqual(manifest.get("contract", {}).get("segmentation_model"), EXPECTED["segmentationModelFile"])
        self.assertEqual(manifest.get("contract", {}).get("tracking_model"), EXPECTED["trackingModelFile"])


class RoleContractTests(unittest.TestCase):
    def test_every_supported_role_has_an_export_contract(self) -> None:
        # Preflight accepts exactly ``SUPPORTED_ROLES``; if a role is added
        # there without a contract branch, the mismatch would otherwise appear
        # only after training, at export time.
        for role in sorted(SUPPORTED_ROLES):
            with self.subTest(role=role):
                onnx_contract.contract_for(role, 1)


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
        self.assertTrue(
            POST_PROCESSOR.exists(),
            f"{POST_PROCESSOR} moved; the ONNX schema cross-check now skips silently",
        )
        self.assertTrue(
            AI_PROCESSOR.exists(),
            f"{AI_PROCESSOR} moved; the tracker threshold cross-check now skips silently",
        )


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


@unittest.skipUnless(POST_PROCESSOR.exists(), f"AIYoloPostProcessor.h not found at {POST_PROCESSOR}")
class OnnxSchemaContractTests(unittest.TestCase):
    """Python exports and the C++ parser must agree on the tensor schema.

    ``onnx_contract.validate_export`` encodes the same ranks/counts the C++
    loader assumes. If one side changes a constant without the other, exports
    either get rejected for a compatible model or accepted for one the app
    cannot parse — so both sides are compared here.
    """

    text: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.text = POST_PROCESSOR.read_text(encoding="utf-8")

    def _constant(self, name: str) -> int:
        match = re.search(rf"{name}\s*=\s*(\d+)", self.text)
        self.assertIsNotNone(match, f"{name} not found in AIYoloPostProcessor.h")
        assert match is not None
        return int(match.group(1))

    def test_cpp_constants_match_the_python_contract(self) -> None:
        expected = {
            "kExpectedInputRank": onnx_contract.INPUT_RANK,
            "kExpectedInputChannels": onnx_contract.INPUT_CHANNELS,
            "kExpectedDetectionOutputRank": onnx_contract.DETECTION_OUTPUT_RANK,
            "kExpectedSegmentationOutputCount": onnx_contract.SEGMENTATION_OUTPUT_COUNT,
            "kExpectedSegmentationProtoRank": onnx_contract.SEGMENTATION_PROTO_RANK,
            "kExpectedBoxCoordinates": onnx_contract.BOX_COORDINATES,
            "kExpectedMaskCoefficients": onnx_contract.MASK_COEFFICIENTS,
        }
        for name, python_value in expected.items():
            with self.subTest(constant=name):
                self.assertEqual(self._constant(name), python_value)

    def test_the_tracker_threshold_matches_the_manifest_info(self) -> None:
        # The artifact records the IoU threshold under which it was published;
        # if C++ changes its matcher the manifest would otherwise keep claiming
        # a value the runtime no longer uses.
        text = AI_PROCESSOR.read_text(encoding="utf-8")
        match = re.search(r"IOU_THRESHOLD\s*=\s*([0-9.]+)f", text)
        self.assertIsNotNone(match, "IOU_THRESHOLD not found in AIProcessor.cpp")
        assert match is not None
        self.assertEqual(float(match.group(1)), TRACKER_INFO["iou_threshold"])


if __name__ == "__main__":
    unittest.main()
