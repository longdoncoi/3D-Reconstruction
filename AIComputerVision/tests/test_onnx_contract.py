"""Tests for the post-export ONNX contract (:mod:`cvtrain.onnx_contract`).

The models here are synthetic graphs written with the ``onnx`` helper: the
point is the tensor schema the Qt loader relies on, not trained weights. Every
rejection path is exercised so the contract can never silently stop checking a
dimension it used to validate.
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper

from cvtrain import onnx_contract
from cvtrain.config import MAX_OPSET, ModelSpec, TrainConfig
from cvtrain.errors import ExportContractError
from cvtrain.onnx_contract import (
    DETECTION_OUTPUT_RANK,
    INPUT_CHANNELS,
    MASK_COEFFICIENTS,
    SEGMENTATION_PROTO_RANK,
    validate_export,
)


def _vi(name: str, dtype: int, dims: Sequence[str | int | None]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, dims)


def _concrete(dims: Sequence[str | int | None]) -> list[int]:
    """Replace symbolic axes with an extent a Constant tensor can materialise."""

    return [extent if isinstance(extent, int) else 1 for extent in dims]


def _constant(name: str, dims: Sequence[str | int | None], dtype: int = TensorProto.FLOAT) -> onnx.NodeProto:
    concrete = _concrete(dims)
    count = 1
    for extent in concrete:
        count *= extent
    tensor = helper.make_tensor(f"{name}_value", dtype, concrete, [0.0] * count)
    return helper.make_node("Constant", [], [name], value=tensor)


def _write_model(
    path: Path,
    *,
    input_dims: Sequence[str | int | None],
    outputs: Sequence[tuple[str, int, Sequence[str | int | None]]],
    input_dtype: int = TensorProto.FLOAT,
    extra_input_dims: Sequence[Sequence[str | int | None]] | None = None,
    opset: int = 12,
) -> Path:
    inputs = [_vi("images", input_dtype, input_dims)]
    inputs += [_vi(f"aux{index}", TensorProto.FLOAT, dims) for index, dims in enumerate(extra_input_dims or [])]
    nodes = [_constant(name, dims, dtype) for name, dtype, dims in outputs]
    output_infos = [_vi(name, dtype, dims) for name, dtype, dims in outputs]
    graph = helper.make_graph(nodes, "contract-test", inputs, output_infos)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    # IR 8 is accepted by every runtime this project has pinned (1.20.1+).
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def _detection_model(
    path: Path,
    *,
    classes: int = 1,
    imgsz: int = 640,
    channels: int | None = None,
    rank: int = DETECTION_OUTPUT_RANK,
    opset: int = 12,
    extra_input_dims: Sequence[Sequence[str | int | None]] | None = None,
) -> Path:
    channel_count = 4 + classes if channels is None else channels
    dims: list[str | int | None] = [1, channel_count, 5] if rank == DETECTION_OUTPUT_RANK else [1, channel_count]
    return _write_model(
        path,
        input_dims=[1, INPUT_CHANNELS, imgsz, imgsz],
        outputs=[("output0", TensorProto.FLOAT, dims)],
        extra_input_dims=extra_input_dims,
        opset=opset,
    )


def _segmentation_model(
    path: Path,
    *,
    classes: int = 1,
    imgsz: int = 640,
    reverse: bool = False,
) -> Path:
    boxes_dims: list[str | int | None] = [1, 4 + classes + MASK_COEFFICIENTS, 5]
    proto_dims: list[str | int | None] = [1, MASK_COEFFICIENTS, 8, 8]
    outputs: list[tuple[str, int, list[str | int | None]]] = [
        ("output0", TensorProto.FLOAT, boxes_dims),
        ("output1", TensorProto.FLOAT, proto_dims),
    ]
    if reverse:
        outputs.reverse()
    return _write_model(path, input_dims=[1, INPUT_CHANNELS, imgsz, imgsz], outputs=outputs)


def _cfg(root: Path, *, classes: int = 1, imgsz: int = 640) -> TrainConfig:
    names = "\n".join(f"  {index}: class{index}" for index in range(classes))
    data = root / "data.yaml"
    data.write_text(f"train: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    return TrainConfig(data_yaml=data, imgsz=imgsz)


def _spec(role: str) -> ModelSpec:
    return ModelSpec("test", "test.pt", "test.onnx", "test", role, "Test")


class _Sandbox(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = _cfg(self.root)

    def assert_rejected(self, pattern: str, path: Path, spec: ModelSpec | None = None) -> None:
        with self.assertRaisesRegex(ExportContractError, pattern):
            validate_export(path, spec or _spec("detection"), self.cfg)


class ContractForTests(unittest.TestCase):
    def test_detection_channels_are_four_boxes_plus_classes(self) -> None:
        contract = onnx_contract.contract_for("detection", 3)
        self.assertEqual(contract.outputs[0].rank, DETECTION_OUTPUT_RANK)
        self.assertEqual(contract.outputs[0].channels, 4 + 3)

    def test_tracking_shares_the_detection_schema(self) -> None:
        self.assertEqual(
            onnx_contract.contract_for("tracking", 3),
            onnx_contract.contract_for("detection", 3),
        )

    def test_segmentation_adds_the_mask_proto(self) -> None:
        contract = onnx_contract.contract_for("segmentation", 2)
        self.assertEqual(
            [(spec.rank, spec.channels) for spec in contract.outputs],
            [
                (DETECTION_OUTPUT_RANK, 4 + 2 + MASK_COEFFICIENTS),
                (SEGMENTATION_PROTO_RANK, MASK_COEFFICIENTS),
            ],
        )

    def test_unknown_class_count_relaxes_only_the_channel_check(self) -> None:
        self.assertIsNone(onnx_contract.contract_for("detection", None).outputs[0].channels)

    def test_an_unknown_role_is_rejected(self) -> None:
        with self.assertRaisesRegex(ExportContractError, "unsupported model role"):
            onnx_contract.contract_for("pose", 1)


class HappyPathTests(_Sandbox):
    def test_a_detection_export_passes(self) -> None:
        path = _detection_model(self.root / "det.onnx", classes=1)
        validate_export(path, _spec("detection"), self.cfg)

    def test_a_tracking_export_passes(self) -> None:
        path = _detection_model(self.root / "track.onnx", classes=1)
        validate_export(path, _spec("tracking"), self.cfg)

    def test_a_segmentation_export_passes_regardless_of_output_order(self) -> None:
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                path = _segmentation_model(self.root / f"seg-{reverse}.onnx", reverse=reverse)
                validate_export(path, _spec("segmentation"), self.cfg)

    def test_the_class_count_comes_from_the_yaml(self) -> None:
        cfg = _cfg(self.root, classes=2)
        path = _detection_model(self.root / "two-classes.onnx", classes=2)
        validate_export(path, _spec("detection"), cfg)

    def test_a_missing_yaml_relaxes_the_channel_count(self) -> None:
        cfg = TrainConfig(data_yaml=self.root / "missing.yaml", imgsz=640)
        path = _detection_model(self.root / "unknown-classes.onnx", classes=7)
        validate_export(path, _spec("detection"), cfg)

    def test_a_dynamic_spatial_axis_is_accepted_at_the_cpp_fallback_size(self) -> None:
        path = _write_model(
            self.root / "dynamic.onnx",
            input_dims=[1, INPUT_CHANNELS, "height", "width"],
            outputs=[("output0", TensorProto.FLOAT, [1, 4 + 1, 5])],
        )
        validate_export(path, _spec("detection"), self.cfg)

    def test_a_custom_domain_opset_does_not_count_against_the_ceiling(self) -> None:
        path = _detection_model(self.root / "custom-domain.onnx", classes=1)
        model = onnx.load(str(path))
        model.opset_import.append(helper.make_opsetid("com.example", 99))
        onnx.save(model, str(path))
        # Only the default ONNX domain is versioned against the runtime table.
        validate_export(path, _spec("detection"), self.cfg)


class GraphValidationTests(_Sandbox):
    def test_an_unloadable_file_is_rejected(self) -> None:
        path = self.root / "garbage.onnx"
        path.write_bytes(b"definitely not protobuf")
        self.assert_rejected("not a loadable ONNX model", path)

    def test_an_opset_above_the_runtime_ceiling_is_rejected(self) -> None:
        path = _detection_model(self.root / "future.onnx", opset=MAX_OPSET + 1)
        self.assert_rejected("exceeds the runtime ceiling", path)

    def test_a_graph_the_checker_rejects_is_rejected(self) -> None:
        # A node consuming a tensor no previous node produces: structurally
        # loadable protobuf, but the ONNX checker rejects the graph.
        node = helper.make_node("Relu", ["missing"], ["output0"])
        graph = helper.make_graph(
            [node],
            "disconnected",
            [_vi("images", TensorProto.FLOAT, [1, 3, 640, 640])],
            [_vi("output0", TensorProto.FLOAT, [1, 5, 5])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
        model.ir_version = 8
        path = self.root / "disconnected.onnx"
        onnx.save(model, str(path))
        self.assert_rejected("not a loadable ONNX model", path)


class InputContractTests(_Sandbox):
    def test_a_second_input_is_rejected(self) -> None:
        path = _detection_model(self.root / "two-inputs.onnx", extra_input_dims=[[1]])
        self.assert_rejected("exactly one input", path)

    def test_a_non_nchw_input_rank_is_rejected(self) -> None:
        path = _write_model(
            self.root / "rank3-input.onnx",
            input_dims=[1, INPUT_CHANNELS, 640],
            outputs=[("output0", TensorProto.FLOAT, [1, 5, 5])],
        )
        self.assert_rejected("input rank must be 4", path)

    def test_a_non_three_channel_input_is_rejected(self) -> None:
        path = _write_model(
            self.root / "one-channel.onnx",
            input_dims=[1, 1, 640, 640],
            outputs=[("output0", TensorProto.FLOAT, [1, 5, 5])],
        )
        self.assert_rejected("3 channels", path)

    def test_a_non_float_input_is_rejected(self) -> None:
        path = _write_model(
            self.root / "double-input.onnx",
            input_dims=[1, INPUT_CHANNELS, 640, 640],
            input_dtype=TensorProto.DOUBLE,
            outputs=[("output0", TensorProto.FLOAT, [1, 5, 5])],
        )
        self.assert_rejected("input must be float32", path)

    def test_a_static_spatial_size_mismatch_is_rejected(self) -> None:
        path = _detection_model(self.root / "wrong-size.onnx", imgsz=512)
        self.assert_rejected("input height is 512", path)

    def test_a_dynamic_spatial_axis_below_the_fallback_is_rejected(self) -> None:
        cfg = _cfg(self.root, imgsz=512)
        path = _write_model(
            self.root / "dynamic-small.onnx",
            input_dims=[1, INPUT_CHANNELS, "height", "width"],
            outputs=[("output0", TensorProto.FLOAT, [1, 5, 5])],
        )
        with self.assertRaisesRegex(ExportContractError, "loader would fall back to 640"):
            validate_export(path, _spec("detection"), cfg)


class OutputContractTests(_Sandbox):
    def test_a_missing_output_is_rejected(self) -> None:
        # A valid detection graph consumed by the segmentation role: the count
        # check must fire before any tensor is inspected.
        path = _detection_model(self.root / "not-seg.onnx", classes=1)
        self.assert_rejected("expects 2 output", path, _spec("segmentation"))

    def test_an_extra_output_is_rejected(self) -> None:
        path = _segmentation_model(self.root / "seg-detection.onnx")
        self.assert_rejected("expects 1 output", path, _spec("detection"))

    def test_a_non_float_output_is_rejected(self) -> None:
        path = _write_model(
            self.root / "double-output.onnx",
            input_dims=[1, INPUT_CHANNELS, 640, 640],
            outputs=[("output0", TensorProto.DOUBLE, [1, 5, 5])],
        )
        self.assert_rejected("output must be float32", path)

    def test_an_unexpected_output_rank_is_rejected(self) -> None:
        path = _detection_model(self.root / "rank2-output.onnx", rank=2)
        self.assert_rejected("unexpected output rank 2", path)

    def test_a_channel_count_mismatch_is_rejected(self) -> None:
        path = _detection_model(self.root / "five-channels.onnx", classes=2)
        # The YAML says one class (5 channels), the graph says two (6).
        self.assert_rejected("output has 6 channel", path)

    def test_a_dynamic_channel_axis_is_accepted(self) -> None:
        path = _write_model(
            self.root / "dynamic-channels.onnx",
            input_dims=[1, INPUT_CHANNELS, 640, 640],
            outputs=[("output0", TensorProto.FLOAT, [1, "channels", 5])],
        )
        validate_export(path, _spec("detection"), self.cfg)


class _FakeSession:
    def __init__(self, *, inputs: list[object], outputs: list[object]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self) -> list[object]:
        return self._inputs

    def run(self, output_names: object, feeds: object) -> list[object]:
        return self._outputs


class SmokeInferenceTests(_Sandbox):
    def test_a_missing_runtime_is_reported_with_the_install_hint(self) -> None:
        path = _detection_model(self.root / "no-runtime.onnx")
        with mock.patch("cvtrain.onnx_contract.module_available", return_value=False):
            self.assert_rejected("onnxruntime is required", path)

    def test_a_runtime_rejection_is_wrapped_as_a_contract_error(self) -> None:
        path = _detection_model(self.root / "runtime-rejects.onnx")
        with mock.patch("onnxruntime.InferenceSession", side_effect=RuntimeError("no kernel")):
            self.assert_rejected("ONNX Runtime rejected the eager inference", path)

    def test_a_session_with_a_second_input_is_rejected(self) -> None:
        path = _detection_model(self.root / "runtime-two-inputs.onnx")
        session = _FakeSession(inputs=[SimpleNamespace(name="images"), SimpleNamespace(name="aux")], outputs=[])
        with mock.patch("onnxruntime.InferenceSession", return_value=session):
            self.assert_rejected("exposes 2 input", path)

    def test_a_runtime_output_count_mismatch_is_rejected(self) -> None:
        path = _detection_model(self.root / "runtime-count.onnx")
        session = _FakeSession(
            inputs=[SimpleNamespace(name="images")],
            outputs=[np.zeros((1, 5, 5), dtype=np.float32), np.zeros((1, 5, 5), dtype=np.float32)],
        )
        with mock.patch("onnxruntime.InferenceSession", return_value=session):
            self.assert_rejected("runtime produced 2 output", path)

    def test_a_runtime_dtype_mismatch_is_rejected(self) -> None:
        path = _detection_model(self.root / "runtime-dtype.onnx")
        session = _FakeSession(
            inputs=[SimpleNamespace(name="images")],
            outputs=[np.zeros((1, 5, 5), dtype=np.float64)],
        )
        with mock.patch("onnxruntime.InferenceSession", return_value=session):
            self.assert_rejected("runtime output must be float32", path)

    def test_a_runtime_rank_mismatch_is_rejected(self) -> None:
        path = _detection_model(self.root / "runtime-rank.onnx")
        session = _FakeSession(
            inputs=[SimpleNamespace(name="images")],
            outputs=[np.zeros((1, 5), dtype=np.float32)],
        )
        with mock.patch("onnxruntime.InferenceSession", return_value=session):
            self.assert_rejected("runtime output rank 2", path)


if __name__ == "__main__":
    unittest.main()
