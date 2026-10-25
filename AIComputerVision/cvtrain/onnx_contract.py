"""Post-export ONNX compatibility contract shared with the Qt runtime.

The desktop app does not just load any valid ONNX file: ``AIProcessor.cpp``
reads exactly one NCHW float32 input, runs the first output through a fixed
YOLO parser, and (for segmentation) expects two outputs, one of which is a
rank-4 mask prototype. A graph can be perfectly valid ONNX and still violate
that contract — a different input rank, a second output the loader ignores, a
segmentation export missing its proto, or an opset newer than the bundled ONNX
Runtime. Those failures used to surface only when the Qt app loaded the file,
long after the training run reported success.

:func:`validate_export` closes that gap: it runs *before* the artifact is
published and asks the same engine the app asks (ONNX Runtime), so an
incompatible export leaves the previously published model untouched.

The numeric constants here are mirrored in ``src/services/ai/AIYoloPostProcessor.h``
and cross-checked by ``tests/test_contract.py`` — the two languages cannot
drift apart without a red test.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from .checks import module_available
from .config import MAX_OPSET, ModelSpec, TrainConfig
from .dataset import dataset_class_count
from .errors import ExportContractError

if TYPE_CHECKING:  # ``onnx`` is imported lazily in ``_load``: --help must not need it
    import onnx

LOG = logging.getLogger("cvtrain.onnx_contract")

_FLOAT32 = 1
"""``onnx.TensorProto.FLOAT``: the only element type the C++ parser accepts."""

INPUT_RANK = 4
"""The C++ loader reads ``shape[2]``/``shape[3]`` as H/W: the input must be NCHW."""

INPUT_CHANNELS = 3
"""``prepareInputTensor`` always fills a 3-channel tensor."""

BOX_COORDINATES = 4
"""``parsePredictions`` reads cx, cy, w, h from the first four columns."""

DETECTION_OUTPUT_RANK = 3
"""``AIProcessor.cpp`` indexes ``dims[1]``/``dims[2]`` then transposes."""

SEGMENTATION_OUTPUT_COUNT = 2
"""Segmentation reads output 0 and output 1 unconditionally."""

SEGMENTATION_PROTO_RANK = 4
"""The proto output is reshaped as a 4-D mat; its channel count is the mask count."""

MASK_COEFFICIENTS = 32
"""Mask coefficients per detection in the shipped segmentation exports."""

DEFAULT_MODEL_INPUT_SIZE = 640
"""The C++ fallback for a dynamic input shape (``kDefaultModelInputSize``)."""


@dataclass(frozen=True)
class OutputSpec:
    """One tensor the Qt runtime expects on the model's output list."""

    rank: int
    channels: int | None
    """Expected size of dimension 1, or ``None`` when the class count is unknown.

    A dynamic axis (0) is also accepted: the C++ loader reads the concrete
    shape at run time, so only a *contradicting* static value is a failure.
    """


@dataclass(frozen=True)
class RoleContract:
    """Output tensors one semantic role must expose."""

    outputs: tuple[OutputSpec, ...]


def contract_for(role: str, class_count: int | None) -> RoleContract:
    """Return the tensor contract for a model ``role``.

    ``detection`` and ``tracking`` share the same graph schema: the tracking
    artifact is a detector; association happens in C++ (see
    :data:`cvtrain.config.TRACKER_INFO`). Segmentation additionally exports a
    mask-prototype tensor, which the loader accepts at either output index
    (``AIProcessor.cpp`` detects the rank-4 output rather than relying on
    order).

    An unknown role is rejected rather than guessed: a new model family needs
    its own C++ parsing, and silently applying the detection schema would move
    that incompatibility back to the desktop app.
    """

    if role == "segmentation":
        detection_channels = (
            None
            if class_count is None
            else BOX_COORDINATES + class_count + MASK_COEFFICIENTS
        )
        return RoleContract(
            outputs=(
                OutputSpec(rank=DETECTION_OUTPUT_RANK, channels=detection_channels),
                OutputSpec(rank=SEGMENTATION_PROTO_RANK, channels=MASK_COEFFICIENTS),
            )
        )
    if role in ("detection", "tracking"):
        channels = None if class_count is None else BOX_COORDINATES + class_count
        return RoleContract(outputs=(OutputSpec(rank=DETECTION_OUTPUT_RANK, channels=channels),))
    raise ExportContractError(
        f"unsupported model role {role!r}: extend the ONNX contract (and the C++ "
        "loader) before exporting this model family"
    )


def _fail(path: Path, message: str) -> NoReturn:
    raise ExportContractError(f"{path.name}: {message}")


def _shape(value_info: onnx.ValueInfoProto) -> list[int]:
    """Return the declared dimensions; symbolic axes are reported as ``0``."""

    tensor_type = value_info.type.tensor_type
    return [dim.dim_value if dim.HasField("dim_value") else 0 for dim in tensor_type.shape.dim]


def _require_float(path: Path, value_info: onnx.ValueInfoProto, what: str) -> None:
    if value_info.type.tensor_type.elem_type != _FLOAT32:
        _fail(path, f"{what} must be float32, got elem_type={value_info.type.tensor_type.elem_type}")


def _load(path: Path) -> onnx.ModelProto:
    """Load and structurally validate the graph (the cheap half of the check)."""

    import onnx

    try:
        model = onnx.load(str(path))
        onnx.checker.check_model(model)
    except Exception as exc:  # turns any parser error into one verdict
        raise ExportContractError(f"{path.name}: not a loadable ONNX model ({exc})") from exc
    return model


def _check_opset(path: Path, model: onnx.ModelProto) -> None:
    # Only the default ONNX domain is relevant: custom domains carry their own
    # versioning and the loader does not recognize them anyway.
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx") and opset.version > MAX_OPSET:
            _fail(
                path,
                f"opset {opset.version} exceeds the runtime ceiling {MAX_OPSET}; "
                "lower --opset or upgrade the bundled ONNX Runtime",
            )


def _check_inputs(path: Path, model: onnx.ModelProto, imgsz: int) -> None:
    inputs = list(model.graph.input)
    if len(inputs) != 1:
        _fail(path, f"the loader reads exactly one input, the graph declares {len(inputs)}")
    value_info = inputs[0]
    _require_float(path, value_info, "input")
    dims = _shape(value_info)
    if len(dims) != INPUT_RANK:
        _fail(path, f"input rank must be {INPUT_RANK} (NCHW), got {len(dims)}")
    if dims[1] != INPUT_CHANNELS:
        _fail(path, f"input must have {INPUT_CHANNELS} channels, got {dims[1]}")
    for axis, name in ((2, "height"), (3, "width")):
        extent = dims[axis]
        if extent == 0:
            # A dynamic spatial axis is accepted only when the C++ fallback
            # (640) is the size the model was exported for; otherwise inference
            # would silently run at a different resolution than training.
            if imgsz != DEFAULT_MODEL_INPUT_SIZE:
                _fail(
                    path,
                    f"input {name} is dynamic but --imgsz={imgsz}; the loader would "
                    f"fall back to {DEFAULT_MODEL_INPUT_SIZE}",
                )
        elif extent != imgsz:
            _fail(path, f"input {name} is {extent}, but the run exported at --imgsz={imgsz}")


def _check_outputs(
    path: Path,
    model: onnx.ModelProto,
    role_contract: RoleContract,
) -> None:
    outputs = list(model.graph.output)
    if len(outputs) != len(role_contract.outputs):
        _fail(
            path,
            f"the Qt loader expects {len(role_contract.outputs)} output(s), "
            f"the graph declares {len(outputs)}",
        )
    # Segmentation output order is not part of the C++ contract (the loader
    # detects the rank-4 proto), so match each tensor to a spec by rank rather
    # than by index.
    unmatched = list(role_contract.outputs)
    for value_info in outputs:
        _require_float(path, value_info, "output")
        dims = _shape(value_info)
        match = next((spec for spec in unmatched if spec.rank == len(dims)), None)
        if match is None:
            _fail(
                path,
                f"unexpected output rank {len(dims)}; expected one of "
                f"{sorted({spec.rank for spec in unmatched})}",
            )
        if match.channels is not None and dims[1] not in (0, match.channels):
            _fail(
                path,
                f"output has {dims[1]} channel(s) at dimension 1, expected {match.channels}",
            )
        unmatched.remove(match)


def _smoke_inference(path: Path, role_contract: RoleContract, imgsz: int) -> None:
    """Run the artifact through ONNX Runtime with the shape the app uses.

    This is the half that catches engine-level incompatibilities a graph check
    cannot: ops the installed runtime cannot execute, kernels that reject the
    declared shape, or an output that materializes with an unexpected rank.
    """

    if not module_available("onnxruntime"):
        raise ExportContractError(
            f"{path.name}: onnxruntime is required for post-export validation; "
            "run: pip install -r requirements.txt"
        )
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        if len(inputs) != 1:
            _fail(path, f"runtime session exposes {len(inputs)} input(s), expected 1")
        dummy = np.zeros((1, INPUT_CHANNELS, imgsz, imgsz), dtype=np.float32)
        outputs = session.run(None, {inputs[0].name: dummy})
    except ExportContractError:
        raise
    except Exception as exc:  # any runtime rejection is one verdict
        raise ExportContractError(f"{path.name}: ONNX Runtime rejected the eager inference ({exc})") from exc

    if len(outputs) != len(role_contract.outputs):
        _fail(
            path,
            f"runtime produced {len(outputs)} output(s), expected {len(role_contract.outputs)}",
        )
    for array in outputs:
        if array.dtype != np.float32:
            _fail(path, f"runtime output must be float32, got {array.dtype}")
        match = next((spec for spec in role_contract.outputs if spec.rank == array.ndim), None)
        if match is None:
            _fail(path, f"runtime output rank {array.ndim} does not match the contract")


def validate_export(path: Path, spec: ModelSpec, cfg: TrainConfig) -> None:
    """Validate ``path`` against the Qt loader contract, or raise.

    Called by the orchestrator after the backend exports and before the file is
    atomically published. ``cfg.data_yaml`` supplies the class count when it is
    readable; without it the rank/count checks still run and only the exact
    channel count is relaxed.
    """

    class_count = dataset_class_count(cfg.data_yaml)
    role_contract = contract_for(spec.role, class_count)
    model = _load(path)
    _check_opset(path, model)
    _check_inputs(path, model, cfg.imgsz)
    _check_outputs(path, model, role_contract)
    _smoke_inference(path, role_contract, cfg.imgsz)
    LOG.info(
        "ONNX contract ok: %s (role=%s, outputs=%d, class_count=%s)",
        path.name,
        spec.role,
        len(role_contract.outputs),
        class_count if class_count is not None else "unknown",
    )
