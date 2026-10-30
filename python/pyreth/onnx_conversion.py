"""Export PyTorch models to ONNX and validate the resulting artifacts.

This module wraps PyTorch's ONNX exporter with helpers for defining input and
output names, optionally making the batch dimension dynamic, checking graph
validity, comparing runtime outputs with PyTorch, and creating a Pyreth inference
session. PyTorch and ONNX are optional dependencies and are imported only when
these helpers are called.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._core import ExecutionProvider, InferenceSession, SessionConfig

DEFAULT_EXPORT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".onnx_cache"

__all__ = [
    "DEFAULT_EXPORT_CACHE_DIR",
    "export_torch_model_to_onnx",
    "load_onnx_session_from_torch",
    "validate_onnx_export",
]


def _ensure_torch_available() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - guarded by tests
        raise ModuleNotFoundError(
            "PyTorch support requires the optional 'torch' dependency."
        ) from exc
    return torch


def _ensure_onnx_available() -> Any:
    try:
        import onnx
    except ModuleNotFoundError as exc:  # pragma: no cover - guarded by tests
        raise ModuleNotFoundError(
            "ONNX validation requires the optional 'onnx' dependency."
        ) from exc
    return onnx


def _normalize_inputs(sample_inputs: Any) -> tuple[Any, ...] | dict[str, Any]:
    if sample_inputs is None:
        raise ValueError("sample_inputs is required to export a PyTorch model to ONNX.")

    if isinstance(sample_inputs, Mapping):
        return {str(k): v for k, v in sample_inputs.items()}

    if isinstance(sample_inputs, (list, tuple)):
        return tuple(sample_inputs)

    return (sample_inputs,)


def _as_torch_tensor(value: Any) -> Any:
    torch = _ensure_torch_available()
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _model_input_names(
    sample_inputs: Any, fallback_prefix: str = "input"
) -> tuple[str, ...]:
    if isinstance(sample_inputs, Mapping):
        return tuple(str(k) for k in sample_inputs)

    size = (
        len(_normalize_inputs(sample_inputs))
        if not isinstance(sample_inputs, (str, bytes))
        else 1
    )
    return tuple(f"{fallback_prefix}_{idx}" for idx in range(size))


def _sanitize_export_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return sanitized or "exported_model"


def _resolve_export_path(
    output_path: str | Path | None,
    *,
    cache_dir: str | Path | None = None,
    model: Any | None = None,
    default_name: str | None = None,
) -> Path:
    if output_path is not None:
        resolved = Path(output_path)
        if resolved.exists() and resolved.is_dir():
            model_name = (
                _sanitize_export_name(model.__class__.__name__)
                if model is not None
                else default_name or "exported_model"
            )
            return resolved / f"{model_name}.onnx"
        if resolved.suffix.lower() != ".onnx":
            resolved = resolved.with_suffix(".onnx")
        return resolved

    base_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_EXPORT_CACHE_DIR
    model_name = (
        _sanitize_export_name(model.__class__.__name__)
        if model is not None
        else default_name or "exported_model"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{model_name}.onnx"


def export_torch_model_to_onnx(
    model: Any,
    sample_inputs: Any,
    output_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force_export: bool = False,
    dynamic_batch: bool = True,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    validate: bool = True,
    training: bool = False,
) -> Path:
    """Export a PyTorch module to an ONNX file compatible with Pyreth.

    Args:
        model: PyTorch model instance to export.
        sample_inputs: Either a single tensor, a tuple/list of tensors, or a mapping
            of input names to tensor values. This input is used as the export-time
            sample for static/dynamic shape generation.
        output_path: Destination path. The `.onnx` suffix is added if needed.
        dynamic_batch: Whether to mark axis 0 of each input and output with the
            symbolic dimension `batch_size`.
        opset_version: ONNX opset version passed to `torch.onnx.export`.
        input_names: Names for positional inputs. The number of names must match
            the number of input tensors. For mapping inputs, keys are used unless
            overridden.
        output_names: Names for model outputs. The number of names must match the
            number of outputs.
        validate: Whether to validate the ONNX graph and compare its outputs with
            the PyTorch model.
        training: Training mode passed to `torch.onnx.export`.

    Returns:
        The path to the generated ONNX model.

    Raises:
        TypeError: If `model` is not a PyTorch `nn.Module`.
        ValueError: If the sample inputs or export naming contract is invalid.
        ModuleNotFoundError: If PyTorch is unavailable, or if validation is
            requested and ONNX is unavailable.
    """

    torch = _ensure_torch_available()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    sample_inputs_norm = _normalize_inputs(sample_inputs)
    if isinstance(sample_inputs_norm, Mapping):
        model_inputs = tuple(_as_torch_tensor(v) for v in sample_inputs_norm.values())
        resolved_input_names = tuple(str(k) for k in sample_inputs_norm)
    else:
        model_inputs = tuple(_as_torch_tensor(v) for v in sample_inputs_norm)
        resolved_input_names = input_names or _model_input_names(sample_inputs)

    if input_names is not None:
        if len(input_names) != len(model_inputs):
            raise ValueError(
                "input_names must match the number of input tensors; "
                f"got {len(input_names)} names for {len(model_inputs)} inputs"
            )
        resolved_input_names = tuple(str(name) for name in input_names)

    if not model_inputs:
        raise ValueError("At least one sample input tensor is required for export.")

    output_file = _resolve_export_path(
        output_path,
        cache_dir=cache_dir,
        model=model,
        default_name=model.__class__.__name__.lower(),
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists() and not force_export:
        if not validate:
            return output_file
        validate_onnx_export(
            output_file, model, sample_inputs, dynamic_batch=dynamic_batch
        )
        return output_file

    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in resolved_input_names:
        dynamic_axes[name] = {0: "batch_size"} if dynamic_batch else {}

    expected_output_count = _infer_model_output_count(model, model_inputs)
    if output_names is None:
        output_names_tuple = tuple(
            str(f"output_{idx}") if expected_output_count > 1 else "output"
            for idx in range(expected_output_count)
        )
    else:
        output_names_tuple = tuple(str(name) for name in output_names)
        if len(output_names_tuple) != expected_output_count:
            raise ValueError(
                "output_names must match the number of model outputs; "
                f"got {len(output_names_tuple)} names for {expected_output_count} outputs"
            )
    for name in output_names_tuple:
        dynamic_axes.setdefault(name, {0: "batch_size"} if dynamic_batch else {})

    with torch.no_grad():
        export_training_mode = training
        if hasattr(torch.onnx, "TrainingMode"):
            export_training_mode = (
                torch.onnx.TrainingMode.TRAINING
                if training
                else torch.onnx.TrainingMode.EVAL
            )
        torch.onnx.export(
            model,
            model_inputs,
            str(output_file),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=list(resolved_input_names),
            output_names=list(output_names_tuple),
            dynamic_axes=dynamic_axes,
            dynamic_shapes=None,
            training=export_training_mode,
            dynamo=False,
            verbose=False,
        )

    if validate:
        validate_onnx_export(
            output_file, model, sample_inputs, dynamic_batch=dynamic_batch
        )

    return output_file


def _infer_model_output_count(model: Any, sample_inputs: Any) -> int:
    torch = _ensure_torch_available()
    with torch.no_grad():
        outputs = (
            model(*sample_inputs)
            if isinstance(sample_inputs, tuple)
            else model(sample_inputs)
        )
    if isinstance(outputs, Mapping):
        return len(outputs)
    if isinstance(outputs, (list, tuple)):
        return len(outputs)
    return 1


def validate_onnx_export(
    onnx_path: str | Path,
    model: Any,
    sample_inputs: Any,
    *,
    dynamic_batch: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Validate an ONNX export against a PyTorch reference model.

    Checks the graph with `onnx.checker`, runs the sample inputs through Pyreth's
    `InferenceSession`, and compares the ONNX results with the reference outputs.

    Args:
        onnx_path: Path to the exported ONNX model.
        model: PyTorch module that produced the export.
        sample_inputs: Inputs in the same format accepted by
            :func:`export_torch_model_to_onnx`.
        dynamic_batch: Whether the export was requested with a dynamic batch
            dimension. The returned value also reflects the model metadata.
        rtol: Relative tolerance used for numerical output comparison.
        atol: Absolute tolerance used for numerical output comparison.

    Returns:
        A dictionary with the model path, dynamic batch status, input and output
        counts, and graph name.

    Raises:
        TypeError: If `model` is not a PyTorch module or output dtypes differ.
        ValueError: If the graph is invalid or ONNX and PyTorch outputs differ in
            count, shape, or numerical values.
        ModuleNotFoundError: If PyTorch or ONNX is unavailable.
    """

    onnx = _ensure_onnx_available()
    torch = _ensure_torch_available()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    model_path = Path(onnx_path)
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)

    sample_inputs_norm = _normalize_inputs(sample_inputs)
    if isinstance(sample_inputs_norm, Mapping):
        named_inputs = {
            str(k): _as_torch_tensor(v) for k, v in sample_inputs_norm.items()
        }
        expected = model(**named_inputs)
        runtime_inputs = {k: v.detach().cpu().numpy() for k, v in named_inputs.items()}
    else:
        model_inputs = tuple(_as_torch_tensor(v) for v in sample_inputs_norm)
        expected = model(*model_inputs)
        runtime_inputs = {
            f"input_{idx}": value.detach().cpu().numpy()
            for idx, value in enumerate(model_inputs)
        }

    session = InferenceSession(
        str(model_path),
        SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        ),
    )

    actual_outputs = session.run_numpy(runtime_inputs)
    expected_outputs = _normalize_expected_outputs(expected)
    if len(actual_outputs) != len(expected_outputs):
        raise ValueError(
            "PyTorch and ONNX export output counts do not match: "
            f"{len(actual_outputs)} vs {len(expected_outputs)}"
        )

    for actual, expected_tensor in zip(actual_outputs, expected_outputs, strict=True):
        expected_array = expected_tensor.detach().cpu().numpy()
        if actual.shape != expected_array.shape:
            raise ValueError(
                f"Output shape mismatch: ONNX output {actual.shape} != expected {expected_array.shape}"
            )
        if not hasattr(actual, "dtype"):
            raise TypeError("Pyreth runtime did not return NumPy-compatible outputs")
        if actual.dtype != expected_array.dtype:
            raise TypeError(
                f"Output dtype mismatch: ONNX output {actual.dtype} != expected {expected_array.dtype}"
            )
        if not np.allclose(actual, expected_array, rtol=rtol, atol=atol):
            raise ValueError(
                "PyTorch reference output does not match the exported ONNX model output."
            )

    info = session.model_info
    return {
        "model_path": str(model_path),
        "dynamic_batch": dynamic_batch and info.has_dynamic_batch,
        "input_count": info.input_count,
        "output_count": info.output_count,
        "has_dynamic_batch": bool(info.has_dynamic_batch),
        "graph_name": info.graph_name,
    }


def _normalize_expected_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, Mapping):
        return tuple(outputs.values())
    if isinstance(outputs, (list, tuple)):
        return tuple(outputs)
    return (outputs,)


def load_onnx_session_from_torch(
    model: Any,
    sample_inputs: Any,
    output_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force_export: bool = False,
    dynamic_batch: bool = True,
    session_config: SessionConfig | None = None,
    validate: bool = True,
) -> InferenceSession:
    """Export a PyTorch module to ONNX and load it as a Pyreth session.

    Args:
        model: PyTorch model instance to export.
        sample_inputs: Sample inputs used for export and optional validation.
        output_path: Destination for the ONNX file. If omitted, a filename based on
            the model class is created in the current working directory.
        dynamic_batch: Whether to export a dynamic batch dimension.
        session_config: Configuration for the returned inference session. If
            omitted, a single-threaded CPU session is used.
        validate: Whether to validate the exported model before creating the
            session.

    Returns:
        An inference session that loads the exported ONNX model.

    Raises:
        TypeError: If `model` is not a PyTorch module.
        ValueError: If the export contract is invalid or validation fails.
        ModuleNotFoundError: If PyTorch is unavailable, or if validation is
            requested and ONNX is unavailable.
    """

    export_path = export_torch_model_to_onnx(
        model,
        sample_inputs,
        output_path,
        cache_dir=cache_dir,
        force_export=force_export,
        dynamic_batch=dynamic_batch,
        validate=validate,
    )
    if session_config is None:
        session_config = SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        )
    return InferenceSession(str(export_path), session_config)
