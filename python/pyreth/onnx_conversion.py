"""Export PyTorch models to ONNX and validate the resulting artifacts.

This module wraps PyTorch's ONNX exporter with helpers for defining input and
output names, optionally making the batch dimension dynamic, checking graph
validity, comparing runtime outputs with PyTorch, and creating a Pyreth inference
session. PyTorch and ONNX are optional dependencies and are imported only when
these helpers are called.
"""

import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._core import ExecutionProvider, InferenceSession, SessionConfig

DEFAULT_EXPORT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".onnx_cache"

SUPPORTED_DTYPES: frozenset[str] = frozenset(
    {"float32", "float16", "int64", "int32", "bool"}
)

SUPPORTED_RANKS: tuple[int, ...] = (1, 2, 3, 4)

__all__ = [
    "DEFAULT_EXPORT_CACHE_DIR",
    "SUPPORTED_DTYPES",
    "SUPPORTED_RANKS",
    "ConversionMetadata",
    "TensorSpec",
    "build_conversion_contract",
    "check_numerical_equivalence",
    "check_onnx_graph",
    "clear_cache",
    "export_torch_model_to_onnx",
    "get_default_cache_dir",
    "load_onnx_session_from_torch",
    "make_dynamic_axes",
    "resolve_cache_path",
    "validate_onnx_export",
]


@dataclass(frozen=True)
class TensorSpec:
    """Specification of an input or output tensor in the conversion contract.

    Attributes:
        name: Tensor identifier used in the ONNX graph schema.
        shape: Tensor shape tuple. Dynamic dimensions (e.g. dynamic batch) are
            represented by -1 or symbolic strings.
        dtype: Standardized data type string (e.g. 'float32', 'int64').
    """

    name: str
    shape: tuple[int | str, ...]
    dtype: str


@dataclass(frozen=True)
class ConversionMetadata:
    """Standardized metadata and validated contract for PyTorch to ONNX conversion.

    Attributes:
        model_name: Base name of the exported model class.
        inputs: Tuple of input tensor specifications.
        outputs: Tuple of output tensor specifications.
        dynamic_batch: Whether axis 0 is designated as a dynamic batch dimension.
        dynamic_axes: Mapping of tensor names to dynamic axis dictionaries passed to
            the ONNX exporter.
        opset_version: Targeted ONNX opset version.
    """

    model_name: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    dynamic_batch: bool
    dynamic_axes: dict[str, dict[int, str]]
    opset_version: int


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
        import onnx.checker
        import onnx.shape_inference
    except ModuleNotFoundError as exc:  # pragma: no cover - guarded by tests
        raise ModuleNotFoundError(
            "ONNX validation requires the optional 'onnx' dependency."
        ) from exc
    return onnx


def get_default_cache_dir() -> Path:
    """Return the default ONNX model cache directory.

    Checks the `PYRETH_CACHE_DIR` environment variable first. If unset, checks if
    `<repo_root>/benchmarks/.onnx_cache` exists (matching the benchmark scripts),
    and falls back to `<repo_root>/.onnx_cache`.
    """
    env_dir = os.environ.get("PYRETH_CACHE_DIR")
    if env_dir:
        path = Path(env_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    repo_root = Path(__file__).resolve().parents[2]
    benchmarks_cache = repo_root / "benchmarks" / ".onnx_cache"
    if benchmarks_cache.exists():
        return benchmarks_cache

    default_cache = repo_root / ".onnx_cache"
    default_cache.mkdir(parents=True, exist_ok=True)
    return default_cache


def resolve_cache_path(
    model_identifier: str,
    cache_dir: str | Path | None = None,
    suffix: str = ".onnx",
    prefix: str | None = None,
) -> Path:
    """Resolve a sanitized file path inside the model cache directory.

    Normalizes forward slashes and hyphens to underscores, consistent with the
    naming conventions used across the Pyreth benchmark suites.

    Args:
        model_identifier: Model name, class name, or HuggingFace model ID
            (e.g. 'google/mobilenet_v2_1.0_224' or 'prajjwal1/bert-tiny').
        cache_dir: Optional cache directory override.
        suffix: File extension (default: '.onnx').
        prefix: Optional filename prefix (e.g. 'regression__').

    Returns:
        Path inside the designated cache directory.
    """
    cleaned = model_identifier.replace("/", "__").replace("-", "_")
    safe_name = re.sub(r"[^A-Za-z0-9_.]+", "_", cleaned).strip("._")
    if not safe_name:
        safe_name = "model"
    if prefix:
        safe_name = f"{prefix}{safe_name}"
    if not suffix.startswith("."):
        suffix = f".{suffix}"

    target_dir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{safe_name}{suffix}"


def clear_cache(cache_dir: str | Path | None = None, pattern: str = "*.onnx") -> int:
    """Remove cached ONNX models matching pattern from the cache directory.

    Args:
        cache_dir: Target cache directory. If None, uses default cache dir.
        pattern: Glob pattern to remove (default: '*.onnx').

    Returns:
        Count of deleted files.
    """
    target_dir = Path(cache_dir) if cache_dir is not None else get_default_cache_dir()
    if not target_dir.exists():
        return 0
    removed = 0
    for file in target_dir.glob(pattern):
        if file.is_file():
            file.unlink(missing_ok=True)
            removed += 1
    return removed


def make_dynamic_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
    *,
    batch_axis: int = 0,
    batch_dim_name: str = "batch_size",
    additional_axes: Mapping[str, Mapping[int, str]] | None = None,
) -> dict[str, dict[int, str]]:
    """Build a deterministic dynamic_axes dictionary for inputs and outputs.

    Assigns `batch_dim_name` at `batch_axis` for all inputs and outputs, and merges
    any optional tensor-specific dynamic axes (e.g. sequence length).

    Args:
        input_names: Sequence of input tensor names.
        output_names: Sequence of output tensor names.
        batch_axis: Axis index for batch dimension (default: 0).
        batch_dim_name: Symbolic name for the dynamic batch dimension.
        additional_axes: Optional mapping of tensor names to extra dynamic axes.

    Returns:
        Dictionary mapping tensor names to their dynamic axis specifications.
    """
    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in list(input_names) + list(output_names):
        dynamic_axes[name] = {batch_axis: batch_dim_name}
    if additional_axes:
        for tensor_name, axes in additional_axes.items():
            dynamic_axes.setdefault(tensor_name, {}).update(axes)
    return dynamic_axes


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


def _validate_tensor_dtype(tensor: Any, name: str) -> str:
    dtype_str = str(tensor.dtype).removeprefix("torch.")
    if dtype_str not in SUPPORTED_DTYPES:
        raise TypeError(
            f"Unsupported dtype '{dtype_str}' for tensor '{name}'. "
            f"Supported dtypes are: {sorted(SUPPORTED_DTYPES)}"
        )
    return dtype_str


def _validate_tensor_rank(tensor: Any, name: str) -> int:
    rank = getattr(tensor, "ndim", len(getattr(tensor, "shape", ())))
    if rank not in SUPPORTED_RANKS:
        raise ValueError(
            f"Unsupported rank {rank} for tensor '{name}' with shape "
            f"{tuple(getattr(tensor, 'shape', ()))}. "
            f"Supported ranks are 1D, 2D, 3D, and 4D."
        )
    return rank


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

    model_name = (
        _sanitize_export_name(model.__class__.__name__)
        if model is not None
        else default_name or "exported_model"
    )
    return resolve_cache_path(model_name, cache_dir=cache_dir)


def build_conversion_contract(
    model: Any,
    sample_inputs: Any,
    *,
    dynamic_batch: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> ConversionMetadata:
    """Validate and build standardized conversion metadata for a PyTorch model.

    Evaluates the model with the given sample inputs to extract and validate input
    and output tensor specifications (ranks, dtypes, shapes, dynamic axes), verifying
    the compatibility matrix and failure semantics prior to ONNX graph export.

    Args:
        model: PyTorch model instance (`torch.nn.Module`).
        sample_inputs: Single tensor, tuple/list of tensors, or dict mapping names
            to tensors.
        dynamic_batch: Whether to assign a dynamic `batch_size` axis at dimension 0.
        dynamic_axes: Optional explicit dynamic axis specifications.
        opset_version: Targeted ONNX opset version.
        input_names: Optional sequence of names overriding input tensor names.
        output_names: Optional sequence of names overriding output tensor names.

    Returns:
        A frozen `ConversionMetadata` instance containing verified input and output
        tensor specifications, dynamic axes, and model metadata.

    Raises:
        TypeError: If `model` is not an instance of `torch.nn.Module`, or if any
            input/output tensor has an unsupported dtype.
        ValueError: If `sample_inputs` is empty, tensor ranks are not in {1, 2, 3, 4},
            or name counts do not match input/output counts.
        ModuleNotFoundError: If `torch` is unavailable.
    """

    torch = _ensure_torch_available()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    sample_inputs_norm = _normalize_inputs(sample_inputs)
    if isinstance(sample_inputs_norm, Mapping):
        model_inputs = tuple(
            _as_torch_tensor(v).contiguous() for v in sample_inputs_norm.values()
        )
        resolved_input_names = tuple(str(k) for k in sample_inputs_norm)
    else:
        model_inputs = tuple(
            _as_torch_tensor(v).contiguous() for v in sample_inputs_norm
        )
        resolved_input_names = (
            tuple(str(name) for name in input_names)
            if input_names is not None
            else _model_input_names(sample_inputs)
        )

    if input_names is not None and len(input_names) != len(model_inputs):
        raise ValueError(
            "input_names must match the number of input tensors; "
            f"got {len(input_names)} names for {len(model_inputs)} inputs"
        )

    if not model_inputs:
        raise ValueError("At least one sample input tensor is required for export.")

    input_specs: list[TensorSpec] = []
    computed_dynamic_axes: dict[str, dict[int, str]] = {}
    for name, tensor in zip(resolved_input_names, model_inputs, strict=True):
        dtype_str = _validate_tensor_dtype(tensor, name)
        rank = _validate_tensor_rank(tensor, name)
        shape: tuple[int | str, ...] = (
            (-1, *tuple(tensor.shape)[1:])
            if dynamic_batch and rank > 0
            else tuple(tensor.shape)
        )
        input_specs.append(TensorSpec(name=name, shape=shape, dtype=dtype_str))
        computed_dynamic_axes[name] = {0: "batch_size"} if dynamic_batch else {}

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(sample_inputs_norm, Mapping):
                named_kwargs = {
                    name: tensor
                    for name, tensor in zip(
                        resolved_input_names, model_inputs, strict=True
                    )
                }
                raw_outputs = model(**named_kwargs)
            else:
                raw_outputs = model(*model_inputs)
    finally:
        if was_training:
            model.train()

    output_tensors = _normalize_expected_outputs(raw_outputs)
    if not output_tensors:
        raise ValueError("Model execution produced no outputs.")

    if output_names is not None:
        if len(output_names) != len(output_tensors):
            raise ValueError(
                "output_names must match the number of model outputs; "
                f"got {len(output_names)} names for {len(output_tensors)} outputs"
            )
        resolved_output_names = tuple(str(name) for name in output_names)
    elif isinstance(raw_outputs, Mapping):
        resolved_output_names = tuple(str(k) for k in raw_outputs)
    else:
        resolved_output_names = tuple(
            f"output_{idx}" if len(output_tensors) > 1 else "output"
            for idx in range(len(output_tensors))
        )

    output_specs: list[TensorSpec] = []
    for name, out_tensor in zip(resolved_output_names, output_tensors, strict=True):
        if not isinstance(out_tensor, torch.Tensor):
            out_tensor = torch.as_tensor(out_tensor)
        dtype_str = _validate_tensor_dtype(out_tensor, name)
        rank = _validate_tensor_rank(out_tensor, name)
        out_shape: tuple[int | str, ...] = (
            (-1, *tuple(out_tensor.shape)[1:])
            if dynamic_batch and rank > 0
            else tuple(out_tensor.shape)
        )
        output_specs.append(TensorSpec(name=name, shape=out_shape, dtype=dtype_str))
        computed_dynamic_axes.setdefault(
            name, {0: "batch_size"} if dynamic_batch else {}
        )

    if dynamic_axes is not None:
        for tensor_name, axes_map in dynamic_axes.items():
            computed_dynamic_axes.setdefault(tensor_name, {}).update(axes_map)

    return ConversionMetadata(
        model_name=model.__class__.__name__,
        inputs=tuple(input_specs),
        outputs=tuple(output_specs),
        dynamic_batch=dynamic_batch,
        dynamic_axes=computed_dynamic_axes,
        opset_version=opset_version,
    )


def export_torch_model_to_onnx(
    model: Any,
    sample_inputs: Any,
    output_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    force_export: bool = False,
    dynamic_batch: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    validate: bool = True,
    training: bool = False,
    seed: int | None = None,
) -> Path:
    """Export a PyTorch module to an ONNX file with deterministic settings.

    Ensures evaluation mode, constant folding, reproducible dynamic batch axes,
    atomic file writing, and numerical equivalence validation against Pyreth runtime.

    Args:
        model: PyTorch model instance to export.
        sample_inputs: Sample input tensor, sequence of tensors, or dict mapping
            names to tensors.
        output_path: Destination path for the `.onnx` file. If omitted, resolved
            in `cache_dir`.
        cache_dir: Directory used when output_path is omitted.
        force_export: Overwrite existing cached ONNX file if True.
        dynamic_batch: Whether to assign axis 0 as dynamic `batch_size`.
        dynamic_axes: Optional custom dynamic axes specifications.
        opset_version: ONNX opset version passed to `torch.onnx.export`.
        input_names: Names for positional inputs.
        output_names: Names for model outputs.
        validate: Whether to validate the ONNX graph and compare its outputs with
            the PyTorch model.
        training: Training mode passed to `torch.onnx.export`.
        seed: Optional random seed for deterministic PyTorch tracing.

    Returns:
        The path to the generated ONNX model file.

    Raises:
        TypeError: If `model` is not a PyTorch `nn.Module` or uses unsupported dtypes.
        ValueError: If sample inputs, tensor ranks, or export naming contracts are invalid.
        ModuleNotFoundError: If PyTorch or ONNX is unavailable.
    """

    torch = _ensure_torch_available()
    if seed is not None:
        torch.manual_seed(seed)

    contract = build_conversion_contract(
        model,
        sample_inputs,
        dynamic_batch=dynamic_batch,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
    )

    sample_inputs_norm = _normalize_inputs(sample_inputs)
    if isinstance(sample_inputs_norm, Mapping):
        model_inputs = tuple(
            _as_torch_tensor(v).contiguous() for v in sample_inputs_norm.values()
        )
    else:
        model_inputs = tuple(
            _as_torch_tensor(v).contiguous() for v in sample_inputs_norm
        )

    resolved_input_names = [spec.name for spec in contract.inputs]
    output_names_list = [spec.name for spec in contract.outputs]

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
            output_file,
            model,
            sample_inputs,
            dynamic_batch=dynamic_batch,
            contract=contract,
        )
        return output_file

    was_training = model.training
    if not training:
        model.eval()

    tmp_output = output_file.with_suffix(".tmp.onnx")
    try:
        with torch.no_grad():
            export_training_mode = training
            if hasattr(torch.onnx, "TrainingMode"):
                export_training_mode = (
                    torch.onnx.TrainingMode.TRAINING
                    if training
                    else torch.onnx.TrainingMode.EVAL
                )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                torch.onnx.export(
                    model,
                    model_inputs,
                    str(tmp_output),
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=resolved_input_names,
                    output_names=output_names_list,
                    dynamic_axes=contract.dynamic_axes,
                    dynamic_shapes=None,
                    training=export_training_mode,
                    dynamo=False,
                    verbose=False,
                )
        # Atomically replace destination path to avoid partial or corrupted files
        tmp_output.replace(output_file)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise
    finally:
        if was_training and not training:
            model.train()

    if validate:
        validate_onnx_export(
            output_file,
            model,
            sample_inputs,
            dynamic_batch=dynamic_batch,
            contract=contract,
        )

    return output_file


def check_onnx_graph(onnx_path: str | Path) -> dict[str, Any]:
    """Validate ONNX graph integrity, schema consistency, and shape inference.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        Dictionary containing graph metadata (model_path, graph_name, ir_version,
        opset_version, producer_name, input_names, output_names).

    Raises:
        FileNotFoundError: If `onnx_path` does not exist.
        ValueError: If `onnx.checker` or shape inference detects graph corruption.
        ModuleNotFoundError: If `onnx` is unavailable.
    """
    onnx = _ensure_onnx_available()
    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model file not found at: {path}")

    try:
        model_proto = onnx.load(str(path))
    except Exception as exc:
        raise ValueError(
            f"Failed to load ONNX model protobuf from '{path}': {exc}"
        ) from exc

    onnx.checker.check_model(model_proto)

    try:
        inferred = onnx.shape_inference.infer_shapes(model_proto)
        onnx.checker.check_model(inferred)
    except Exception as exc:
        raise ValueError(f"ONNX shape inference failed for '{path}': {exc}") from exc

    opset = None
    for entry in model_proto.opset_import:
        if entry.domain in ("", "ai.onnx"):
            opset = entry.version
            break

    input_names = [inp.name for inp in model_proto.graph.input]
    output_names = [out.name for out in model_proto.graph.output]

    return {
        "model_path": str(path),
        "graph_name": model_proto.graph.name,
        "ir_version": model_proto.ir_version,
        "opset_version": opset,
        "producer_name": model_proto.producer_name,
        "input_names": input_names,
        "output_names": output_names,
    }


def check_numerical_equivalence(
    model: Any,
    onnx_path: str | Path,
    sample_inputs: Any,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    session_config: SessionConfig | None = None,
) -> dict[str, Any]:
    """Check numerical parity between a PyTorch reference module and exported ONNX model.

    Evaluates the model in evaluation mode with no gradients, runs inference via
    Pyreth's zero-copy `InferenceSession`, and computes maximum/mean absolute and
    relative discrepancies.

    Args:
        model: Reference PyTorch `torch.nn.Module`.
        onnx_path: Path to the exported ONNX model.
        sample_inputs: Test inputs (tensor, sequence, or dict).
        rtol: Relative numerical tolerance.
        atol: Absolute numerical tolerance.
        session_config: Optional custom `SessionConfig`.

    Returns:
        Dictionary containing max_abs_diff, mean_abs_diff, max_rel_diff, and matches.

    Raises:
        TypeError: If model is not an nn.Module or outputs are incompatible.
        ValueError: If outputs differ in count, shape, or exceed numerical tolerances.
    """
    torch = _ensure_torch_available()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")

    model_path = Path(onnx_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Exported model not found at: {model_path}")

    sample_inputs_norm = _normalize_inputs(sample_inputs)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(sample_inputs_norm, Mapping):
                named_inputs = {
                    str(k): _as_torch_tensor(v).contiguous()
                    for k, v in sample_inputs_norm.items()
                }
                expected = model(**named_inputs)
                runtime_inputs: Any = {
                    k: v.detach().cpu().numpy() for k, v in named_inputs.items()
                }
            else:
                model_inputs = tuple(
                    _as_torch_tensor(v).contiguous() for v in sample_inputs_norm
                )
                expected = model(*model_inputs)
                runtime_inputs = [
                    value.detach().cpu().numpy() for value in model_inputs
                ]
    finally:
        if was_training:
            model.train()

    if session_config is None:
        session_config = SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        )

    session = InferenceSession(str(model_path), session_config)
    actual_outputs = session.run_numpy(runtime_inputs)
    expected_outputs = _normalize_expected_outputs(expected)

    if len(actual_outputs) != len(expected_outputs):
        raise ValueError(
            "PyTorch and ONNX export output counts do not match: "
            f"{len(actual_outputs)} vs {len(expected_outputs)}"
        )

    max_abs_diff = 0.0
    total_abs_diff = 0.0
    total_elements = 0
    max_rel_diff = 0.0

    for actual, expected_tensor in zip(actual_outputs, expected_outputs, strict=True):
        if not isinstance(expected_tensor, torch.Tensor):
            expected_tensor = torch.as_tensor(expected_tensor)
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

        if np.issubdtype(actual.dtype, np.number):
            abs_diff = np.abs(
                actual.astype(np.float64) - expected_array.astype(np.float64)
            )
            max_abs_diff = max(max_abs_diff, float(np.max(abs_diff)))
            total_abs_diff += float(np.sum(abs_diff))
            total_elements += abs_diff.size

            denom = np.abs(expected_array.astype(np.float64))
            denom = np.where(denom == 0, 1.0, denom)
            rel_diff = abs_diff / denom
            max_rel_diff = max(max_rel_diff, float(np.max(rel_diff)))

        if not np.allclose(actual, expected_array, rtol=rtol, atol=atol):
            raise ValueError(
                f"PyTorch reference output does not match exported ONNX model within "
                f"tolerance (rtol={rtol}, atol={atol}). Max abs diff: {max_abs_diff:.6e}, "
                f"Max rel diff: {max_rel_diff:.6e}."
            )

    mean_abs_diff = (total_abs_diff / total_elements) if total_elements > 0 else 0.0
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "matches": True,
    }


def validate_onnx_export(
    onnx_path: str | Path,
    model: Any,
    sample_inputs: Any,
    *,
    dynamic_batch: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    contract: ConversionMetadata | None = None,
) -> dict[str, Any]:
    """Validate an ONNX export against a PyTorch reference model.

    Combines graph integrity validation via `check_onnx_graph` and numerical parity
    checks via `check_numerical_equivalence`, confirming dynamic batch and IO contracts.

    Args:
        onnx_path: Path to the exported ONNX model.
        model: PyTorch module that produced the export.
        sample_inputs: Inputs in the same format accepted by
            :func:`export_torch_model_to_onnx`.
        dynamic_batch: Whether the export was requested with a dynamic batch
            dimension. The returned value also reflects the model metadata.
        rtol: Relative tolerance used for numerical output comparison.
        atol: Absolute tolerance used for numerical output comparison.
        contract: Optional precomputed `ConversionMetadata` instance.

    Returns:
        A dictionary with graph info, dynamic batch status, input and output counts,
        and numerical divergence metrics.

    Raises:
        TypeError: If `model` is not a PyTorch module or output dtypes differ.
        ValueError: If the graph is invalid or ONNX and PyTorch outputs differ in
            count, shape, or numerical values.
        ModuleNotFoundError: If PyTorch or ONNX is unavailable.
    """
    _ = check_onnx_graph(onnx_path)
    numerical_metrics = check_numerical_equivalence(
        model, onnx_path, sample_inputs, rtol=rtol, atol=atol
    )

    session = InferenceSession(
        str(onnx_path),
        SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        ),
    )
    info = session.model_info

    return {
        "model_path": str(onnx_path),
        "dynamic_batch": dynamic_batch and info.has_dynamic_batch,
        "input_count": info.input_count,
        "output_count": info.output_count,
        "has_dynamic_batch": bool(info.has_dynamic_batch),
        "graph_name": info.graph_name,
        "max_abs_diff": numerical_metrics["max_abs_diff"],
        "mean_abs_diff": numerical_metrics["mean_abs_diff"],
        "max_rel_diff": numerical_metrics["max_rel_diff"],
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
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
    session_config: SessionConfig | None = None,
    validate: bool = True,
    opset_version: int = 17,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> InferenceSession:
    """Export a PyTorch module to ONNX and load it as a Pyreth session.

    Args:
        model: PyTorch model instance to export.
        sample_inputs: Sample inputs used for export and optional validation.
        output_path: Destination for the ONNX file. If omitted, resolved in `cache_dir`.
        cache_dir: Directory used when output_path is omitted.
        force_export: Overwrite existing cached ONNX file if True.
        dynamic_batch: Whether to export a dynamic batch dimension.
        dynamic_axes: Optional explicit dynamic axes mapping.
        session_config: Configuration for the returned inference session.
        validate: Whether to validate the exported model before creating the session.
        opset_version: ONNX opset version passed to the exporter.
        input_names: Optional sequence of input names.
        output_names: Optional sequence of output names.
        seed: Optional random seed for deterministic PyTorch tracing.
        **kwargs: Additional parameters passed to `export_torch_model_to_onnx`.

    Returns:
        An inference session that loads the exported ONNX model.

    Raises:
        TypeError: If `model` is not a PyTorch module.
        ValueError: If the export contract is invalid or validation fails.
        ModuleNotFoundError: If PyTorch or ONNX is unavailable.
    """

    export_path = export_torch_model_to_onnx(
        model,
        sample_inputs,
        output_path,
        cache_dir=cache_dir,
        force_export=force_export,
        dynamic_batch=dynamic_batch,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        validate=validate,
        seed=seed,
        **kwargs,
    )
    if session_config is None:
        session_config = SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        )
    return InferenceSession(str(export_path), session_config)
