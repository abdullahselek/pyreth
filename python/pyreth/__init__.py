"""Pyreth AI Inference Engine.

High-throughput, low-latency parallel processing AI model inference engine
built with C++20, ONNX Runtime, and nanobind.
"""

from ._core import (
    ArenaAllocator,
    Backend,
    BatcherConfig,
    BatcherMetrics,
    DataType,
    DynamicBatcher,
    DynamicBatcherConfig,
    DynamicBatcherMetrics,
    ExecutionProvider,
    InferenceFuture,
    InferenceSession,
    IOTensorInfo,
    LogLevel,
    MemoryPool,
    ModelInfo,
    OptimizationLevel,
    Runtime,
    SessionConfig,
    SessionExecutionMode,
    Shape,
    StatusCode,
    StorageType,
    Tensor,
    TensorInfo,
    ThreadBudget,
    ThreadPool,
    ThreadPoolConfig,
    ThreadPoolMetrics,
    __version__,
    get_best_backend,
    get_runtime_version,
    has_hardware_acceleration,
)
from .onnx_conversion import (
    DEFAULT_EXPORT_CACHE_DIR,
    SUPPORTED_DTYPES,
    SUPPORTED_RANKS,
    ConversionMetadata,
    TensorSpec,
    build_conversion_contract,
    check_numerical_equivalence,
    check_onnx_graph,
    clear_cache,
    export_torch_model_to_onnx,
    get_default_cache_dir,
    load_onnx_session_from_torch,
    make_dynamic_axes,
    resolve_cache_path,
    validate_onnx_export,
)


def _from_torch(
    model,
    sample_inputs,
    output_path=None,
    *,
    cache_dir=None,
    force_export=False,
    dynamic_batch=True,
    dynamic_axes=None,
    session_config=None,
    validate=True,
    opset_version=17,
    input_names=None,
    output_names=None,
    seed=None,
    **kwargs,
):
    """Export a PyTorch model to ONNX and load it as a Pyreth InferenceSession.

    Args:
        model: PyTorch model instance (torch.nn.Module) to export.
        sample_inputs: Representative input tensor(s) used for tracing and shape inference.
        output_path: Destination path for the exported .onnx file. If omitted, resolved in cache.
        cache_dir: Directory used when output_path is omitted (default: benchmarks/.onnx_cache/).
        force_export: Overwrite existing cached ONNX file if True.
        dynamic_batch: Whether axis 0 is designated as a dynamic batch dimension.
        dynamic_axes: Optional custom dynamic axes dictionary mapping.
        session_config: Optional pyreth.SessionConfig (defaults to 1 intra/inter thread CPU).
        validate: Whether to validate the exported model graph and numerical equivalence.
        opset_version: Target ONNX opset version (default: 17).
        input_names: Optional sequence of custom input node names.
        output_names: Optional sequence of custom output node names.
        seed: Optional random seed for reproducible PyTorch tracing.
        **kwargs: Additional options forwarded to export_torch_model_to_onnx.

    Returns:
        InferenceSession initialized with the exported ONNX model.
    """
    return load_onnx_session_from_torch(
        model=model,
        sample_inputs=sample_inputs,
        output_path=output_path,
        cache_dir=cache_dir,
        force_export=force_export,
        dynamic_batch=dynamic_batch,
        dynamic_axes=dynamic_axes,
        session_config=session_config,
        validate=validate,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        seed=seed,
        **kwargs,
    )


def _tensor_from_torch(torch_tensor):
    """Create a zero-copy Pyreth Tensor from a PyTorch tensor.

    Args:
        torch_tensor: A torch.Tensor instance on CPU.

    Returns:
        pyreth.Tensor sharing the underlying memory buffer.
    """
    try:
        import torch

        if not isinstance(torch_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(torch_tensor).__name__}")
        return Tensor.from_numpy(torch_tensor.detach().cpu().numpy())
    except ImportError as err:
        raise ModuleNotFoundError(
            "PyTorch is required to use Tensor.from_torch"
        ) from err


def _tensor_to_torch(self):
    """Convert this Pyreth Tensor to a PyTorch tensor (zero-copy when contiguous).

    Returns:
        torch.Tensor sharing the underlying memory buffer.
    """
    try:
        import torch

        return torch.from_numpy(self.to_numpy())
    except ImportError as err:
        raise ModuleNotFoundError("PyTorch is required to use Tensor.to_torch") from err


InferenceSession.from_torch = staticmethod(_from_torch)
Tensor.from_torch = staticmethod(_tensor_from_torch)
Tensor.to_torch = _tensor_to_torch
from_torch = _from_torch

__all__ = [
    "DEFAULT_EXPORT_CACHE_DIR",
    "SUPPORTED_DTYPES",
    "SUPPORTED_RANKS",
    "ArenaAllocator",
    "Backend",
    "BatcherConfig",
    "BatcherMetrics",
    "ConversionMetadata",
    "DataType",
    "DynamicBatcher",
    "DynamicBatcherConfig",
    "DynamicBatcherMetrics",
    "ExecutionProvider",
    "IOTensorInfo",
    "InferenceFuture",
    "InferenceSession",
    "LogLevel",
    "MemoryPool",
    "ModelInfo",
    "OptimizationLevel",
    "Runtime",
    "SessionConfig",
    "SessionExecutionMode",
    "Shape",
    "StatusCode",
    "StorageType",
    "Tensor",
    "TensorInfo",
    "TensorSpec",
    "ThreadBudget",
    "ThreadPool",
    "ThreadPoolConfig",
    "ThreadPoolMetrics",
    "__version__",
    "build_conversion_contract",
    "check_numerical_equivalence",
    "check_onnx_graph",
    "clear_cache",
    "export_torch_model_to_onnx",
    "from_torch",
    "get_best_backend",
    "get_default_cache_dir",
    "get_runtime_version",
    "has_hardware_acceleration",
    "load_onnx_session_from_torch",
    "make_dynamic_axes",
    "resolve_cache_path",
    "validate_onnx_export",
]
