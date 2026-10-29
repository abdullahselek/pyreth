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
    export_torch_model_to_onnx,
    load_onnx_session_from_torch,
    validate_onnx_export,
)


def _from_torch(
    model,
    sample_inputs,
    output_path=None,
    *,
    session_config=None,
    dynamic_batch=True,
    validate=True,
    **kwargs,
):
    """Export a PyTorch model to ONNX and load it with Pyreth."""
    export_path = export_torch_model_to_onnx(
        model,
        sample_inputs,
        output_path or f"{model.__class__.__name__.lower()}_pyreth_export.onnx",
        dynamic_batch=dynamic_batch,
        validate=validate,
        **kwargs,
    )
    if session_config is None:
        session_config = SessionConfig(
            intra_op_num_threads=1,
            inter_op_num_threads=1,
            providers=[ExecutionProvider.CPU],
        )
    return InferenceSession(str(export_path), session_config)


InferenceSession.from_torch = staticmethod(_from_torch)

__all__ = [
    "ArenaAllocator",
    "Backend",
    "BatcherConfig",
    "BatcherMetrics",
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
    "ThreadBudget",
    "ThreadPool",
    "ThreadPoolConfig",
    "ThreadPoolMetrics",
    "__version__",
    "export_torch_model_to_onnx",
    "get_best_backend",
    "get_runtime_version",
    "has_hardware_acceleration",
    "load_onnx_session_from_torch",
    "validate_onnx_export",
]
