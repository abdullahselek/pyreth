"""Sample 02: Runtime, Hardware Backend Discovery, Model Inspection & Sync Inference.

Demonstrates:
  1. Initializing `Runtime` and querying `Backend` for execution providers
  2. Configuring `SessionConfig` and loading an `InferenceSession` (from bytes or file)
  3. Inspecting `ModelInfo` and `TensorInfo` schemas and validating runtime inputs
  4. Running synchronous zero-copy inference via:
     - Positional inputs (`session.run`)
     - Named dictionary inputs (`session.run_numpy`)
     - Reusable `ArenaAllocator` output buffers (`session.run(..., arena=arena)`)
     - In-place pre-allocated output tensors (`session.run(..., outputs=[out])`)

Usage:
    uv run python samples/02_model_inspection_and_sync_inference.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pyreth
from models import build_mlp_onnx_bytes, expected_mlp_numpy
from pyreth import (
    ArenaAllocator,
    Backend,
    DataType,
    ExecutionProvider,
    InferenceSession,
    LogLevel,
    OptimizationLevel,
    Runtime,
    SessionConfig,
    SessionExecutionMode,
    StorageType,
    Tensor,
)


def demo_runtime_and_backends() -> None:
    print("=== 1. Runtime Initialization & Hardware Backend Discovery ===")

    rt = Runtime.instance()
    rt.init(log_level=LogLevel.Warning, log_id="PyrethSample")

    print(f"Pyreth version        : {pyreth.__version__}")
    print(f"ONNX Runtime version  : {Backend.get_runtime_version()}")
    print(
        f"Runtime initialized   : {rt.is_initialized} (log_id='{rt.log_id}', level={rt.log_level})"
    )
    print(f"Available providers   : {Backend.get_available_providers()}")
    print(f"Best provider         : {Backend.get_best_available_provider()}")
    print(f"Hardware acceleration : {Backend.has_hardware_acceleration()}")


def demo_model_loading_and_inspection() -> InferenceSession:
    print("\n=== 2. SessionConfig, Model Loading & Metadata Inspection ===")

    model_bytes = build_mlp_onnx_bytes()

    config = SessionConfig(
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        graph_optimization_level=OptimizationLevel.All,
        execution_mode=SessionExecutionMode.Sequential,
        providers=[ExecutionProvider.COREML, ExecutionProvider.CPU],
        allow_provider_fallback=True,
    )

    # Load from a file path on disk (or pass `model_bytes` directly to InferenceSession)
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "sample_mlp.onnx"
        model_path.write_bytes(model_bytes)
        session = InferenceSession(model_path, config)

    info = session.model_info
    print(
        f"Loaded Graph          : '{info.graph_name}' (producer='{info.producer_name}', v{info.version})"
    )
    print(f"Active Providers      : {session.active_providers}")
    print(f"Dynamic Batch Support : {info.has_dynamic_batch}")
    print(f"Custom Metadata       : {info.custom_metadata}")

    for idx, inp in enumerate(session.inputs):
        print(
            f"  Input[{idx}]  name='{inp.name}', dtype={inp.dtype}, "
            f"shape={inp.shape}, symbolic_dims={inp.symbolic_dims}"
        )
    for idx, out in enumerate(session.outputs):
        print(
            f"  Output[{idx}] name='{out.name}', dtype={out.dtype}, "
            f"shape={out.shape}, symbolic_dims={out.symbolic_dims}"
        )

    # Schema validation before execution
    sample_x = np.ones((2, 4), dtype=np.float32)
    session.validate_inputs(sample_x)
    print("Input validation passed for shape (2, 4) float32.")

    try:
        session.validate_inputs(np.ones((2, 99), dtype=np.float32))
    except ValueError as exc:
        print(f"Caught expected validation error for invalid shape (2, 99): {exc}")

    return session


def demo_synchronous_inference(session: InferenceSession) -> None:
    print("\n=== 3. Synchronous Zero-Copy Inference Modes ===")

    x_np = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [-0.5, 0.0, 3.0, 1.5],
            [2.0, -1.0, 0.5, -2.0],
        ],
        dtype=np.float32,
    )
    expected = expected_mlp_numpy(x_np)

    # Mode A: Positional input -> list[pyreth.Tensor]
    tensor_outputs = session.run(x_np)
    y_view = tensor_outputs[0].numpy()
    np.testing.assert_allclose(y_view, expected, rtol=1e-5, atol=1e-5)
    print(
        f"[A] session.run(x_np)        -> output shape={y_view.shape}, storage={tensor_outputs[0].storage_type}"
    )

    # Mode B: Named dict input -> list[np.ndarray] directly via run_numpy()
    np_outputs = session.run_numpy({"x": x_np})
    np.testing.assert_allclose(np_outputs[0], expected, rtol=1e-5, atol=1e-5)
    print(f"[B] session.run_numpy(dict)  -> output[0] row 0 = {np_outputs[0][0]}")

    # Mode C: ArenaAllocator reuse across an inference loop (zero heap allocations)
    arena = ArenaAllocator(block_size=64 * 1024)
    for step in range(3):
        outs = session.run(x_np, arena=arena)
        assert outs[0].storage_type == StorageType.ARENA
        np.testing.assert_allclose(outs[0].numpy(), expected, rtol=1e-5, atol=1e-5)
        del outs
        arena.reset()  # O(1) pointer bump reset
    print(
        f"[C] ArenaAllocator loop      -> completed 3 iterations (arena bytes_allocated={arena.bytes_allocated})"
    )

    # Mode D: Pre-allocated in-place output Tensor
    preallocated_out = Tensor.zeros((3, 3), DataType.Float32)
    original_ptr = preallocated_out.data_ptr
    returned_outs = session.run(x_np, outputs=[preallocated_out])
    assert returned_outs[0].data_ptr == original_ptr
    np.testing.assert_allclose(preallocated_out.numpy(), expected, rtol=1e-5, atol=1e-5)
    print(f"[D] In-place output Tensor   -> wrote directly to ptr 0x{original_ptr:x}")


def main() -> None:
    demo_runtime_and_backends()
    session = demo_model_loading_and_inspection()
    demo_synchronous_inference(session)


if __name__ == "__main__":
    main()
