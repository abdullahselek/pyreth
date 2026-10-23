"""Sample 05: Advanced Tensor DTypes, Granular Schema Validation, Fluent Configs & Lifecycle.

Demonstrates:
  1. Multi-dtype Tensors (`Float16`, `BFloat16`, `Int8`, `Bool`, dtype strings),
     0-D scalars (`Shape.is_scalar`), `np.asarray(tensor)`, `zero()`,
     arena-backed `clone(arena)` / `to_contiguous(arena)`, and `ArenaAllocator.force_reset()`
  2. Granular `ModelInfo` & `TensorInfo` schema queries (`find_input`, `get_input_index`,
     `is_dynamic_dim`, `fixed_element_count`, `validate_shape`, `validate_tensor`, `validate_input`)
  3. Top-level backend helpers, fluent `SessionConfig` builder with ONNX Runtime profiling,
     and `MemoryPool`-backed synchronous inference (`session.run(x, pool=mem_pool)`)
  4. Fluent `DynamicBatcherConfig` builder, `DynamicBatcher.create()`, `InferenceFuture`
     methods (`is_ready`, `done`, `wait`, `get`, `result`), `reset_metrics()`, and `Runtime.shutdown()`

Usage:
    uv run python samples/05_advanced_features.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pyreth
from models import build_mlp_onnx_bytes, expected_mlp_numpy
from pyreth import (
    ArenaAllocator,
    Backend,
    BatcherConfig,
    DataType,
    DynamicBatcher,
    ExecutionProvider,
    InferenceSession,
    LogLevel,
    MemoryPool,
    OptimizationLevel,
    Runtime,
    SessionConfig,
    SessionExecutionMode,
    Shape,
    StorageType,
    Tensor,
)


def demo_advanced_tensor_and_arena() -> None:
    print("=== 1. Multi-DType Tensors, 0-D Scalars, Arena Clone & Force Reset ===")

    # 1a. Create tensors across diverse DataTypes (including dtype strings and np.dtype)
    t_f16 = Tensor.zeros((2, 4), DataType.Float16)
    t_bf16 = Tensor.zeros((2, 4), DataType.BFloat16)
    t_f64 = Tensor.zeros((2, 4), "float64")
    t_i32 = Tensor.create((2, 4), np.int32)
    t_i32.fill(7)
    t_bool = Tensor.zeros((2, 4), DataType.Bool)
    t_bool.fill(1)

    print(
        f"Created dtypes: f16={t_f16.dtype} ({t_f16.itemsize}B), "
        f"bf16={t_bf16.dtype} ({t_bf16.itemsize}B), f64={t_f64.dtype} ({t_f64.itemsize}B), "
        f"i32={t_i32.dtype}, bool={t_bool.dtype}"
    )

    # Zero-copy NumPy __array__ protocol via np.asarray() and .to_numpy()
    i32_arr = np.asarray(t_i32)
    assert i32_arr.__array_interface__["data"][0] == t_i32.data_ptr
    t_i32.zero()
    assert np.all(t_i32.to_numpy() == 0)

    # 1b. 0-D scalar Shape and Tensor
    scalar_shape = Shape([])
    scalar_np = np.array(42.5, dtype=np.float32)
    scalar_tensor = Tensor(scalar_np)
    print(
        f"0-D Scalar: is_scalar={scalar_shape.is_scalar()}, ndim={scalar_tensor.ndim}, "
        f"shape={scalar_tensor.shape}, value={float(scalar_tensor.numpy())}"
    )

    # 1c. Arena-backed clone(), to_contiguous(arena), and force_reset()
    arena = ArenaAllocator(block_size=64 * 1024)
    base = Tensor.from_numpy(np.arange(16, dtype=np.float32).reshape(4, 4))
    strided = base.slice(dim=1, start=0, end=4, step=2)

    contig_in_arena = strided.to_contiguous(arena=arena)
    cloned_in_arena = contig_in_arena.clone(arena=arena)
    print(
        f"Arena stats: blocks={arena.block_count}, reserved={arena.bytes_reserved}B, "
        f"allocated={arena.bytes_allocated}B, pins={arena.pin_count}"
    )
    assert contig_in_arena.storage_type == StorageType.ARENA
    assert cloned_in_arena.storage_type == StorageType.ARENA

    # Drop references and demonstrate force_reset()
    del contig_in_arena, cloned_in_arena
    arena.force_reset()
    print(f"After arena.force_reset(): allocated={arena.bytes_allocated}B")


def demo_granular_schema_inspection() -> None:
    print("\n=== 2. Granular ModelInfo & TensorInfo Schema Lookup & Validation ===")

    session = InferenceSession.create(build_mlp_onnx_bytes())
    info = session.model_info

    # Lookup by name and index
    in_idx = info.get_input_index("x")
    out_idx = info.get_output_index("y")
    in_info = info.find_input("x")
    out_info = session.get_output_info(out_idx)
    assert in_info is not None
    assert in_info == session.get_input_info(in_idx)
    assert info.find_input("non_existent") is None

    print(
        f"Input 'x' (index={in_idx}): rank={in_info.rank}, has_dynamic_shape={in_info.has_dynamic_shape}, "
        f"is_dynamic_dim(0)={in_info.is_dynamic_dim(0)}, is_dynamic_dim(1)={in_info.is_dynamic_dim(1)}, "
        f"fixed_element_count={in_info.fixed_element_count()}"
    )
    print(f"Output 'y' (index={out_idx}): {out_info}")

    # Validate shapes and tensors directly against TensorInfo and ModelInfo
    in_info.validate_shape((16, 4))
    sample_x = np.ones((8, 4), dtype=np.float32)
    in_info.validate_tensor(sample_x)
    info.validate_input("x", sample_x)
    info.validate_input(0, sample_x)
    print("Granular shape and tensor validations succeeded for (16, 4) and (8, 4).")


def demo_fluent_configs_profiling_and_pooled_sync() -> None:
    print(
        "\n=== 3. Top-Level Helpers, Fluent SessionConfig, Profiling & Pooled Sync Run ==="
    )

    print(
        f"Top-level helpers: runtime={pyreth.get_runtime_version()}, "
        f"best_backend={pyreth.get_best_backend()}, "
        f"hw_accel={pyreth.has_hardware_acceleration()}, "
        f"cpu_available={Backend.is_provider_available(ExecutionProvider.CPU) and Backend.is_provider_available('CPUExecutionProvider')}"
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        profile_prefix = str(Path(tmp_dir) / "pyreth_ort_profile")

        # Fluent method chaining on SessionConfig
        cfg = SessionConfig()
        (
            cfg.set_intra_op_num_threads(1)
            .set_inter_op_num_threads(1)
            .set_graph_optimization_level(OptimizationLevel.All)
            .set_execution_mode(SessionExecutionMode.Sequential)
            .set_enable_cpu_mem_arena(True)
            .set_enable_mem_pattern(True)
            .set_log_severity_level(LogLevel.Warning)
            .set_log_id("FluentSession")
            .set_providers([ExecutionProvider.CPU])
            .add_provider(ExecutionProvider.CPU)
            .set_allow_provider_fallback(True)
            .set_enable_profiling(profile_prefix)
        )
        assert cfg.enable_profiling

        # Toggle profiling off or keep it enabled for the session
        session = InferenceSession.create(build_mlp_onnx_bytes(), cfg)

        # Synchronous execution using a MemoryPool for zero-copy pooled output allocation
        mem_pool = MemoryPool(initial_count=4, block_size=64 * 1024)
        x = np.array([[1.0, 0.5, -0.5, 2.0]], dtype=np.float32)

        pooled_outs = session.run(x, pool=mem_pool)
        assert pooled_outs[0].storage_type == StorageType.POOLED
        np.testing.assert_allclose(
            pooled_outs[0].numpy(), expected_mlp_numpy(x), rtol=1e-5
        )

        pooled_np_outs = session.run_numpy(x, pool=mem_pool)
        np.testing.assert_allclose(pooled_np_outs[0], expected_mlp_numpy(x), rtol=1e-5)

        print(
            f"Executed pooled synchronous inference: storage={pooled_outs[0].storage_type}, "
            f"pool_available={mem_pool.available_count}"
        )
        del pooled_outs, pooled_np_outs
        del session

        # Verify ORT profile file was written
        profile_files = list(Path(tmp_dir).glob("pyreth_ort_profile*.json"))
        print(f"Generated ONNX Runtime profile trace files: {len(profile_files)}")


def demo_future_and_batcher_lifecycle() -> None:
    print("\n=== 4. Fluent BatcherConfig, Future Polling & Lifecycle Controls ===")

    session = InferenceSession.create(build_mlp_onnx_bytes())

    # Fluent method chaining on BatcherConfig (alias for DynamicBatcherConfig)
    batcher_cfg = BatcherConfig()
    (
        batcher_cfg.set_max_batch_size(4)
        .set_max_batch_delay_ms(5)
        .set_max_batch_delay_us(500)
        .set_max_queue_capacity(128)
        .set_num_workers(2)
        .set_arena_block_size(64 * 1024)
    )

    batcher = DynamicBatcher.create(session, batcher_cfg)
    x = np.array([[0.5, 1.5, -1.0, 0.0]], dtype=np.float32)

    # 4a. Submit via run_async() and use explicit InferenceFuture wait() / is_ready() / done() / result()
    fut1 = batcher.run_async(x)
    fut2 = batcher.submit(x)
    print(
        f"Submitted 2 futures (pending_count={batcher.pending_count}, fut1.valid={fut1.valid})"
    )

    batcher.flush()
    fut1.wait()
    assert fut1.is_ready() and fut1.done()

    # Retrieve list[Tensor] via .result() and .get()
    out_tensors_1 = fut1.result()
    out_tensors_2 = fut2.get()
    np.testing.assert_allclose(
        out_tensors_1[0].numpy(), expected_mlp_numpy(x), rtol=1e-5
    )
    np.testing.assert_allclose(
        out_tensors_2[0].numpy(), expected_mlp_numpy(x), rtol=1e-5
    )

    # 4b. Synchronous batcher.run() returning list[Tensor]
    sync_tensors = batcher.run(x)
    np.testing.assert_allclose(
        sync_tensors[0].numpy(), expected_mlp_numpy(x), rtol=1e-5
    )

    print(f"Batcher metrics before reset : {batcher.metrics()}")
    batcher.reset_metrics()
    print(f"Batcher metrics after reset  : {batcher.metrics()}")

    batcher.shutdown()
    assert batcher.is_shutdown
    del batcher, session

    # Explicitly shut down the global Runtime singleton (can be lazily re-initialized later)
    Runtime.instance().shutdown()
    print(
        f"Runtime shut down cleanly    : is_initialized={Runtime.instance().is_initialized}"
    )


def main() -> None:
    demo_advanced_tensor_and_arena()
    demo_granular_schema_inspection()
    demo_fluent_configs_profiling_and_pooled_sync()
    demo_future_and_batcher_lifecycle()


if __name__ == "__main__":
    main()
