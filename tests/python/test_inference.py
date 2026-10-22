import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from pyreth import (
    ArenaAllocator,
    Backend,
    DataType,
    DynamicBatcher,
    DynamicBatcherConfig,
    ExecutionProvider,
    InferenceSession,
    LogLevel,
    MemoryPool,
    OptimizationLevel,
    Runtime,
    SessionConfig,
    SessionExecutionMode,
    StorageType,
    Tensor,
    ThreadPool,
    ThreadPoolConfig,
)

# ---------------------------------------------------------------------------
# Minimal Pure-Python ONNX Protobuf Builder Helpers
# ---------------------------------------------------------------------------


def _encode_varint(val: int) -> bytes:
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _encode_tag(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_varint_field(field_number: int, val: int) -> bytes:
    return _encode_tag(field_number, 0) + _encode_varint(val)


def _encode_bytes_field(field_number: int, data: bytes) -> bytes:
    return _encode_tag(field_number, 2) + _encode_varint(len(data)) + data


def _encode_string_field(field_number: int, text: str) -> bytes:
    raw = text.encode("utf-8")
    return _encode_bytes_field(field_number, raw)


def _encode_value_info(name: str, elem_type: int, dims: list[int | str]) -> bytes:
    shape_bytes = bytearray()
    for d in dims:
        if isinstance(d, str):
            dim_proto = _encode_string_field(2, d)
        else:
            dim_proto = _encode_varint_field(1, d)
        shape_bytes += _encode_bytes_field(1, dim_proto)

    tensor_type = _encode_varint_field(1, elem_type) + _encode_bytes_field(
        2, bytes(shape_bytes)
    )
    type_proto = _encode_bytes_field(1, tensor_type)
    return _encode_string_field(1, name) + _encode_bytes_field(2, type_proto)


def _encode_node(
    inputs: list[str], outputs: list[str], name: str, op_type: str
) -> bytes:
    buf = bytearray()
    for i in inputs:
        buf += _encode_string_field(1, i)
    for o in outputs:
        buf += _encode_string_field(2, o)
    buf += _encode_string_field(3, name)
    buf += _encode_string_field(4, op_type)
    return bytes(buf)


def _encode_float_initializer(name: str, dims: list[int], values: list[float]) -> bytes:
    buf = bytearray()
    for d in dims:
        buf += _encode_varint_field(1, d)
    buf += _encode_varint_field(2, 1)  # FLOAT = 1
    buf += _encode_string_field(8, name)
    raw_floats = struct.pack(f"<{len(values)}f", *values)
    buf += _encode_bytes_field(9, raw_floats)
    return bytes(buf)


def _encode_string_entry(key: str, val: str) -> bytes:
    return _encode_string_field(1, key) + _encode_string_field(2, val)


def build_mlp_onnx_bytes() -> bytes:
    """Build y = Relu(MatMul(x, W) + b) where x: [batch_size, 4] -> y: [batch_size, 3]."""
    w_vals = [
        1.0,
        -1.0,
        0.5,
        0.0,
        2.0,
        -0.5,
        2.0,
        0.0,
        1.0,
        -1.0,
        1.0,
        2.0,
    ]
    b_vals = [0.5, -1.0, 0.0]

    graph = bytearray()
    graph += _encode_bytes_field(
        1, _encode_node(["x", "W"], ["xw"], "mm_node", "MatMul")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["xw", "b"], ["xw_b"], "add_node", "Add")
    )
    graph += _encode_bytes_field(1, _encode_node(["xw_b"], ["y"], "relu_node", "Relu"))
    graph += _encode_string_field(2, "mlp_linear_relu_graph")
    graph += _encode_bytes_field(5, _encode_float_initializer("W", [4, 3], w_vals))
    graph += _encode_bytes_field(5, _encode_float_initializer("b", [3], b_vals))
    graph += _encode_bytes_field(11, _encode_value_info("x", 1, ["batch_size", 4]))
    graph += _encode_bytes_field(12, _encode_value_info("y", 1, ["batch_size", 3]))

    opset = _encode_string_field(1, "") + _encode_varint_field(2, 17)

    model = bytearray()
    model += _encode_varint_field(1, 9)  # ir_version = 9
    model += _encode_string_field(2, "pyreth_pytest_producer")
    model += _encode_string_field(4, "ai.pyreth.test")
    model += _encode_varint_field(5, 3)  # model_version = 3
    model += _encode_string_field(6, "Test MLP model for Pyreth Python bindings")
    model += _encode_bytes_field(7, bytes(graph))
    model += _encode_bytes_field(8, opset)
    model += _encode_bytes_field(14, _encode_string_entry("architecture", "mlp_4x3"))
    return bytes(model)


def build_arithmetic_onnx_bytes() -> bytes:
    """Build dual-input dual-output model: sum = a + b, prod = a * b ([batch_size, 4])."""
    graph = bytearray()
    graph += _encode_bytes_field(
        1, _encode_node(["a", "b"], ["sum"], "add_node", "Add")
    )
    graph += _encode_bytes_field(
        1, _encode_node(["a", "b"], ["prod"], "mul_node", "Mul")
    )
    graph += _encode_string_field(2, "arithmetic_graph")
    graph += _encode_bytes_field(11, _encode_value_info("a", 1, ["batch_size", 4]))
    graph += _encode_bytes_field(11, _encode_value_info("b", 1, ["batch_size", 4]))
    graph += _encode_bytes_field(12, _encode_value_info("sum", 1, ["batch_size", 4]))
    graph += _encode_bytes_field(12, _encode_value_info("prod", 1, ["batch_size", 4]))

    opset = _encode_string_field(1, "") + _encode_varint_field(2, 17)
    model = bytearray()
    model += _encode_varint_field(1, 9)
    model += _encode_string_field(2, "pyreth_pytest")
    model += _encode_bytes_field(7, bytes(graph))
    model += _encode_bytes_field(8, opset)
    return bytes(model)


def expected_mlp_numpy(x: np.ndarray) -> np.ndarray:
    w = np.array(
        [
            [1.0, -1.0, 0.5],
            [0.0, 2.0, -0.5],
            [2.0, 0.0, 1.0],
            [-1.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    b = np.array([0.5, -1.0, 0.0], dtype=np.float32)
    return np.maximum(0.0, x @ w + b)


# ---------------------------------------------------------------------------
# 1. Runtime, Backend & SessionConfig Tests
# ---------------------------------------------------------------------------


def test_runtime_and_backend_discovery():
    rt = Runtime.instance()
    rt.init(LogLevel.Warning, "PyrethPythonTest")
    assert rt.is_initialized
    assert rt.log_level == LogLevel.Warning

    providers = Backend.get_available_execution_providers()
    assert ExecutionProvider.CPU in providers
    assert Backend.is_provider_available(ExecutionProvider.CPU)
    assert isinstance(Backend.get_best_available_provider(), ExecutionProvider)
    assert len(Backend.get_available_providers()) >= 1


def test_session_config_and_filesystem_loading(tmp_path: Path):
    model_file = tmp_path / "mlp.onnx"
    model_file.write_bytes(build_mlp_onnx_bytes())

    cfg = SessionConfig(
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        graph_optimization_level=OptimizationLevel.All,
        execution_mode=SessionExecutionMode.Sequential,
        providers=[ExecutionProvider.CPU],
    )
    cfg.set_intra_op_num_threads(1).set_log_severity_level(LogLevel.Warning)

    # Load via pathlib.Path (os.PathLike)
    session = InferenceSession(model_file, cfg)
    assert session.is_valid
    assert session.model_path == str(model_file)
    assert ExecutionProvider.CPU in session.active_providers
    assert session.config.intra_op_num_threads == 1

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        InferenceSession(tmp_path / "nonexistent_model.onnx")

    # Corrupted bytes raise RuntimeError
    with pytest.raises(RuntimeError):
        InferenceSession(b"not_a_valid_onnx_protobuf")


# ---------------------------------------------------------------------------
# 2. ModelInfo & TensorInfo Metadata Inspection Tests
# ---------------------------------------------------------------------------


def test_model_info_and_io_schema_discovery():
    session = InferenceSession.create(build_mlp_onnx_bytes())
    info = session.model_info

    assert info.graph_name == "mlp_linear_relu_graph"
    assert info.producer_name == "pyreth_pytest_producer"
    assert info.domain == "ai.pyreth.test"
    assert info.version == 3
    assert info.custom_metadata.get("architecture") == "mlp_4x3"
    assert info.input_count == 1
    assert info.output_count == 1
    assert info.has_dynamic_batch
    assert info.has_dynamic_inputs
    assert info.has_dynamic_outputs

    in_info = session.get_input_info(0)
    assert in_info == session.get_input_info("x")
    assert in_info.name == "x"
    assert in_info.dtype == DataType.Float32
    assert in_info.shape == (-1, 4)
    assert in_info.symbolic_dims == ("batch_size", "")
    assert in_info.rank == 2
    assert in_info.has_dynamic_batch
    assert in_info.is_dynamic_dim(0)
    assert not in_info.is_dynamic_dim(1)
    assert in_info.fixed_element_count() == 4

    out_info = session.get_output_info("y")
    assert out_info.name == "y"
    assert out_info.shape == (-1, 3)

    assert info.get_input_index("x") == 0
    assert info.get_output_index("y") == 0
    assert info.find_input("unknown") is None

    # Validate inputs directly via session and ModelInfo
    valid_arr = np.ones((4, 4), dtype=np.float32)
    session.validate_inputs(valid_arr)
    info.validate_inputs({"x": valid_arr})

    bad_arr = np.ones((4, 7), dtype=np.float32)
    with pytest.raises(ValueError):
        session.validate_inputs(bad_arr)


# ---------------------------------------------------------------------------
# 3. Synchronous & Asynchronous InferenceSession Execution
# ---------------------------------------------------------------------------


def test_synchronous_inference_numpy_arena_and_in_place():
    session = InferenceSession(build_mlp_onnx_bytes())
    x_np = np.array(
        [
            [1.0, 2.0, -1.0, 0.5],
            [-0.5, 0.0, 3.0, 1.5],
            [2.0, -1.0, 0.5, -2.0],
        ],
        dtype=np.float32,
    )
    expected = expected_mlp_numpy(x_np)

    # 1. Direct NumPy input -> list[Tensor] output
    outputs = session.run(x_np)
    assert len(outputs) == 1
    y_np = outputs[0].numpy()
    assert y_np.shape == (3, 3)
    np.testing.assert_allclose(y_np, expected, rtol=1e-5, atol=1e-5)

    # 2. run_numpy convenience method -> list[np.ndarray]
    out_arrays = session.run_numpy({"x": x_np})
    assert len(out_arrays) == 1
    np.testing.assert_allclose(out_arrays[0], expected, rtol=1e-5, atol=1e-5)

    # 3. Pre-allocated output in ArenaAllocator
    arena = ArenaAllocator(64 * 1024)
    arena_outs = session.run(x_np, arena=arena)
    assert arena_outs[0].storage_type == StorageType.ARENA
    assert arena.is_pinned
    np.testing.assert_allclose(arena_outs[0].numpy(), expected, rtol=1e-5, atol=1e-5)
    del arena_outs

    # 4. In-place pre-allocated output Tensor
    out_tensor = Tensor.zeros((3, 3), DataType.Float32)
    out_ptr = out_tensor.data_ptr
    returned = session.run(x_np, outputs=[out_tensor])
    assert returned[0].data_ptr == out_ptr
    np.testing.assert_allclose(out_tensor.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_asynchronous_inference_with_thread_pool_and_metrics():
    session = InferenceSession(build_arithmetic_onnx_bytes())
    budget = ThreadPool.compute_thread_budget(reserved_for_ort=1)
    assert budget.pyreth_worker_threads >= 1

    pool_cfg = ThreadPoolConfig(
        num_threads=2, queue_capacity=256, reserve_threads_for_ort=1
    )
    mem_pool = MemoryPool(initial_count=8, block_size=64 * 1024)

    with ThreadPool(pool_cfg) as pool:
        assert pool.num_workers == 2
        assert not pool.is_shutdown

        futures = []
        expected_pairs = []
        for i in range(16):
            a = np.full((2, 4), float(i + 1), dtype=np.float32)
            b = np.full((2, 4), float((i + 1) * 0.5), dtype=np.float32)
            expected_pairs.append((a + b, a * b))
            fut = session.run_async(pool, [a, b], output_pool=mem_pool)
            assert fut.valid
            futures.append(fut)

        for idx, fut in enumerate(futures):
            assert fut.wait_for(5000.0)
            out_arrays = fut.get_numpy()
            assert len(out_arrays) == 2
            exp_sum, exp_prod = expected_pairs[idx]
            np.testing.assert_allclose(out_arrays[0], exp_sum, rtol=1e-5)
            np.testing.assert_allclose(out_arrays[1], exp_prod, rtol=1e-5)

        pool_metrics = pool.metrics()
        assert pool_metrics.tasks_submitted == 16
        assert pool_metrics.tasks_executed == 16
        assert pool_metrics.to_dict()["tasks_executed"] == 16

    assert pool.is_shutdown


# ---------------------------------------------------------------------------
# 4. DynamicBatcher & Multi-Threaded GIL Release Scaling Tests
# ---------------------------------------------------------------------------


def test_dynamic_batcher_async_submit_and_metrics():
    session = InferenceSession(build_mlp_onnx_bytes())
    cfg = DynamicBatcherConfig(
        max_batch_size=8,
        max_batch_delay_ms=10,
        max_queue_capacity=256,
        num_workers=2,
    )

    with DynamicBatcher(session, cfg) as batcher:
        assert batcher.is_valid
        assert not batcher.is_shutdown
        assert batcher.config.max_batch_size == 8

        inputs = []
        futures = []
        for i in range(8):
            x = np.array([[float(i), -float(i) * 0.5, 1.0, 0.25]], dtype=np.float32)
            inputs.append(x)
            futures.append(batcher.submit(x))

        for i, fut in enumerate(futures):
            outs = fut.get_numpy()
            assert len(outs) == 1
            assert outs[0].shape == (1, 3)
            np.testing.assert_allclose(
                outs[0], expected_mlp_numpy(inputs[i]), rtol=1e-5, atol=1e-5
            )

        batcher.flush()
        m = batcher.metrics()
        assert m.requests_submitted == 8
        assert m.requests_completed == 8
        assert m.requests_rejected == 0
        assert m.total_batched_samples == 8
        assert m.batches_dispatched >= 1
        assert m.average_batch_size >= 1.0
        assert m.to_dict()["requests_completed"] == 8

        batcher.reset_metrics()
        assert batcher.metrics().requests_submitted == 0

    assert batcher.is_shutdown


def test_dynamic_batcher_multithreaded_gil_release_coalesces_concurrent_threads():
    """Verify that Python threads calling batcher.run() release the GIL and coalesce.

    We configure max_batch_size=8 and a long max_batch_delay_ms=500ms, then spawn
    8 Python threads that simultaneously call the synchronous `batcher.run(x)`.
    Because `batcher.run` releases the Python GIL while waiting for the batch to form,
    all 8 threads enter the C++ queue concurrently, trigger an immediate full batch of 8,
    and return in << 500ms.
    """
    session = InferenceSession(build_mlp_onnx_bytes())
    cfg = DynamicBatcherConfig(max_batch_size=8, max_batch_delay_ms=500)

    num_threads = 8
    barrier = threading.Barrier(num_threads)

    with DynamicBatcher(session, cfg) as batcher:

        def worker_fn(tid: int) -> np.ndarray:
            x = np.array([[float(tid + 1), 0.5, -0.25, 1.0]], dtype=np.float32)
            barrier.wait()
            out_arrays = batcher.run_numpy(x)
            np.testing.assert_allclose(
                out_arrays[0], expected_mlp_numpy(x), rtol=1e-5, atol=1e-5
            )
            return out_arrays[0]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(worker_fn, range(num_threads)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(results) == num_threads
        # Full batch coalescing must complete well before the 500ms timeout window
        assert elapsed_ms < 300.0

        m = batcher.metrics()
        assert m.requests_completed == num_threads
        assert m.total_batched_samples == num_threads
        assert m.full_batches == 1
        assert m.batches_dispatched == 1
        assert m.average_batch_size == pytest.approx(8.0)


def test_concurrent_python_threads_scaling_across_session_and_batcher():
    session = InferenceSession(build_arithmetic_onnx_bytes())
    pool = ThreadPool(4)
    mem_pool = MemoryPool(16, 128 * 1024)
    cfg = DynamicBatcherConfig(max_batch_size=16, max_batch_delay_ms=5)

    with DynamicBatcher(session, cfg, pool=pool, memory_pool=mem_pool) as batcher:

        def client_task(req_id: int) -> None:
            a = np.full((2, 4), float(req_id + 1), dtype=np.float32)
            b = np.full((2, 4), 3.0, dtype=np.float32)
            outs = batcher.run_numpy({"a": a, "b": b})
            np.testing.assert_allclose(outs[0], a + b, rtol=1e-5)
            np.testing.assert_allclose(outs[1], a * b, rtol=1e-5)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(client_task, range(64)))

        m = batcher.metrics()
        assert m.requests_submitted == 64
        assert m.requests_completed == 64
        assert m.total_batched_samples == 128
        assert m.batches_dispatched < 64
        assert m.average_batch_size > 2.0

    pool.shutdown()
