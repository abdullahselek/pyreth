"""End-to-end throughput and latency serving benchmark suite for Pyreth.

Measures P50, P90, and P99 latency (in microseconds/milliseconds) as well as
request throughput (requests/sec) and sample throughput (samples/sec) across
batch sizes 1, 2, 4, 8, 16, 32, and 64 for:
  1. Synchronous InferenceSession execution (Heap vs ArenaAllocator reuse)
  2. Asynchronous ThreadPool + MemoryPool execution (InferenceSession.run_async)
  3. Concurrent multi-threaded serving with DynamicBatcher (GIL-free scaling)

Usage:
    uv run python benchmarks/benchmark_serving.py
"""

import argparse
import json
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyreth import (
    ArenaAllocator,
    Backend,
    DynamicBatcher,
    DynamicBatcherConfig,
    ExecutionProvider,
    InferenceSession,
    MemoryPool,
    OptimizationLevel,
    SessionConfig,
    SessionExecutionMode,
    Tensor,
    ThreadPool,
    ThreadPoolConfig,
)

BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Minimal Pure-Python ONNX Protobuf Builder (2-Layer MLP: X @ W1 + B1 -> ReLU -> @ W2 + B2)
# ---------------------------------------------------------------------------


def _encode_varint(val: int) -> bytes:
    if val < 0:
        val += 1 << 64
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val)
    return bytes(out)


def _encode_tag(field_num: int, wire_type: int) -> bytes:
    return _encode_varint((field_num << 3) | wire_type)


def _encode_int64(field_num: int, val: int) -> bytes:
    return _encode_tag(field_num, 0) + _encode_varint(val)


def _encode_string(field_num: int, text: str) -> bytes:
    raw = text.encode("utf-8")
    return _encode_tag(field_num, 2) + _encode_varint(len(raw)) + raw


def _encode_bytes(field_num: int, payload: bytes) -> bytes:
    return _encode_tag(field_num, 2) + _encode_varint(len(payload)) + payload


def _encode_shape_dim(dim_val: int = -1, dim_param: str = "") -> bytes:
    if dim_param:
        return _encode_string(2, dim_param)
    return _encode_int64(1, dim_val)


def _encode_value_info(name: str, elem_type: int, dims: list[bytes]) -> bytes:
    shape_proto = b"".join(_encode_bytes(1, d) for d in dims)
    tensor_type = _encode_int64(1, elem_type) + _encode_bytes(2, shape_proto)
    type_proto = _encode_bytes(1, tensor_type)
    return _encode_string(1, name) + _encode_bytes(2, type_proto)


def _encode_float_initializer(name: str, dims: list[int], values: np.ndarray) -> bytes:
    tp = bytearray()
    for d in dims:
        tp += _encode_int64(1, d)
    tp += _encode_int64(2, 1)  # FLOAT = 1
    tp += _encode_string(8, name)
    raw = np.asarray(values, dtype=np.float32).tobytes()
    tp += _encode_bytes(9, raw)
    return bytes(tp)


def _encode_node(
    op_type: str, inputs: list[str], outputs: list[str], name: str
) -> bytes:
    node = bytearray()
    for inp in inputs:
        node += _encode_string(1, inp)
    for out in outputs:
        node += _encode_string(2, out)
    node += _encode_string(3, name)
    node += _encode_string(4, op_type)
    return bytes(node)


def build_serving_mlp_onnx_bytes(
    in_features: int = 64, hidden_features: int = 128, out_features: int = 16
) -> bytes:
    """Build a 2-layer MLP ONNX model with dynamic batch axis:
    Y = Relu(X @ W1 + B1) @ W2 + B2
    """
    rng = np.random.default_rng(42)
    w1 = rng.standard_normal((in_features, hidden_features), dtype=np.float32) * 0.1
    b1 = np.zeros((hidden_features,), dtype=np.float32)
    w2 = rng.standard_normal((hidden_features, out_features), dtype=np.float32) * 0.1
    b2 = np.zeros((out_features,), dtype=np.float32)

    graph = bytearray()
    graph += _encode_string(2, "serving_mlp_graph")
    graph += _encode_bytes(
        5, _encode_float_initializer("W1", [in_features, hidden_features], w1)
    )
    graph += _encode_bytes(5, _encode_float_initializer("B1", [hidden_features], b1))
    graph += _encode_bytes(
        5, _encode_float_initializer("W2", [hidden_features, out_features], w2)
    )
    graph += _encode_bytes(5, _encode_float_initializer("B2", [out_features], b2))

    graph += _encode_bytes(1, _encode_node("MatMul", ["x", "W1"], ["h_mm"], "mm1"))
    graph += _encode_bytes(1, _encode_node("Add", ["h_mm", "B1"], ["h_add"], "add1"))
    graph += _encode_bytes(1, _encode_node("Relu", ["h_add"], ["h_relu"], "relu1"))
    graph += _encode_bytes(1, _encode_node("MatMul", ["h_relu", "W2"], ["y_mm"], "mm2"))
    graph += _encode_bytes(1, _encode_node("Add", ["y_mm", "B2"], ["y"], "add2"))

    graph += _encode_bytes(
        11,
        _encode_value_info(
            "x",
            1,
            [
                _encode_shape_dim(dim_param="batch_size"),
                _encode_shape_dim(dim_val=in_features),
            ],
        ),
    )
    graph += _encode_bytes(
        12,
        _encode_value_info(
            "y",
            1,
            [
                _encode_shape_dim(dim_param="batch_size"),
                _encode_shape_dim(dim_val=out_features),
            ],
        ),
    )

    opset = _encode_string(1, "") + _encode_int64(2, 13)
    model = bytearray()
    model += _encode_int64(1, 7)
    model += _encode_string(2, "pyreth_serving_benchmark")
    model += _encode_int64(5, 1)
    model += _encode_bytes(7, bytes(graph))
    model += _encode_bytes(8, opset)
    return bytes(model)


# ---------------------------------------------------------------------------
# Benchmark Data Structures
# ---------------------------------------------------------------------------


@dataclass
class BatchBenchmarkEntry:
    mode: str
    batch_size: int
    p50_us: float
    p90_us: float
    p99_us: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    requests_per_sec: float
    samples_per_sec: float
    avg_coalesced_batch: float = 0.0


def _compute_percentiles_us(
    latencies_us: list[float] | np.ndarray,
) -> tuple[float, float, float]:
    arr = np.asarray(latencies_us, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    return p50, p90, p99


# ---------------------------------------------------------------------------
# 1. Synchronous InferenceSession Benchmarks (Heap vs ArenaAllocator)
# ---------------------------------------------------------------------------


def benchmark_sync_session(
    session: InferenceSession,
    in_features: int,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = 30,
    iterations: int = 200,
    use_arena: bool = False,
) -> list[BatchBenchmarkEntry]:
    results: list[BatchBenchmarkEntry] = []
    arena = ArenaAllocator(512 * 1024) if use_arena else None
    mode_name = "sync_arena" if use_arena else "sync_heap"

    for b in batch_sizes:
        x_np = np.ones((b, in_features), dtype=np.float32)
        x_tensor = Tensor.from_numpy(x_np)

        for _ in range(warmup):
            if arena is not None:
                outs = session.run(x_tensor, arena=arena)
                del outs
                arena.reset()
            else:
                outs = session.run(x_tensor)
                del outs

        latencies_us: list[float] = []
        t_start_wall = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            if arena is not None:
                outs = session.run(x_tensor, arena=arena)
                del outs
                arena.reset()
            else:
                outs = session.run(x_tensor)
                del outs
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)
        total_wall_s = time.perf_counter() - t_start_wall

        p50_us, p90_us, p99_us = _compute_percentiles_us(latencies_us)
        req_per_sec = iterations / max(total_wall_s, 1e-9)
        samples_per_sec = (iterations * b) / max(total_wall_s, 1e-9)

        results.append(
            BatchBenchmarkEntry(
                mode=mode_name,
                batch_size=b,
                p50_us=round(p50_us, 2),
                p90_us=round(p90_us, 2),
                p99_us=round(p99_us, 2),
                p50_ms=round(p50_us / 1000.0, 4),
                p90_ms=round(p90_us / 1000.0, 4),
                p99_ms=round(p99_us / 1000.0, 4),
                requests_per_sec=round(req_per_sec, 1),
                samples_per_sec=round(samples_per_sec, 1),
                avg_coalesced_batch=float(b),
            )
        )

    return results


# ---------------------------------------------------------------------------
# 2. Asynchronous ThreadPool + MemoryPool Benchmarks
# ---------------------------------------------------------------------------


def benchmark_async_thread_pool(
    session: InferenceSession,
    in_features: int,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = 20,
    iterations: int = 200,
    num_workers: int = 4,
) -> list[BatchBenchmarkEntry]:
    results: list[BatchBenchmarkEntry] = []
    mem_pool = MemoryPool(initial_count=16, block_size=256 * 1024)
    pool_cfg = ThreadPoolConfig(
        num_threads=num_workers, queue_capacity=1024, reserve_threads_for_ort=1
    )

    with ThreadPool(pool_cfg) as pool:
        for b in batch_sizes:
            x_np = np.ones((b, in_features), dtype=np.float32)
            x_tensor = Tensor.from_numpy(x_np)

            # Warmup
            warmup_futs = [
                session.run_async(pool, x_tensor, output_pool=mem_pool)
                for _ in range(warmup)
            ]
            for f in warmup_futs:
                _ = f.get()
            del warmup_futs

            latencies_us: list[float] = [0.0] * iterations
            submit_times: list[float] = [0.0] * iterations
            futures = []

            t_start_wall = time.perf_counter()
            for i in range(iterations):
                submit_times[i] = time.perf_counter()
                futures.append(session.run_async(pool, x_tensor, output_pool=mem_pool))

            for i, f in enumerate(futures):
                outs = f.get()
                done_t = time.perf_counter()
                latencies_us[i] = (done_t - submit_times[i]) * 1e6
                del outs
            total_wall_s = time.perf_counter() - t_start_wall

            p50_us, p90_us, p99_us = _compute_percentiles_us(latencies_us)
            req_per_sec = iterations / max(total_wall_s, 1e-9)
            samples_per_sec = (iterations * b) / max(total_wall_s, 1e-9)

            results.append(
                BatchBenchmarkEntry(
                    mode="async_thread_pool",
                    batch_size=b,
                    p50_us=round(p50_us, 2),
                    p90_us=round(p90_us, 2),
                    p99_us=round(p99_us, 2),
                    p50_ms=round(p50_us / 1000.0, 4),
                    p90_ms=round(p90_us / 1000.0, 4),
                    p99_ms=round(p99_us / 1000.0, 4),
                    requests_per_sec=round(req_per_sec, 1),
                    samples_per_sec=round(samples_per_sec, 1),
                    avg_coalesced_batch=float(b),
                )
            )

    return results


# ---------------------------------------------------------------------------
# 3. Concurrent DynamicBatcher Serving Benchmarks (Batch Sizes 1 to 64)
# ---------------------------------------------------------------------------


def benchmark_dynamic_batcher_serving(
    session: InferenceSession,
    in_features: int,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    total_requests: int = 512,
    client_threads: int = 8,
) -> list[BatchBenchmarkEntry]:
    results: list[BatchBenchmarkEntry] = []
    single_sample = np.ones((1, in_features), dtype=np.float32)
    single_tensor = Tensor.from_numpy(single_sample)
    requests_per_thread = max(1, total_requests // client_threads)
    effective_total = requests_per_thread * client_threads
    pipeline_depth = 8

    for max_b in batch_sizes:
        cfg = DynamicBatcherConfig(
            max_batch_size=max_b,
            max_batch_delay_ms=1,
            max_batch_delay_us=500,
            max_queue_capacity=1024,
            num_workers=4,
        )

        with DynamicBatcher(session, cfg) as batcher:
            # Warmup
            for _ in range(16):
                _ = batcher.run(single_tensor)
            batcher.reset_metrics()

            latencies_us: list[float] = [0.0] * effective_total

            def _client_worker(
                tid: int,
                batcher_ref: DynamicBatcher = batcher,
                lat_out: list[float] = latencies_us,
            ) -> None:
                base_idx = tid * requests_per_thread
                offset = 0
                while offset < requests_per_thread:
                    window = min(pipeline_depth, requests_per_thread - offset)
                    sub_times = [0.0] * window
                    futs = []
                    for w in range(window):
                        sub_times[w] = time.perf_counter()
                        futs.append(batcher_ref.submit(single_tensor))
                    for w, fut in enumerate(futs):
                        outs = fut.get()
                        done_t = time.perf_counter()
                        lat_out[base_idx + offset + w] = (done_t - sub_times[w]) * 1e6
                        del outs
                    offset += window

            t_start_wall = time.perf_counter()
            with ThreadPoolExecutor(max_workers=client_threads) as executor:
                list(executor.map(_client_worker, range(client_threads)))
            total_wall_s = time.perf_counter() - t_start_wall

            m = batcher.metrics()
            p50_us, p90_us, p99_us = _compute_percentiles_us(latencies_us)
            req_per_sec = effective_total / max(total_wall_s, 1e-9)

            results.append(
                BatchBenchmarkEntry(
                    mode="dynamic_batcher",
                    batch_size=max_b,
                    p50_us=round(p50_us, 2),
                    p90_us=round(p90_us, 2),
                    p99_us=round(p99_us, 2),
                    p50_ms=round(p50_us / 1000.0, 4),
                    p90_ms=round(p90_us / 1000.0, 4),
                    p99_ms=round(p99_us / 1000.0, 4),
                    requests_per_sec=round(req_per_sec, 1),
                    samples_per_sec=round(req_per_sec, 1),
                    avg_coalesced_batch=round(m.average_batch_size, 2),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Automated Regression Checks
# ---------------------------------------------------------------------------


def verify_benchmark_regressions(
    sync_heap: list[BatchBenchmarkEntry],
    sync_arena: list[BatchBenchmarkEntry],
    async_pool: list[BatchBenchmarkEntry],
    dynamic_batcher: list[BatchBenchmarkEntry],
) -> list[str]:
    """Validate latency and throughput scaling invariants across batch sizes 1..64."""
    checks_passed: list[str] = []

    # 1. All batch sizes 1..64 present
    for entries, label in (
        (sync_heap, "sync_heap"),
        (sync_arena, "sync_arena"),
        (async_pool, "async_thread_pool"),
        (dynamic_batcher, "dynamic_batcher"),
    ):
        observed_batches = tuple(e.batch_size for e in entries)
        if observed_batches != BATCH_SIZES:
            raise AssertionError(
                f"{label}: expected batch sizes {BATCH_SIZES}, got {observed_batches}"
            )
        for e in entries:
            if not (0.0 < e.p50_us <= e.p90_us <= e.p99_us):
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): invalid latency percentile ordering "
                    f"P50={e.p50_us}us, P90={e.p90_us}us, P99={e.p99_us}us"
                )
            if e.p99_ms > 100.0:
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): P99 latency {e.p99_ms:.2f}ms exceeds 100ms ceiling"
                )
            if e.requests_per_sec <= 0 or e.samples_per_sec <= 0:
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): non-positive throughput"
                )
    checks_passed.append(
        "All batch sizes (1, 2, 4, 8, 16, 32, 64) completed within P99 < 100ms SLA"
    )

    # 2. Sample throughput scales significantly from batch=1 to batch=64
    heap_b1 = sync_heap[0].samples_per_sec
    heap_b64 = sync_heap[-1].samples_per_sec
    arena_b1 = sync_arena[0].samples_per_sec
    arena_b64 = sync_arena[-1].samples_per_sec

    if heap_b64 < heap_b1 * 5.0:
        raise AssertionError(
            f"sync_heap sample throughput failed to scale: B1={heap_b1:.1f} vs B64={heap_b64:.1f}"
        )
    if arena_b64 < arena_b1 * 5.0:
        raise AssertionError(
            f"sync_arena sample throughput failed to scale: B1={arena_b1:.1f} vs B64={arena_b64:.1f}"
        )
    checks_passed.append(
        f"Synchronous sample throughput scaled {arena_b64 / arena_b1:.1f}x from Batch=1 to Batch=64"
    )

    # 3. DynamicBatcher coalesces concurrent requests when max_batch_size > 1
    db_b64 = dynamic_batcher[-1]
    if db_b64.avg_coalesced_batch <= 1.0:
        raise AssertionError(
            f"DynamicBatcher failed to coalesce concurrent requests (avg_batch={db_b64.avg_coalesced_batch})"
        )
    checks_passed.append(
        f"DynamicBatcher coalesced concurrent Python threads (avg batch size = {db_b64.avg_coalesced_batch:.2f} at max_batch=64)"
    )

    return checks_passed


# ---------------------------------------------------------------------------
# Formatting & CLI Runner
# ---------------------------------------------------------------------------


def _print_table(title: str, entries: list[BatchBenchmarkEntry]) -> None:
    print(f"\n=== {title} ===")
    header = (
        f"{'Batch':>6} | {'P50 (us)':>10} | {'P90 (us)':>10} | {'P99 (us)':>10} | "
        f"{'P99 (ms)':>9} | {'Req/sec':>12} | {'Samples/sec':>14} | {'AvgBatch':>8}"
    )
    print(header)
    print("-" * len(header))
    for e in entries:
        print(
            f"{e.batch_size:>6d} | {e.p50_us:>10.2f} | {e.p90_us:>10.2f} | {e.p99_us:>10.2f} | "
            f"{e.p99_ms:>9.4f} | {e.requests_per_sec:>12.1f} | {e.samples_per_sec:>14.1f} | "
            f"{e.avg_coalesced_batch:>8.2f}"
        )


def run_all_benchmarks(
    iterations: int = 200,
    warmup: int = 30,
    dynamic_requests: int = 256,
    client_threads: int = 8,
    check_regressions: bool = True,
) -> dict[str, object]:
    in_features = 64
    model_bytes = build_serving_mlp_onnx_bytes(
        in_features=in_features, hidden_features=128, out_features=16
    )

    cfg = SessionConfig(
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        graph_optimization_level=OptimizationLevel.All,
        execution_mode=SessionExecutionMode.Sequential,
        providers=[ExecutionProvider.CPU],
    )
    session = InferenceSession(model_bytes, cfg)

    sync_heap = benchmark_sync_session(
        session,
        in_features=in_features,
        warmup=warmup,
        iterations=iterations,
        use_arena=False,
    )
    sync_arena = benchmark_sync_session(
        session,
        in_features=in_features,
        warmup=warmup,
        iterations=iterations,
        use_arena=True,
    )
    async_pool = benchmark_async_thread_pool(
        session,
        in_features=in_features,
        warmup=warmup,
        iterations=iterations,
        num_workers=4,
    )
    dynamic_batcher = benchmark_dynamic_batcher_serving(
        session,
        in_features=in_features,
        total_requests=dynamic_requests,
        client_threads=client_threads,
    )

    checks: list[str] = []
    if check_regressions:
        checks = verify_benchmark_regressions(
            sync_heap, sync_arena, async_pool, dynamic_batcher
        )

    return {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "onnxruntime_version": Backend.get_runtime_version(),
            "best_provider": str(Backend.get_best_available_provider()),
        },
        "sync_heap": [asdict(e) for e in sync_heap],
        "sync_arena": [asdict(e) for e in sync_arena],
        "async_thread_pool": [asdict(e) for e in async_pool],
        "dynamic_batcher": [asdict(e) for e in dynamic_batcher],
        "regression_checks": checks,
        "_objects": (sync_heap, sync_arena, async_pool, dynamic_batcher),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pyreth End-to-End Throughput & Latency Serving Benchmark Suite"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Iterations per batch size for sync/async benchmarks (default: 200)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Warmup iterations per batch size (default: 30)",
    )
    parser.add_argument(
        "--dynamic-requests",
        type=int,
        default=256,
        help="Total concurrent requests per batch size for DynamicBatcher (default: 256)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Concurrent Python client threads for DynamicBatcher (default: 8)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report",
    )
    parser.add_argument(
        "--skip-regression-check",
        action="store_true",
        help="Skip automated regression assertions",
    )
    args = parser.parse_args()

    report = run_all_benchmarks(
        iterations=args.iterations,
        warmup=args.warmup,
        dynamic_requests=args.dynamic_requests,
        client_threads=args.threads,
        check_regressions=not args.skip_regression_check,
    )

    sync_heap, sync_arena, async_pool, dynamic_batcher = report.pop("_objects")  # type: ignore[misc]
    env = report["environment"]  # type: ignore[index]

    print(
        "================================================================================"
    )
    print("Pyreth End-to-End Serving Benchmark Suite")
    print(f"Platform     : {env['platform']} ({env['processor']})")
    print(f"Python       : {env['python_version']}")
    print(
        f"ONNX Runtime : {env['onnxruntime_version']} (Best Provider: {env['best_provider']})"
    )
    print("Model        : 2-Layer MLP (64 -> 128 -> ReLU -> 16, Float32)")
    print("Batch Sizes  : 1, 2, 4, 8, 16, 32, 64")
    print(
        "================================================================================"
    )

    _print_table(
        "1A. Synchronous InferenceSession.run (Heap-Allocated Outputs)", sync_heap
    )
    _print_table(
        "1B. Synchronous InferenceSession.run (ArenaAllocator Reuse)", sync_arena
    )
    _print_table(
        "2. Asynchronous InferenceSession.run_async (ThreadPool + MemoryPool)",
        async_pool,
    )
    _print_table(
        f"3. Concurrent DynamicBatcher Serving ({args.threads} Python Threads)",
        dynamic_batcher,
    )

    if report["regression_checks"]:
        print("\n=== Automated Regression Checks ===")
        for chk in report["regression_checks"]:  # type: ignore[attr-defined]
            print(f"  [PASS] {chk}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON benchmark report to {args.json_out}")


if __name__ == "__main__":
    main()
