"""Shared benchmarking harness for Pyreth model benchmarks.

Provides reusable benchmark runners across batch sizes (1, 2, 4, 8, 16, 32, 64) for:
  1A. Synchronous InferenceSession.run (Heap-Allocated Outputs)
  1B. Synchronous InferenceSession.run (ArenaAllocator O(1) Reset Reuse)
  2.  Asynchronous InferenceSession.run_async (ThreadPool + MemoryPool)
  3.  Concurrent DynamicBatcher Serving (Multi-Threaded Python Clients with GIL Release)
"""

import json
import platform
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyreth import (
    ArenaAllocator,
    Backend,
    DataType,
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
DEFAULT_CACHE_DIR: Path = Path(__file__).resolve().parent / ".onnx_cache"

ModelInputs = Tensor | list[Tensor] | dict[str, Tensor]
InputBuilder = Callable[[int], ModelInputs]


@dataclass
class BatchBenchmarkEntry:
    """Single benchmark measurement entry for a specific execution mode and batch size.

    Attributes:
        mode: Execution mode identifier (`sync_heap`, `sync_arena`, `async_thread_pool`,
            or `dynamic_batcher`).
        batch_size: Input batch size (or `max_batch_size` for `dynamic_batcher`).
        p50_us: Median (P50) request latency in microseconds.
        p90_us: 90th percentile (P90) request latency in microseconds.
        p99_us: 99th percentile (P99) request latency in microseconds.
        p50_ms: Median (P50) request latency in milliseconds.
        p90_ms: 90th percentile (P90) request latency in milliseconds.
        p99_ms: 99th percentile (P99) request latency in milliseconds.
        requests_per_sec: Measured request throughput in requests per second.
        samples_per_sec: Measured sample throughput in samples per second.
        avg_coalesced_batch: Average coalesced batch size per dispatched ONNX Runtime run.
    """

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


def compute_percentiles_us(
    latencies_us: list[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Compute P50, P90, and P99 latency percentiles in microseconds.

    Args:
        latencies_us: Sequence or array of per-request latencies in microseconds.

    Returns:
        A tuple of `(p50_us, p90_us, p99_us)` as floats.
    """
    arr = np.asarray(latencies_us, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    return p50, p90, p99


def _dtype_byte_size(dtype: DataType) -> int:
    """Return the element byte size for a given Pyreth `DataType`.

    Args:
        dtype: The Pyreth tensor `DataType` enum value.

    Returns:
        Number of bytes per element (1, 2, 4, or 8).
    """
    if dtype in (DataType.Float64, DataType.Int64, DataType.UInt64):
        return 8
    if dtype in (DataType.Float32, DataType.Int32, DataType.UInt32):
        return 4
    if dtype in (
        DataType.Float16,
        DataType.BFloat16,
        DataType.Int16,
        DataType.UInt16,
    ):
        return 2
    return 1


def _inputs_total_bytes(inputs: ModelInputs) -> int:
    """Compute the total byte footprint of a model input container including alignment padding.

    Args:
        inputs: Single `Tensor`, positional `list[Tensor]`, or named `dict[str, Tensor]`.

    Returns:
        Total payload bytes plus 64-byte alignment padding per tensor.
    """
    if isinstance(inputs, Tensor):
        return inputs.nbytes + 64
    if isinstance(inputs, dict):
        return sum(t.nbytes + 64 for t in inputs.values())
    return sum(t.nbytes + 64 for t in inputs)


def compute_recommended_arena_block_size(
    session: InferenceSession,
    input_builder: InputBuilder | None = None,
    max_batch_size: int = 64,
    min_block_size: int = 1024 * 1024,
) -> int:
    """Compute a 64 KiB-aligned arena block size capable of holding max-batch inputs and outputs.

    Args:
        session: Initialized `InferenceSession` whose output schemas are inspected.
        input_builder: Optional callable returning sample inputs for a given batch size.
        max_batch_size: Maximum batch size that will be allocated in a single arena block.
        min_block_size: Minimum block size in bytes (defaults to 1 MiB).

    Returns:
        Recommended arena block size in bytes (multiple of 64 KiB).
    """
    inferred_seq_dim = 64
    input_bytes = 0
    if input_builder is not None:
        sample_inputs = input_builder(max_batch_size)
        input_bytes = _inputs_total_bytes(sample_inputs)
        first_tensor = (
            sample_inputs
            if isinstance(sample_inputs, Tensor)
            else (
                next(iter(sample_inputs.values()))
                if isinstance(sample_inputs, dict)
                else sample_inputs[0]
            )
        )
        if first_tensor.rank >= 2:
            inferred_seq_dim = max(inferred_seq_dim, int(first_tensor.shape[1]))

    output_bytes = 0
    for out_info in session.outputs:
        elems = 1
        for axis, dim in enumerate(out_info.shape):
            if axis == 0:
                elems *= max_batch_size
            elif dim > 0:
                elems *= int(dim)
            else:
                elems *= inferred_seq_dim
        output_bytes += elems * _dtype_byte_size(out_info.dtype) + 64

    required = max(input_bytes, output_bytes) * 2
    aligned = ((max(min_block_size, required) + 65535) // 65536) * 65536
    return int(aligned)


def create_default_session(
    model_path_or_bytes: Path | str | bytes,
    intra_op_threads: int = 1,
    providers: list[ExecutionProvider] | None = None,
) -> InferenceSession:
    """Create an `InferenceSession` configured for low-overhead benchmark execution.

    Args:
        model_path_or_bytes: Path to an ONNX file on disk or serialized ONNX model bytes.
        intra_op_threads: Number of ONNX Runtime intra-op threads per session run.
        providers: Optional ordered list of `ExecutionProvider`s (defaults to `[CPU]`).

    Returns:
        Configured and initialized `InferenceSession`.
    """
    cfg = SessionConfig(
        intra_op_num_threads=intra_op_threads,
        inter_op_num_threads=1,
        graph_optimization_level=OptimizationLevel.All,
        execution_mode=SessionExecutionMode.Sequential,
        providers=providers if providers is not None else [ExecutionProvider.CPU],
    )
    return InferenceSession(model_path_or_bytes, cfg)


def benchmark_sync_session(
    session: InferenceSession,
    input_builder: InputBuilder,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = 10,
    iterations: int = 50,
    use_arena: bool = False,
    arena_block_size: int = 0,
) -> list[BatchBenchmarkEntry]:
    """Benchmark synchronous `InferenceSession.run` across the specified batch sizes.

    Args:
        session: Active `InferenceSession` to benchmark.
        input_builder: Callable taking `batch_size` and returning model input tensor(s).
        batch_sizes: Tuple of batch sizes to evaluate.
        warmup: Number of untimed warmup iterations per batch size.
        iterations: Number of timed iterations per batch size.
        use_arena: If True, pre-allocate outputs in a reusable `ArenaAllocator` and call
            `arena.reset()` after each iteration; otherwise allocate outputs on the heap.
        arena_block_size: Optional custom arena block size in bytes (0 to auto-compute).

    Returns:
        List of `BatchBenchmarkEntry` results, one per batch size.
    """
    results: list[BatchBenchmarkEntry] = []
    if arena_block_size <= 0:
        arena_block_size = compute_recommended_arena_block_size(
            session, input_builder=input_builder, max_batch_size=max(batch_sizes)
        )
    arena = ArenaAllocator(arena_block_size) if use_arena else None
    mode_name = "sync_arena" if use_arena else "sync_heap"

    for b in batch_sizes:
        inputs = input_builder(b)

        for _ in range(warmup):
            if arena is not None:
                outs = session.run(inputs, arena=arena)
                del outs
                arena.reset()
            else:
                outs = session.run(inputs)
                del outs

        latencies_us: list[float] = []
        t_start_wall = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            if arena is not None:
                outs = session.run(inputs, arena=arena)
                del outs
                arena.reset()
            else:
                outs = session.run(inputs)
                del outs
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1e6)
        total_wall_s = time.perf_counter() - t_start_wall

        p50_us, p90_us, p99_us = compute_percentiles_us(latencies_us)
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


def benchmark_async_thread_pool(
    session: InferenceSession,
    input_builder: InputBuilder,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    warmup: int = 10,
    iterations: int = 50,
    num_workers: int = 4,
    arena_block_size: int = 0,
) -> list[BatchBenchmarkEntry]:
    """Benchmark asynchronous `InferenceSession.run_async` on a `ThreadPool` and `MemoryPool`.

    Args:
        session: Active `InferenceSession` to benchmark.
        input_builder: Callable taking `batch_size` and returning model input tensor(s).
        batch_sizes: Tuple of batch sizes to evaluate.
        warmup: Number of untimed warmup submissions per batch size.
        iterations: Number of timed asynchronous submissions per batch size.
        num_workers: Number of worker threads in the C++ `ThreadPool`.
        arena_block_size: Optional custom arena block size in bytes for `MemoryPool`.

    Returns:
        List of `BatchBenchmarkEntry` results, one per batch size.
    """
    results: list[BatchBenchmarkEntry] = []
    if arena_block_size <= 0:
        arena_block_size = compute_recommended_arena_block_size(
            session, input_builder=input_builder, max_batch_size=max(batch_sizes)
        )
    mem_pool = MemoryPool(
        initial_count=max(8, num_workers * 2), block_size=arena_block_size
    )
    pool_cfg = ThreadPoolConfig(
        num_threads=num_workers, queue_capacity=1024, reserve_threads_for_ort=1
    )

    with ThreadPool(pool_cfg) as pool:
        for b in batch_sizes:
            inputs = input_builder(b)

            warmup_futs = [
                session.run_async(pool, inputs, output_pool=mem_pool)
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
                futures.append(session.run_async(pool, inputs, output_pool=mem_pool))

            for i, f in enumerate(futures):
                outs = f.get()
                done_t = time.perf_counter()
                latencies_us[i] = (done_t - submit_times[i]) * 1e6
                del outs
            total_wall_s = time.perf_counter() - t_start_wall

            p50_us, p90_us, p99_us = compute_percentiles_us(latencies_us)
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


def benchmark_dynamic_batcher_serving(
    session: InferenceSession,
    input_builder: InputBuilder,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    total_requests: int = 128,
    client_threads: int = 8,
    num_workers: int = 4,
    max_batch_delay_ms: int = 1,
    max_batch_delay_us: int = 500,
    arena_block_size: int = 0,
) -> list[BatchBenchmarkEntry]:
    """Benchmark concurrent multi-threaded serving through `DynamicBatcher`.

    Spawns `client_threads` concurrent Python threads that submit `batch_size=1` requests
    while releasing the Python GIL, allowing `DynamicBatcher` to coalesce requests up to
    each `max_batch_size` in `batch_sizes`.

    Args:
        session: Active `InferenceSession` to serve.
        input_builder: Callable taking `batch_size` and returning model input tensor(s).
        batch_sizes: Tuple of `max_batch_size` configurations to evaluate.
        total_requests: Total number of individual `B=1` requests across all client threads.
        client_threads: Number of concurrent Python client threads submitting requests.
        num_workers: Number of worker threads in `DynamicBatcher`'s backend pool.
        max_batch_delay_ms: Millisecond batch coalescing timeout window.
        max_batch_delay_us: Additional microsecond batch coalescing timeout window.
        arena_block_size: Optional custom arena block size in bytes for `DynamicBatcher`.

    Returns:
        List of `BatchBenchmarkEntry` results, one per `max_batch_size`.
    """
    results: list[BatchBenchmarkEntry] = []
    if arena_block_size <= 0:
        arena_block_size = compute_recommended_arena_block_size(
            session, input_builder=input_builder, max_batch_size=max(batch_sizes)
        )
    single_inputs = input_builder(1)
    requests_per_thread = max(1, total_requests // client_threads)
    effective_total = requests_per_thread * client_threads
    pipeline_depth = 8

    for max_b in batch_sizes:
        cfg = DynamicBatcherConfig(
            max_batch_size=max_b,
            max_batch_delay_ms=max_batch_delay_ms,
            max_batch_delay_us=max_batch_delay_us,
            max_queue_capacity=1024,
            num_workers=num_workers,
            arena_block_size=arena_block_size,
        )

        with DynamicBatcher(session, cfg) as batcher:
            warmup_count = min(8, effective_total)
            for _ in range(warmup_count):
                _ = batcher.run(single_inputs)
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
                        futs.append(batcher_ref.submit(single_inputs))
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
            p50_us, p90_us, p99_us = compute_percentiles_us(latencies_us)
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


def verify_benchmark_regressions(
    sync_heap: list[BatchBenchmarkEntry],
    sync_arena: list[BatchBenchmarkEntry],
    async_pool: list[BatchBenchmarkEntry],
    dynamic_batcher: list[BatchBenchmarkEntry],
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    max_p99_ms: float = 5000.0,
    min_scaling_factor: float = 1.2,
) -> list[str]:
    """Validate latency ordering, throughput scaling, and batch coalescing invariants.

    Args:
        sync_heap: Benchmark entries for synchronous heap-allocated execution.
        sync_arena: Benchmark entries for synchronous `ArenaAllocator` execution.
        async_pool: Benchmark entries for asynchronous `ThreadPool` execution.
        dynamic_batcher: Benchmark entries for concurrent `DynamicBatcher` serving.
        batch_sizes: Expected tuple of evaluated batch sizes.
        max_p99_ms: Maximum allowable P99 latency in milliseconds.
        min_scaling_factor: Minimum sample throughput ratio between max and min batch sizes.

    Returns:
        List of human-readable descriptions of all passed regression checks.

    Raises:
        AssertionError: If any latency, throughput, or batch coalescing invariant fails.
    """
    checks_passed: list[str] = []

    for entries, label in (
        (sync_heap, "sync_heap"),
        (sync_arena, "sync_arena"),
        (async_pool, "async_thread_pool"),
        (dynamic_batcher, "dynamic_batcher"),
    ):
        observed_batches = tuple(e.batch_size for e in entries)
        if observed_batches != batch_sizes:
            raise AssertionError(
                f"{label}: expected batch sizes {batch_sizes}, got {observed_batches}"
            )
        for e in entries:
            if not (0.0 < e.p50_us <= e.p90_us <= e.p99_us):
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): invalid latency percentile ordering "
                    f"P50={e.p50_us}us, P90={e.p90_us}us, P99={e.p99_us}us"
                )
            if e.p99_ms > max_p99_ms:
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): P99 latency {e.p99_ms:.2f}ms exceeds {max_p99_ms:.1f}ms ceiling"
                )
            if e.requests_per_sec <= 0 or e.samples_per_sec <= 0:
                raise AssertionError(
                    f"{label} (batch={e.batch_size}): non-positive throughput"
                )
    checks_passed.append(
        f"All batch sizes {batch_sizes} completed within P99 < {max_p99_ms:.0f}ms SLA"
    )

    if len(batch_sizes) >= 2 and batch_sizes[-1] > batch_sizes[0]:
        heap_b1 = sync_heap[0].samples_per_sec
        peak_sync_arena = max(e.samples_per_sec for e in sync_arena)
        peak_async_pool = max(e.samples_per_sec for e in async_pool)
        peak_samples = max(peak_sync_arena, peak_async_pool)

        if peak_samples < heap_b1 * min_scaling_factor:
            raise AssertionError(
                f"Sample throughput failed to scale: sync B{batch_sizes[0]}={heap_b1:.1f} vs peak={peak_samples:.1f}"
            )
        checks_passed.append(
            f"Peak sample throughput scaled {peak_samples / heap_b1:.2f}x over single-sample sync baseline ({heap_b1:.1f} -> {peak_samples:.1f} samples/sec)"
        )

    db_last = dynamic_batcher[-1]
    if db_last.batch_size > 1 and db_last.avg_coalesced_batch <= 1.0:
        raise AssertionError(
            f"DynamicBatcher failed to coalesce concurrent requests (avg_batch={db_last.avg_coalesced_batch})"
        )
    checks_passed.append(
        f"DynamicBatcher coalesced concurrent Python threads (avg batch size = {db_last.avg_coalesced_batch:.2f} at max_batch={db_last.batch_size})"
    )

    return checks_passed


def run_model_benchmark_suite(
    session: InferenceSession,
    input_builder: InputBuilder,
    model_id: str,
    task_name: str,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    iterations: int = 50,
    warmup: int = 10,
    dynamic_requests: int = 128,
    client_threads: int = 8,
    num_workers: int = 4,
    check_regressions: bool = True,
    max_p99_ms: float = 5000.0,
    min_scaling_factor: float = 1.2,
) -> dict[str, object]:
    """Execute all 4 Pyreth benchmark modes for a loaded `InferenceSession`.

    Args:
        session: Initialized `InferenceSession` for the target ONNX model.
        input_builder: Callable taking `batch_size` and returning model input tensor(s).
        model_id: HuggingFace model identifier or descriptive model name.
        task_name: Benchmark task category (e.g., `image_classification`, `small_llm`,
            `regression`).
        batch_sizes: Tuple of batch sizes to benchmark.
        iterations: Number of timed iterations per batch size for sync/async modes.
        warmup: Number of warmup iterations per batch size.
        dynamic_requests: Total concurrent requests per batch size for `DynamicBatcher`.
        client_threads: Number of concurrent Python client threads for `DynamicBatcher`.
        num_workers: Number of C++ worker threads in `ThreadPool` and `DynamicBatcher`.
        check_regressions: Whether to run `verify_benchmark_regressions` on the results.
        max_p99_ms: Maximum allowable P99 latency in milliseconds for regression checks.
        min_scaling_factor: Minimum sample throughput scaling ratio from min to max batch.

    Returns:
        Dictionary containing environment metadata, benchmark entries per mode, and
        regression check results.
    """
    arena_block_size = compute_recommended_arena_block_size(
        session, input_builder=input_builder, max_batch_size=max(batch_sizes)
    )

    sync_heap = benchmark_sync_session(
        session,
        input_builder=input_builder,
        batch_sizes=batch_sizes,
        warmup=warmup,
        iterations=iterations,
        use_arena=False,
        arena_block_size=arena_block_size,
    )
    sync_arena = benchmark_sync_session(
        session,
        input_builder=input_builder,
        batch_sizes=batch_sizes,
        warmup=warmup,
        iterations=iterations,
        use_arena=True,
        arena_block_size=arena_block_size,
    )
    async_pool = benchmark_async_thread_pool(
        session,
        input_builder=input_builder,
        batch_sizes=batch_sizes,
        warmup=warmup,
        iterations=iterations,
        num_workers=num_workers,
        arena_block_size=arena_block_size,
    )
    dynamic_batcher = benchmark_dynamic_batcher_serving(
        session,
        input_builder=input_builder,
        batch_sizes=batch_sizes,
        total_requests=dynamic_requests,
        client_threads=client_threads,
        num_workers=num_workers,
        arena_block_size=arena_block_size,
    )

    checks: list[str] = []
    if check_regressions:
        checks = verify_benchmark_regressions(
            sync_heap,
            sync_arena,
            async_pool,
            dynamic_batcher,
            batch_sizes=batch_sizes,
            max_p99_ms=max_p99_ms,
            min_scaling_factor=min_scaling_factor,
        )

    return {
        "task": task_name,
        "model_id": model_id,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "onnxruntime_version": Backend.get_runtime_version(),
            "best_provider": str(Backend.get_best_available_provider()),
            "active_providers": [str(p) for p in session.active_providers],
            "arena_block_size_bytes": arena_block_size,
        },
        "sync_heap": [asdict(e) for e in sync_heap],
        "sync_arena": [asdict(e) for e in sync_arena],
        "async_thread_pool": [asdict(e) for e in async_pool],
        "dynamic_batcher": [asdict(e) for e in dynamic_batcher],
        "regression_checks": checks,
        "_objects": (sync_heap, sync_arena, async_pool, dynamic_batcher),
    }


def print_table(title: str, entries: list[BatchBenchmarkEntry]) -> None:
    """Print a formatted ASCII benchmark table for a single execution mode.

    Args:
        title: Section title to display above the table.
        entries: List of `BatchBenchmarkEntry` rows to format.
    """
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


def print_benchmark_report(
    title: str,
    model_desc: str,
    report: dict[str, object],
    client_threads: int = 8,
    json_out: Path | None = None,
) -> None:
    """Print the full 4-mode benchmark report to stdout and optionally save it as JSON.

    Args:
        title: Banner title for the benchmark suite.
        model_desc: Human-readable model architecture and I/O summary.
        report: Benchmark report dictionary returned by `run_model_benchmark_suite`.
        client_threads: Number of concurrent Python client threads used in Mode 3.
        json_out: Optional file path where the JSON report should be written.
    """
    sync_heap, sync_arena, async_pool, dynamic_batcher = report.pop("_objects")  # type: ignore[misc]
    env = report["environment"]  # type: ignore[index]

    print(
        "================================================================================"
    )
    print(title)
    print(f"Platform     : {env['platform']} ({env['processor']})")
    print(f"Python       : {env['python_version']}")
    print(
        f"ONNX Runtime : {env['onnxruntime_version']} (Active Providers: {env['active_providers']})"
    )
    print(f"Model        : {model_desc}")
    print(f"Arena Block  : {env['arena_block_size_bytes'] // 1024} KiB")
    print(
        "================================================================================"
    )

    print_table(
        "1A. Synchronous InferenceSession.run (Heap-Allocated Outputs)", sync_heap
    )
    print_table(
        "1B. Synchronous InferenceSession.run (ArenaAllocator Reuse)", sync_arena
    )
    print_table(
        "2. Asynchronous InferenceSession.run_async (ThreadPool + MemoryPool)",
        async_pool,
    )
    print_table(
        f"3. Concurrent DynamicBatcher Serving ({client_threads} Python Threads)",
        dynamic_batcher,
    )

    if report["regression_checks"]:
        print("\n=== Automated Regression Checks ===")
        for chk in report["regression_checks"]:  # type: ignore[attr-defined]
            print(f"  [PASS] {chk}")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved JSON benchmark report to {json_out}")
