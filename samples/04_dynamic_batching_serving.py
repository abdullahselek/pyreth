"""Sample 04: High-Throughput Concurrent Model Serving with DynamicBatcher.

Demonstrates:
  1. Configuring `DynamicBatcherConfig` (max batch size, latency deadline, queue capacity)
  2. Asynchronous request pipelining via `batcher.submit()` and `batcher.flush()`
  3. Multi-threaded Python serving (`ThreadPoolExecutor`) calling `batcher.run_numpy()`
     concurrently—releasing the Python GIL so individual requests coalesce into
     high-throughput batched ONNX Runtime executions and demultiplex with zero copy
  4. Inspecting `DynamicBatcherMetrics` telemetry

Usage:
    uv run python samples/04_dynamic_batching_serving.py
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from models import (
    build_mlp_onnx_bytes,
    build_serving_mlp_onnx_bytes,
    expected_mlp_numpy,
)
from pyreth import (
    DynamicBatcher,
    DynamicBatcherConfig,
    InferenceSession,
    MemoryPool,
    SessionConfig,
    ThreadPool,
)


def demo_async_submit_and_flush() -> None:
    print("=== 1. Asynchronous Batch Submission (`submit` + `flush`) ===")

    session = InferenceSession(
        build_mlp_onnx_bytes(), SessionConfig(intra_op_num_threads=1)
    )
    cfg = DynamicBatcherConfig(
        max_batch_size=8,
        max_batch_delay_ms=20,
        max_queue_capacity=256,
        num_workers=2,
    )

    with DynamicBatcher(session, cfg) as batcher:
        samples = [
            np.array([[float(i), -float(i) * 0.5, 1.0, 0.25]], dtype=np.float32)
            for i in range(12)
        ]

        # Enqueue 12 single-sample requests asynchronously; first 8 trigger a full batch
        # immediately, and flush() dispatches the remaining 4 without waiting for timeout.
        futures = [batcher.submit(x) for x in samples]
        batcher.flush()

        for x, fut in zip(samples, futures, strict=True):
            y_out = fut.get_numpy()[0]
            np.testing.assert_allclose(
                y_out, expected_mlp_numpy(x), rtol=1e-5, atol=1e-5
            )

        m = batcher.metrics()
        print(
            f"Submitted {m.requests_submitted} requests -> {m.batches_dispatched} batches"
        )
        print(
            f"  Full batches: {m.full_batches}, Timeout/flushed batches: {m.timeout_batches}, "
            f"Avg batch size: {m.average_batch_size:.2f}"
        )


def demo_multithreaded_gil_free_serving() -> None:
    print("\n=== 2. Concurrent Multi-Threaded Serving (GIL Released) ===")

    in_features = 64
    out_features = 16
    model_bytes = build_serving_mlp_onnx_bytes(
        in_features=in_features, hidden_features=128, out_features=out_features
    )

    session = InferenceSession(model_bytes, SessionConfig(intra_op_num_threads=1))
    worker_pool = ThreadPool(num_threads=4)
    mem_pool = MemoryPool(initial_count=16, block_size=256 * 1024)

    batcher_cfg = DynamicBatcherConfig(
        max_batch_size=8,
        max_batch_delay_ms=2,
        max_queue_capacity=1024,
    )

    num_client_threads = 8
    requests_per_client = 32
    total_requests = num_client_threads * requests_per_client

    with DynamicBatcher(
        session, batcher_cfg, pool=worker_pool, memory_pool=mem_pool
    ) as batcher:

        def client_worker(request_id: int) -> tuple[int, int]:
            # Simulate an incoming single-sample client request (e.g., FastAPI / gRPC handler)
            x = np.full(
                (1, in_features), float(request_id % 10) * 0.1, dtype=np.float32
            )
            # `run_numpy` releases the Python GIL while waiting for batch formation & execution
            outputs = batcher.run_numpy(x)
            return outputs[0].shape

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_client_threads) as executor:
            output_shapes = list(executor.map(client_worker, range(total_requests)))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert all(s == (1, out_features) for s in output_shapes)

        metrics = batcher.metrics()
        throughput = total_requests / (elapsed_ms / 1000.0)
        print(
            f"Served {total_requests} concurrent requests across {num_client_threads} Python threads in {elapsed_ms:.2f} ms"
        )
        print(f"  Throughput           : {throughput:,.1f} req/sec")
        print(
            f"  Batches dispatched   : {metrics.batches_dispatched} (full={metrics.full_batches}, timeout={metrics.timeout_batches})"
        )
        print(
            f"  Average batch size   : {metrics.average_batch_size:.2f} samples/batch"
        )
        print(f"  Avg batch wait time  : {metrics.average_batch_wait_us:.2f} us")
        print(f"  Avg ORT exec time    : {metrics.average_execution_us:.2f} us")

    worker_pool.shutdown()


def main() -> None:
    demo_async_submit_and_flush()
    demo_multithreaded_gil_free_serving()


if __name__ == "__main__":
    main()
