"""Sample 03: Asynchronous Parallel Inference with Work-Stealing ThreadPool.

Demonstrates:
  1. Computing optimal thread budgets between Pyreth and ONNX Runtime (`compute_thread_budget`)
  2. Configuring Pyreth's lock-free Chase-Lev `ThreadPool` and reusable `MemoryPool`
  3. Dispatching non-blocking multi-input/multi-output inference via `session.run_async`
  4. Polling, waiting with timeouts, and retrieving zero-copy outputs via `InferenceFuture`
  5. Inspecting `ThreadPoolMetrics` (submitted, executed, and work-stolen tasks)

Usage:
    uv run python samples/03_async_thread_pool.py
"""

import numpy as np
from models import build_dual_io_onnx_bytes
from pyreth import (
    InferenceSession,
    MemoryPool,
    SessionConfig,
    ThreadPool,
    ThreadPoolConfig,
)


def main() -> None:
    print("=== Asynchronous Inference with ThreadPool & MemoryPool ===")

    # 1. Compute recommended thread split between Pyreth workers and ORT intra-op threads
    budget = ThreadPool.compute_thread_budget(reserved_for_ort=1)
    print(
        f"Recommended ThreadBudget: pyreth_workers={budget.pyreth_worker_threads}, "
        f"ort_intra_op={budget.ort_intra_op_threads}, ort_inter_op={budget.ort_inter_op_threads}"
    )

    # 2. Create a session for a dual-input, dual-output model: sum = a + b, prod = a * b
    features = 8
    session_cfg = SessionConfig(intra_op_num_threads=budget.ort_intra_op_threads)
    session = InferenceSession(build_dual_io_onnx_bytes(features=features), session_cfg)
    print(
        f"Loaded model '{session.model_info.graph_name}': inputs={session.input_names}, outputs={session.output_names}"
    )

    # 3. Configure a reusable MemoryPool for output tensors and a lock-free ThreadPool
    output_pool = MemoryPool(initial_count=16, block_size=64 * 1024)
    pool_cfg = ThreadPoolConfig(
        num_threads=max(2, budget.pyreth_worker_threads),
        queue_capacity=512,
        reserve_threads_for_ort=budget.ort_intra_op_threads,
    )

    num_requests = 32
    with ThreadPool(pool_cfg) as pool:
        print(f"Started ThreadPool with {pool.num_workers} worker threads.")

        futures = []
        expected_results = []

        # Submit 32 asynchronous inference requests without blocking the main thread
        for i in range(num_requests):
            a = np.full((4, features), float(i + 1), dtype=np.float32)
            b = np.full((4, features), 0.5, dtype=np.float32)
            expected_results.append((a + b, a * b))

            future = session.run_async(pool, {"a": a, "b": b}, output_pool=output_pool)
            futures.append(future)

        # Wait for each future and verify both output tensors (sum and product)
        for i, fut in enumerate(futures):
            ready = fut.wait_for(timeout_ms=5000.0)
            assert ready, f"Request {i} timed out"

            sum_out, prod_out = fut.get_numpy()
            exp_sum, exp_prod = expected_results[i]
            np.testing.assert_allclose(sum_out, exp_sum, rtol=1e-5)
            np.testing.assert_allclose(prod_out, exp_prod, rtol=1e-5)

        metrics = pool.metrics()
        print(f"\nCompleted all {num_requests} async requests successfully!")
        print(f"ThreadPool Metrics      : {metrics.to_dict()}")
        print(f"MemoryPool Arenas Ready : {output_pool.available_count}")


if __name__ == "__main__":
    main()
