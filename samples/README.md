# Pyreth Python Samples

Self-contained runnable Python examples demonstrating the core use cases of the **Pyreth** inference engine. All samples generate minimal ONNX models in memory via [`models.py`](models.py), so no external model downloads or protobuf dependencies are required.

## Sample Overview

| File | Use Case Covered |
| :--- | :--- |
| [`01_tensor_and_memory.py`](01_tensor_and_memory.py) | Bidirectional zero-copy `numpy.ndarray` $\leftrightarrow$ `pyreth.Tensor` interoperability, 64-byte SIMD alignment, zero-copy `reshape` and strided `slice`, `ArenaAllocator` pin counting & $O(1)$ `reset()`, and `MemoryPool` lease recycling. |
| [`02_model_inspection_and_sync_inference.py`](02_model_inspection_and_sync_inference.py) | `Runtime` initialization, `Backend` hardware provider discovery, `SessionConfig`, `ModelInfo` / `TensorInfo` inspection, input validation, and synchronous inference (`run`, `run_numpy`, `ArenaAllocator` output reuse, and in-place pre-allocated outputs). |
| [`03_async_thread_pool.py`](03_async_thread_pool.py) | `ThreadBudget` calculation, lock-free work-stealing `ThreadPool` + `MemoryPool`, non-blocking multi-input/multi-output `session.run_async()`, `InferenceFuture` synchronization, and `ThreadPoolMetrics`. |
| [`04_dynamic_batching_serving.py`](04_dynamic_batching_serving.py) | High-throughput request coalescing with `DynamicBatcher`, async pipelining (`submit` + `flush`), multi-threaded GIL-free serving (`ThreadPoolExecutor` + `batcher.run_numpy`), and `DynamicBatcherMetrics` telemetry. |
| [`05_advanced_features.py`](05_advanced_features.py) | Multi-dtype tensors (`Float16`, `BFloat16`, `Int32`, `Bool`, dtype strings), 0-D scalars, `np.asarray` protocol, arena-backed `clone` / `to_contiguous` & `force_reset`, granular `ModelInfo` / `TensorInfo` lookups & validation, fluent `SessionConfig` / `BatcherConfig` builders with ONNX Runtime profiling, pooled sync inference (`run(..., pool=...)`), `InferenceFuture` polling (`wait`, `is_ready`, `done`, `result`), `reset_metrics`, and `Runtime.shutdown()`. |
| [`models.py`](models.py) | Pure-Python ONNX model builders used by the sample scripts. |

## Running the Samples

From the repository root using `uv`:

```sh
uv run python samples/01_tensor_and_memory.py
uv run python samples/02_model_inspection_and_sync_inference.py
uv run python samples/03_async_thread_pool.py
uv run python samples/04_dynamic_batching_serving.py
uv run python samples/05_advanced_features.py
```
