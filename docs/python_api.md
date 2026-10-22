# Python API: Zero-Copy Tensor, Engine API & GIL Management

Pyreth exposes its full C++20 inference engine—`Tensor`, `ArenaAllocator`, `MemoryPool`, `InferenceSession`, `ModelInfo`, `ThreadPool`, `DynamicBatcher`, and runtime telemetry—to Python via `nanobind`, combining zero-copy NumPy interoperability with explicit Python GIL release (`nb::gil_scoped_release`) for true multi-threaded scaling across CPU cores.

---

## 1. Bidirectional Zero-Copy Architecture

```
+--------------------------------------------------------------------------------+
|                             Python / NumPy Layer                               |
|                                                                                |
|   +------------------------------------+  +--------------------------------+   |
|   | numpy.ndarray (Source Buffer)      |  | numpy.ndarray (Exported View)  |   |
|   +-----------------+------------------+  +---------------+----------------+   |
+---------------------|-------------------------------------|--------------------+
                      |                                     ^
                      | Tensor.from_numpy(arr)              | tensor.numpy() /
                      | (Zero-Copy Pointer)                 | np.asarray(tensor)
                      v                                     |
+--------------------------------------------------------------------------------+
|                          nanobind Zero-Copy Bridge                             |
|                                                                                |
|   +------------------------------------+  +--------------------------------+   |
|   | shared_ptr<void> Custom Deleter    |  | nanobind::capsule Owner        |   |
|   | (Py_INCREF / Py_DECREF with GIL)   |  | (shared_ptr<TensorStorage>)    |   |
|   +-----------------+------------------+  +---------------+----------------+   |
+---------------------|-------------------------------------|--------------------+
                      |                                     |
                      v                                     v
+--------------------------------------------------------------------------------+
|                             Pyreth C++20 Core                                  |
|                                                                                |
|   +------------------------------------------------------------------------+   |
|   |                             pyreth::Tensor                             |   |
|   +-----------------------------------+------------------------------------+   |
|                                       |                                        |
|                                       v                                        |
|   +-----------------+  +--------------+---+  +-------------+  +------------+   |
|   | ExternalStorage |  | OwnedStorage     |  | ArenaStorage|  | PooledArena|   |
|   | (Pins PyObject*)|  | (64B SIMD Heap)  |  | (Pins Arena)|  | (PoolLease)|   |
|   +-----------------+  +------------------+  +-------------+  +------------+   |
+--------------------------------------------------------------------------------+
```

---

## 2. Memory Ownership & Lifetime Guarantees

### 2.1 NumPy $\rightarrow$ `pyreth.Tensor` (`StorageType.EXTERNAL`)
When wrapping an existing `numpy.ndarray` via `Tensor.from_numpy(arr)` or `Tensor(arr)`:
1. **Zero-Copy Buffer Wrapping**: Pyreth extracts the raw data pointer, element shape, element strides, and DLPack data type directly from `nanobind::ndarray` without copying payload bytes (`tensor.data_ptr == arr.__array_interface__["data"][0]`).
2. **Python Reference Counting**: `ExternalStorage` is initialized with a custom `std::shared_ptr<void>` deleter that immediately calls `Py_INCREF` on the underlying `PyObject*`.
3. **Thread-Safe Destruction**: Even if the caller deletes `arr` in Python (`del arr; gc.collect()`), the NumPy memory buffer remains valid as long as the `pyreth.Tensor` (or any C++ slice, reshape, or in-flight `InferenceSession` / `DynamicBatcher` request) holds a reference to `ExternalStorage`. When the last reference drops—even on a C++ worker thread—the custom deleter safely acquires the GIL (`nb::gil_scoped_acquire`) and calls `Py_DECREF`.

### 2.2 `pyreth.Tensor` $\rightarrow$ NumPy (`StorageType.OWNED` / `ARENA` / `POOLED`)
When exporting a `pyreth.Tensor` to NumPy via `tensor.numpy()`, `tensor.to_numpy()`, or `np.asarray(tensor)`:
1. **64-Byte SIMD Alignment**: Tensors allocated by Pyreth (`Tensor.create`, `Tensor.zeros`) are 64-byte aligned (`arr.ctypes.data % 64 == 0`), enabling optimal AVX-512, ARM NEON, and Apple AMX vectorization in downstream NumPy/SciPy kernels.
2. **Capsule-Managed `TensorStorage` Pinning**: `tensor.numpy()` attaches a `nanobind::capsule` owning a heap-allocated `std::shared_ptr<TensorStorage>` to the returned `numpy.ndarray`.
3. **Arena & Pool Safety**:
   - For **Arena-backed** tensors (`StorageType.ARENA`), the `ArenaAllocator` remains pinned (`arena.is_pinned == True`, `arena.pin_count >= 1`) for as long as *either* the `pyreth.Tensor` *or* any derived NumPy array/slice exists in Python.
   - For **Pool-backed** tensors (`StorageType.POOLED`), the `MemoryPool::Lease` is retained until the last NumPy view is garbage-collected, at which point the arena automatically resets in $O(1)$ and returns to the `MemoryPool`.

---

## 3. Supported Data Types

| `pyreth.DataType` | NumPy `dtype` | Element Size (Bytes) | Bidirectional Zero-Copy |
| :--- | :--- | :--- | :--- |
| `DataType.Float32` | `np.float32` | 4 | Yes |
| `DataType.Float64` | `np.float64` | 8 | Yes |
| `DataType.Float16` | `np.float16` | 2 | Yes |
| `DataType.BFloat16` | *(N/A in standard NumPy)* | 2 | C++ / ORT only |
| `DataType.Int8` | `np.int8` | 1 | Yes |
| `DataType.UInt8` | `np.uint8` | 1 | Yes |
| `DataType.Int16` | `np.int16` | 2 | Yes |
| `DataType.UInt16` | `np.uint16` | 2 | Yes |
| `DataType.Int32` | `np.int32` | 4 | Yes |
| `DataType.UInt32` | `np.uint32` | 4 | Yes |
| `DataType.Int64` | `np.int64` | 8 | Yes |
| `DataType.UInt64` | `np.uint64` | 8 | Yes |
| `DataType.Bool` | `np.bool_` | 1 | Yes |

### Validation Rules
- **Read-Only Arrays**: Passing a read-only NumPy array (`arr.flags.writeable = False`) to `Tensor.from_numpy()` raises `ValueError` to prevent unintended mutation of immutable buffers.
- **Negative Strides**: Arrays with negative strides (e.g. `arr[::-1]`) raise `ValueError`; call `np.ascontiguousarray(arr)` prior to wrapping.
- **Unsupported DTypes**: Complex numbers (`np.complex64`, `np.complex128`), strings, and Python object arrays raise `TypeError`.

---

## 4. GIL Management & Multi-Threaded Scaling

Python's Global Interpreter Lock (GIL) serializes bytecode execution across Python threads. To allow Python serving frameworks (e.g. FastAPI, gRPC, `ThreadPoolExecutor`) to scale concurrently across all CPU cores, Pyreth explicitly releases the GIL via `nb::gil_scoped_release` across all compute-bound and blocking C++ operations:

```
Python Worker Thread 1               Python Worker Thread N
          |                                    |
          | 1. Coerce np.ndarray -> Tensor     | 1. Coerce np.ndarray -> Tensor
          |    (Py_INCREF under GIL)           |    (Py_INCREF under GIL)
          v                                    v
+------------------------------------------------------------------------+
|                 nb::gil_scoped_release (GIL Released)                  |
|                                                                        |
|  - InferenceSession.run() / run_numpy() / run_async()                  |
|  - DynamicBatcher.submit() / run() / run_numpy() / flush() / shutdown()|
|  - InferenceFuture.wait() / wait_for() / get() / get_numpy()           |
|  - ThreadPool.shutdown()                                               |
|                                                                        |
|  Concurrent Python threads coalesce into batches inside DynamicBatcher |
|  and execute in parallel on Pyreth's lock-free C++20 ThreadPool.       |
+------------------------------------------------------------------------+
          |                                    |
          | 2. Re-acquire GIL                  | 2. Re-acquire GIL
          v                                    v
Return Zero-Copy Tensor / ndarray    Return Zero-Copy Tensor / ndarray
```

---

## 5. Quick Start Guide

### 5.1 Zero-Copy Tensor & NumPy Interoperability
```python
import numpy as np
import pyreth

# 1. NumPy -> pyreth.Tensor (Zero-Copy)
arr = np.zeros((4, 8), dtype=np.float32)
tensor = pyreth.Tensor.from_numpy(arr)
assert tensor.data_ptr == arr.__array_interface__["data"][0]

# In-place mutation via Pyreth reflects immediately in NumPy
tensor.fill(2.5)
assert np.all(arr == 2.5)

# 2. pyreth.Tensor -> NumPy (Zero-Copy, 64-byte SIMD aligned)
aligned_tensor = pyreth.Tensor.zeros((16, 64), dtype=pyreth.DataType.Float32)
view = aligned_tensor.numpy()
assert view.ctypes.data % 64 == 0
```

### 5.2 Loading a Model & Inspecting Metadata (`InferenceSession` & `ModelInfo`)
```python
import numpy as np
import pyreth

config = pyreth.SessionConfig(
    intra_op_num_threads=1,
    graph_optimization_level=pyreth.OptimizationLevel.All,
    providers=[pyreth.ExecutionProvider.COREML, pyreth.ExecutionProvider.CPU],
)

session = pyreth.InferenceSession("model.onnx", config)

# Inspect cached ModelInfo
info = session.model_info
print(f"Graph: {info.graph_name}, Dynamic Batch: {info.has_dynamic_batch}")
for inp in session.inputs:
    print(
        f"Input: {inp.name}, dtype={inp.dtype}, shape={inp.shape}, symbolic={inp.symbolic_dims}"
    )

# Run synchronous zero-copy inference directly on NumPy arrays (releases GIL)
x = np.random.randn(4, 3, 224, 224).astype(np.float32)
outputs = session.run_numpy(x)
print("Output shape:", outputs[0].shape)
```

### 5.3 Asynchronous Execution on `ThreadPool`
```python
import numpy as np
import pyreth

session = pyreth.InferenceSession("model.onnx")
mem_pool = pyreth.MemoryPool(initial_count=8, block_size=1024 * 1024)

with pyreth.ThreadPool(num_threads=4) as pool:
    futures = [
        session.run_async(pool, np.ones((1, 4), dtype=np.float32), output_pool=mem_pool)
        for _ in range(16)
    ]
    results = [fut.get_numpy()[0] for fut in futures]
    print("ThreadPool metrics:", pool.metrics().to_dict())
```

### 5.4 High-Throughput Serving with `DynamicBatcher`
```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyreth

session = pyreth.InferenceSession("model.onnx")
batch_cfg = pyreth.DynamicBatcherConfig(
    max_batch_size=16,
    max_batch_delay_ms=5,
    max_queue_capacity=1024,
)

with pyreth.DynamicBatcher(session, batch_cfg) as batcher:

    def infer_one(req_id: int) -> np.ndarray:
        x = np.full((1, 4), float(req_id), dtype=np.float32)
        # Releases GIL while waiting for batch coalescing and ORT execution
        return batcher.run_numpy(x)[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(infer_one, range(64)))

    metrics = batcher.metrics()
    print(
        f"Dispatched {metrics.batches_dispatched} batches "
        f"(avg batch size: {metrics.average_batch_size:.1f})"
    )
```
