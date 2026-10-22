# Python API: Tensor & Zero-Copy NumPy Integration

Pyreth exposes its C++20 `Tensor`, `ArenaAllocator`, and `MemoryPool` primitives to Python via `nanobind::ndarray`, providing bidirectional **zero-copy** interoperability with NumPy while enforcing strict reference-counted memory lifetime guarantees across the Python/C++ boundary.

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

### 2.1 NumPy -> `pyreth.Tensor` (`StorageType.EXTERNAL`)
When wrapping an existing `numpy.ndarray` via `Tensor.from_numpy(arr)` or `Tensor(arr)`:
1. **Zero-Copy Buffer Wrapping**: Pyreth extracts the raw data pointer, element shape, element strides, and DLPack data type directly from `nanobind::ndarray` without copying payload bytes (`tensor.data_ptr == arr.__array_interface__["data"][0]`).
2. **Python Reference Counting**: `ExternalStorage` is initialized with a custom `std::shared_ptr<void>` deleter that immediately calls `Py_INCREF` on the underlying `PyObject*`.
3. **Thread-Safe Destruction**: Even if the caller deletes `arr` in Python (`del arr; gc.collect()`), the NumPy memory buffer remains valid as long as the `pyreth.Tensor` (or any C++ slice, reshape, or in-flight `InferenceSession` / `DynamicBatcher` request) holds a reference to `ExternalStorage`. When the last reference drops—even on a C++ worker thread—the custom deleter safely acquires the GIL (`nb::gil_scoped_acquire`) and calls `Py_DECREF`.

### 2.2 `pyreth.Tensor` -> NumPy (`StorageType.OWNED` / `ARENA` / `POOLED`)
When exporting a `pyreth.Tensor` to NumPy via `tensor.numpy()`, `tensor.to_numpy()`, or `np.asarray(tensor)`:
1. **64-Byte SIMD Alignment**: Tensors allocated by Pyreth (`Tensor.create`, `Tensor.zeros`) are 64-byte aligned (`arr.ctypes.data % 64 == 0`), enabling optimal AVX-512, ARM NEON, and Apple AMX vectorization in downstream NumPy/SciPy kernels.
2. **Capsule-Managed `TensorStorage` Pinning**: `tensor.numpy()` attaches a `nanobind::capsule` owning a heap-allocated `std::shared_ptr<TensorStorage>` to the returned `numpy.ndarray`.
3. **Arena & Pool Safety**:
   - For **Arena-backed** tensors (`StorageType.ARENA`), the `ArenaAllocator` remains pinned (`arena.is_pinned == True`, `arena.pin_count >= 1`) for as long as *either* the `pyreth.Tensor` *or* any derived NumPy array/slice exists in Python. Calling `arena.reset()` while a NumPy view is alive is prevented from recycling active memory.
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

## 4. Python Usage Examples

### 4.1 Zero-Copy Wrapping & In-Place Mutation
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

view[0, :] = 1.0
assert aligned_tensor.numpy()[0, 0] == 1.0
```

### 4.2 Arena-Backed Tensors in Python
```python
import gc
import pyreth

arena = pyreth.ArenaAllocator(block_size=1024 * 1024)
tensor = pyreth.Tensor.zeros((32, 32), dtype="float32", arena=arena)

view = tensor.numpy()
del tensor
gc.collect()

# Arena remains safely pinned while `view` is alive in Python
assert arena.is_pinned

del view
gc.collect()

# Arena is automatically unpinned and ready for O(1) reset
assert not arena.is_pinned
arena.reset()
```
