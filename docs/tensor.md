# Tensor Abstraction & Zero-Copy Lifetime

High-performance AI model inference requires an expressive multidimensional tensor abstraction with zero-copy view transformations (reshaping, slicing, transposing) and strict memory lifetime contracts.

Pyreth introduces **`Tensor`**, **`Shape`**, **`DataType`**, and **`TensorStorage`** ([`include/pyreth/tensor.hpp`](../include/pyreth/tensor.hpp)), backed directly by Pyreth's 64-byte aligned SIMD memory architecture.

---

## 1. Core Architecture

```
+--------------------------------------------------------------------------------+
|                                 pyreth::Tensor                                 |
|                                                                                |
|   +-------------------+  +-------------------+  +--------------------------+   |
|   | Shape             |  | Strides           |  | DataType                 |   |
|   | [D_0, D_1, ..., ] |  | [S_0, S_1, ..., ] |  | Float32, Int64, etc.     |   |
|   +-------------------+  +-------------------+  +--------------------------+   |
|                                                                                |
|   +-------------------+  +-------------------+                                 |
|   | Byte Offset       |  | Data Pointer      |                                 |
|   | (Start in buffer) |  | (Aligned element) |                                 |
|   +-------------------+  +-------------------+                                 |
|             |                      |                                           |
+-------------|----------------------|-------------------------------------------+
              |                      |
              v                      v
+--------------------------------------------------------------------------------+
|                        std::shared_ptr<TensorStorage>                          |
|                                                                                |
|   +------------------+  +-------------------+  +---------------------------+   |
|   | OwnedStorage     |  | ArenaStorage      |  | PooledArenaStorage        |   |
|   | (SIMD 64-B Heap) |  | (Arena Pinning)   |  | (MemoryPool RAII Lease)   |   |
|   +------------------+  +-------------------+  +---------------------------+   |
|                                                                                |
|   +------------------------------------------------------------------------+   |
|   | ExternalStorage (Zero-copy wrapper over NumPy, ORT, or mapped memory)  |   |
|   +------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------+
```

### Key Principles
1. **Zero-Copy Views**: Reshaping and slicing do not allocate or copy payload memory. They construct lightweight `Tensor` views that share the underlying `TensorStorage`.
2. **Hardware Alignment**: All newly allocated tensors (owned, arena, or pooled) are strictly 64-byte aligned (`reinterpret_cast<uintptr_t>(ptr) % 64 == 0`), eliminating unaligned vector penalties on AVX-512 and ARM NEON cache lines.
3. **Strict Lifetime Pinning**: Backing arenas cannot be reset while active tensor references exist. The memory subsystem enforces RAII pinning and shared leases to guarantee that backing buffers are never recycled out from under active inference pipelines.
4. **Exception-Free Monadic API**: Factory, reshape, and slice operations return `Result<Tensor>` with discrete `StatusCode` errors, maintaining zero-allocation determinism along hot paths.

---

## 2. Strided Memory Layout & Row-Major Order

### 2.1 Indexing & Strided Offset Formula
For a tensor with rank $K$, shape $[D_0, D_1, \dots, D_{K-1}]$, strides $[S_0, S_1, \dots, S_{K-1}]$, and element size $E$, the address of element at coordinates $(c_0, c_1, \dots, c_{K-1})$ is computed as:

$$\text{Address} = \text{base\_data} + \left( \sum_{i=0}^{K-1} c_i \times S_i \right) \times E$$

For standard contiguous row-major layout:
- $S_{K-1} = 1$
- $S_i = S_{i+1} \times D_{i+1}$ for $i = K-2, \dots, 0$

### 2.2 Contiguity Detection
A tensor is contiguous in row-major order if and only if adjacent elements along the last non-singleton dimension reside contiguously in memory without strides or gaps.
- Slicing dimension 0 of a contiguous tensor retains contiguity:
  $$\text{slice}(0, \text{start}, \text{end}) \implies \text{isContiguous}() == \text{true}$$
- Slicing columns or inner dimensions introduces non-unit strides:
  $$\text{slice}(1, \text{start}, \text{end}) \implies \text{isContiguous}() == \text{false}$$

---

## 3. Transformations: Reshaping & Slicing

### 3.1 Reshaping
Reshaping constructs a new `Tensor` view over the same `TensorStorage`:
- **Element Invariance**: The target shape must have the same total element count as the original shape (`target.numElements() == original.numElements()`).
- **Dimension Inference (`-1`)**: Exactly one dimension in the target shape may be specified as `-1`. The engine infers the missing dimension by dividing the total element count by the product of the known dimensions.
- **Contiguity Requirement**: Reshaping requires the source tensor to be contiguous. For non-contiguous views (e.g. column slices), `toContiguous()` must be invoked first.

```cpp
auto t = Tensor::create({2, 3, 4}, DataType::Float32).value(); // 24 elements

// Contiguous zero-copy reshape
Tensor reshaped = t.reshape(Shape({6, 4})).value();

// Inferred dimension (-1 -> 8)
Tensor inferred = t.reshape(Shape({3, -1})).value();
```

### 3.2 Contiguous Slicing
Slicing creates a sub-view over dimension `dim` in the range `[start, end)`:
- **New Shape**: `new_shape[dim] = (end - start + step - 1) / step`
- **New Strides**: `new_strides[dim] = strides[dim] * step`
- **New Offset**: `byte_offset + start * strides[dim] * elementSize()`

```cpp
auto matrix = Tensor::create({4, 8}, DataType::Float32).value();

// Contiguous row slice: rows 1..3
Tensor row_slice = matrix.slice(0, 1, 3).value();
assert(row_slice.isContiguous());

// Non-contiguous column slice: columns 2..6
Tensor col_slice = matrix.slice(1, 2, 6).value();
assert(!col_slice.isContiguous());
```

---

## 4. Storage Hierarchy & Lifetime Contracts

Pyreth provides four backing storage strategies managed via `TensorStorage`:

| Storage Type | Allocation Source | Lifetime Contract | Use Case |
| :--- | :--- | :--- | :--- |
| **`OwnedStorage`** | 64-byte aligned heap | Freed when last `Tensor` reference drops | Unit tests, standalone buffers |
| **`ArenaStorage`** | `ArenaAllocator` | **Pins arena**; arena `reset()` is safely blocked until unpinned | Per-request scratchpads |
| **`PooledArenaStorage`**| `MemoryPool::Lease` | **Holds lease**; arena reset in $O(1)$ and returned to pool upon destruction | High-throughput concurrent serving |
| **`ExternalStorage`** | Raw external pointer | Retains custom `std::shared_ptr<void>` keepalive handle | NumPy / ONNX Runtime interop |

### 4.1 Arena Pinning Contract
When an `ArenaStorage` is instantiated:
1. `arena.pin()` increments the atomic pin counter (`pinned_count_`).
2. If `arena.reset()` or `arena.tryReset()` is invoked while `arena.isPinned()` is true, the allocator safely refuses to rewind the bump offset, protecting active tensor buffers from memory corruption.
3. When the last referencing `Tensor` or slice is destroyed, `arena.unpin()` decrements the pin count. Once all pins are released, the arena can be safely reset in $O(1)$.

```cpp
ArenaAllocator arena(1024 * 1024);

{
  Tensor t = Tensor::create({8, 8}, DataType::Float32, arena).value();
  assert(arena.isPinned());
  
  // tryReset fails while t is alive
  Result<void> res = arena.tryReset();
  assert(!res.isOk());
  
  // reset() safely no-ops while t is alive
  arena.reset();
  assert(t.at<float>(0, 0) == /* preserved */);
}
// t destroyed: arena unpinned
assert(!arena.isPinned());
assert(arena.tryReset().isOk());
```

### 4.2 MemoryPool Lease Retention
When allocating from a `MemoryPool`:
```cpp
MemoryPool pool(4);
{
  Tensor t1 = Tensor::create({64, 64}, DataType::Float32, pool).value();
  assert(pool.availableCount() == 3); // 1 arena leased

  {
    Tensor slice = t1.slice(0, 0, 32).value();
    assert(pool.availableCount() == 3); // Lease still active
  }
}
// Both t1 and slice destroyed: arena reset in O(1) and returned to pool
assert(pool.availableCount() == 4);
```

---

## 5. API Reference

### 5.1 Factory Methods
```cpp
// Standalone SIMD 64-byte aligned
Result<Tensor> t1 = Tensor::create(Shape({1, 3, 224, 224}), DataType::Float32);

// Arena-backed with pinning
Result<Tensor> t2 = Tensor::create(Shape({128, 768}), DataType::Float32, arena);

// Pooled arena with automatic RAII lease recycling
Result<Tensor> t3 = Tensor::create(Shape({32, 1024}), DataType::Float32, pool);

// External buffer zero-copy wrapper
Result<Tensor> t4 = Tensor::fromBuffer(raw_ptr, Shape({10, 10}), DataType::Int64);
```

### 5.2 Element Indexing & Mutation
```cpp
Tensor t = Tensor::create({4, 8}, DataType::Float32).value();

// Multi-dimensional coordinate indexing
t.at<float>(1, 2) = 3.14f;
float val = t.at<float>(1, 2);

// Linear flat indexing (contiguous or strided)
t.flatAt<float>(0) = 1.0f;

// Fill & Zero
t.fill(0.5f);
t.zero();
```

### 5.3 Move Semantics & Ownership Transfer
```cpp
Tensor t1 = Tensor::create({4, 4}, DataType::Float32).value();

// Move constructor: transfers ownership, invalidates t1
Tensor t2 = std::move(t1);
assert(!t1.isValid());
assert(t2.isValid());

// Shallow copy: shares storage view
Tensor t3 = t2;
assert(t2.useCount() == 2);
```

### 5.4 Contiguous Normalization & Cloning
```cpp
// If slice is non-contiguous, convert to contiguous memory
Result<Tensor> contig = col_slice.toContiguous();

// Deep copy: independent storage
Tensor cloned = t2.clone();
```
