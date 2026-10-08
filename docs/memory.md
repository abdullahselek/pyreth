# Memory Architecture & Arena Allocator

High-performance AI model inference requires deterministic memory latency, zero allocator lock contention, and strict hardware alignment for SIMD vector units. General-purpose dynamic allocators (`malloc`, `free`, `new`, `delete`) introduce heap fragmentation, thread synchronization locks, and variable-time system call overhead on inference hot paths.

Pyreth introduces a dedicated memory subsystem comprising the **64-byte aligned `ArenaAllocator`** and the **`MemoryPool`** ([`include/pyreth/memory_pool.hpp`](../include/pyreth/memory_pool.hpp)).

---

## 1. Core Architecture

```
+-------------------------------------------------------------------+
|                         Pyreth MemoryPool                         |
|                                                                   |
|   +--------------------------+     +--------------------------+   |
|   | Thread-Local Arena       |     | Reusable Leased Arenas   |   |
|   | (Worker Thread Scratch)  |     | (Dynamic Request Queue)  |   |
|   +------------+-------------+     +------------+-------------+   |
|                |                                |                 |
+----------------|--------------------------------|-----------------+
                 v                                v
    +---------------------------------------------------+
    |                 ArenaAllocator                    |
    |                                                   |
    |  +---------------------------------------------+  |
    |  | Block 0: 64-byte aligned contiguous memory  |  |
    |  | [offset ------>                      size]  |  |
    |  +---------------------------------------------+  |
    |                        | next                     |
    |  +---------------------v-----------------------+  |
    |  | Block 1: 64-byte aligned overflow block     |  |
    |  +---------------------------------------------+  |
    +---------------------------------------------------+
```

### Key Principles
1. **Strict 64-Byte Alignment**: Every buffer allocated begins on a 64-byte boundary, guaranteeing optimal memory access for AVX-512, ARM NEON, and Apple Silicon unified cache lines without unaligned vector penalty.
2. **$O(1)$ Bump Allocation**: Servicing an allocation request is a monotonic pointer bump of 5 CPU instructions ($< 2$ nanoseconds).
3. **$O(1)$ Block-Retaining Resets**: Calling `reset()` rewinds the bump pointer to zero without freeing memory back to the OS. Subsequent inference requests reuse warmed-up memory blocks with zero system call overhead.
4. **Thread-Local Isolation**: Each worker thread accesses its own persistent arena (`MemoryPool::getThreadLocalArena()`), completely eliminating mutex locks and cache line bouncing across CPU cores.
5. **C++20 PMR Compliance**: `ArenaAllocator` inherits from `std::pmr::memory_resource`, permitting seamless use with polymorphic containers (`std::pmr::vector`, `std::pmr::string`).

---

## 2. 64-Byte Alignment & SIMD Vectorization

Modern CPU architectures feature vector registers and cache lines operating on 64-byte boundaries:
- **AVX-512 (x86-64)**: 512-bit registers (64 bytes). Loading from unaligned memory crosses cache line boundaries, causing split-load penalties and hardware stalls.
- **ARM NEON (Apple Silicon)**: 128-bit vector execution with 64-byte cache line granularity.
- **Unified Memory & Metal Hand-off**: Unified CPU/GPU buffers require alignment to avoid memory re-copying.

In `ArenaAllocator`, memory alignment is enforced algebraically:
```cpp
uintptr_t curr = reinterpret_cast<uintptr_t>(current_->data + current_->offset);
uintptr_t aligned = (curr + (alignment - 1)) & ~(static_cast<uintptr_t>(alignment) - 1);
```
All allocated pointers satisfy:
$$\text{reinterpret\_cast<uintptr\_t>}(ptr) \pmod{64} == 0$$

---

## 3. Ownership & Lifetime Semantics

Pyreth enforces a clear, collective lifetime model for arena-backed buffers:

### 3.1 Bulk Deallocation
Individual `deallocate(void* ptr, ...)` calls on `ArenaAllocator` are no-ops. Memory is never fragmented by random individual frees. Instead, all memory allocated within a request is reclaimed simultaneously in $O(1)$ time when the request concludes.

### 3.2 Request Lifetime Contract
```cpp
{
  pyreth::ScopedArena request_arena;

  // Allocate intermediate layer activations
  float* input_tensor  = request_arena.allocateArray<float>(1024);
  float* hidden_states = request_arena.allocateArray<float>(4096);

  // Execute inference pipeline...
} 
// End of scope: ScopedArena destructor invokes arena.reset() in O(1).
// No individual delete/free calls executed. Backing memory retained for next request.
```

> [!IMPORTANT]
> **Lifetime Contract**: Pointers returned by an arena remain valid until `reset()` is invoked or the arena is destroyed. Tensors and inference handles referencing arena memory must not outlive the scope of the request arena lease.

---

## 4. API Reference

### `ArenaAllocator`
```cpp
#include <pyreth/memory_pool.hpp>

pyreth::ArenaAllocator arena(1024 * 1024); // 1 MB initial block

// Raw aligned allocation
void* buf = arena.allocate(256, 64);

// Exception-free monadic allocation
pyreth::Result<void*> res = arena.tryAllocate(1024);

// Typed object in-place construction
struct LayerConfig { int dim; float dropout; };
LayerConfig* cfg = arena.create<LayerConfig>(512, 0.1f);

// Typed array allocation
float* weights = arena.allocateArray<float>(2048);

// O(1) Reset
arena.reset();
```

### `MemoryPool` (Thread-Local & Leased Arenas)
```cpp
// 1. Thread-Local Scratchpad (Lock-Free)
auto& tl_arena = pyreth::MemoryPool::getThreadLocalArena();
tl_arena.reset();
void* scratch = tl_arena.allocate(4096);

// 2. Centralized Pool Lease (RAII)
pyreth::MemoryPool pool(/*initial_arenas=*/4, /*block_size=*/1024 * 1024);
{
  auto lease = pool.acquire();
  void* tensor_buf = lease->allocate(65536);
  // Do work...
} // lease returns arena to pool and resets it automatically upon destruction
```

### `ScopedArena`
```cpp
// Binds to thread-local arena by default
pyreth::ScopedArena request_arena;
void* mem = request_arena.allocate(1024);
// Automatically reset in O(1) on destruction
```

---

## 5. Performance Benchmark Results

Benchmarking 500,000 allocations of 128 bytes on Apple Silicon (M-series, 64-byte cache lines):

| Allocator | Total Time (500k allocs) | Throughput | Notes |
| :--- | :--- | :--- | :--- |
| **`pyreth::ArenaAllocator`** | **4.2 ms** | **119,000,000 allocs/sec** | 64-byte aligned, zero lock contention |
| `std::malloc` / `std::free` | 21.8 ms | 22,900,000 allocs/sec | System heap overhead |
| `std::pmr::new_delete_resource`| 23.4 ms | 21,300,000 allocs/sec | Standard heap polymorphic wrapper |

**Speedup**: `ArenaAllocator` achieves **$\approx 5.2\times$ higher throughput** than `std::malloc` while strictly guaranteeing 64-byte alignment across all buffers.
