# Parallelism & Concurrency Architecture

Pyreth is engineered for high-throughput, deterministic low-latency AI model inference. To achieve this on modern multicore CPUs (including Apple Silicon ARM64 and x86-64 server architectures), Pyreth avoids central lock contention on the scheduling hot path by using **lock-free work-stealing deques** paired with **cooperative worker parking primitives**.

This document outlines the design, invariants, and formal memory ordering proofs for the Phase 2 concurrency subsystem.

---

## 1. Concurrency Model Overview

```
                      +-----------------------------+
                      |  Inference Request Dispatch |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Pyreth Worker Threads   |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-------------------+                                   +-------------------+
|  Worker Thread 0  |                                   |  Worker Thread 1  |
|  +-------------+  |                                   |  +-------------+  |
|  | Local WSQ   |  |   Steal oldest tasks (FIFO)       |  | Local WSQ   |  |
|  |  (LIFO pop) |  |<==================================|  |  (LIFO pop) |  |
|  +-------------+  |==================================>|  +-------------+  |
|  | Parker/Sem  |  |   Steal oldest tasks (FIFO)       |  | Parker/Sem  |  |
|  +-------------+  |                                   |  +-------------+  |
+-------------------+                                   +-------------------+
```

### Key Principles
1. **Per-Thread Local Work Queues**: Each worker thread owns a private work-stealing queue (`WorkStealingQueue`). Worker pushes and pops occur locally without acquiring mutexes or competing on global lock queues.
2. **LIFO Locality for Producers**: The owning worker pushes and pops from the `bottom` of its queue in LIFO order, ensuring that hot cache lines (recently produced tasks, tensor buffers, intermediate weights) remain in L1/L2 cache.
3. **FIFO Stealing for Consumers**: Idle workers (thieves) steal tasks from the `top` of peer queues in FIFO order. These oldest tasks typically represent the roots of decomposed workloads, minimizing future stealing overhead.
4. **Cooperative Worker Parking**: When a worker exhausts its own queue and fails all stealing attempts, it parks via `WorkerParker` rather than busy-spinning at 100% CPU.

---

## 2. Bounded Chase-Lev Deque (`WorkStealingQueue`)

The `WorkStealingQueue<T, Capacity>` is a lock-free Single-Producer Multi-Consumer (SPMC) work-stealing deque based on the Chase-Lev algorithm, updated for the C++20 memory model following the formal proofs by **Lê et al. (PPoPP 2013)**.

### Class Layout & Cache Line Alignment
To eliminate false sharing across CPU cores:
- `top_` (frequently written by thieves and owner) is placed on its own 64-byte cache line via `alignas(64)`.
- `bottom_` (frequently updated by the owner thread) is placed on a separate 64-byte cache line via `alignas(64)`.
- The backing circular slot buffer is pre-allocated as a power-of-two contiguous array.

```cpp
template <typename T, size_t Capacity = 1024>
class WorkStealingQueue {
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

  struct Slot {
    std::atomic<bool> ready{false};
    std::optional<T> item{std::nullopt};
  };

  alignas(64) std::atomic<int64_t> top_{0};
  alignas(64) std::atomic<int64_t> bottom_{0};
  alignas(64) std::unique_ptr<Slot[]> buffer_;
};
```

---

## 3. Formal Memory Ordering Proofs (Lê et al., PPoPP 2013)

The classical Chase-Lev deque relies on strong sequential consistency on x86-64 (`mfence`). On weakly ordered memory architectures like ARM64 (Apple Silicon), explicit atomic memory orders and sequential consistency fences are required to prevent data races and missed updates.

### 3.1 Push Operation (Owner Thread)

```cpp
bool push(T&& item) {
  int64_t b = bottom_.load(std::memory_order_relaxed);
  int64_t t = top_.load(std::memory_order_acquire);

  if (b - t >= static_cast<int64_t>(Capacity)) {
    return false; // Queue full
  }

  Slot& slot = buffer_[b & MASK];
  if (slot.ready.load(std::memory_order_acquire)) {
    return false; // Wraparound protection
  }

  slot.item.emplace(std::move(item));
  slot.ready.store(true, std::memory_order_release);
  bottom_.store(b + 1, std::memory_order_release);
  return true;
}
```

* **Invariant 1 (`top_.load(acquire)`)**:
  Pairs with the `top_.compare_exchange_strong(..., seq_cst, ...)` in `steal()` and `pop()`. This ensures the owner observes all stealer advances of `top_` before computing the remaining capacity `b - t`.
* **Invariant 2 (`bottom_.store(b + 1, release)`)**:
  Ensures that the initialization of `slot.item` and `slot.ready = true` is committed to memory before `bottom_` is incremented. Any thief that subsequently loads `bottom_` with `acquire` is guaranteed to observe the fully constructed task payload.

---

### 3.2 Pop Operation (Owner Thread, LIFO)

```cpp
std::optional<T> pop() {
  int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
  bottom_.store(b, std::memory_order_relaxed);

  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t t = top_.load(std::memory_order_relaxed);

  if (t <= b) {
    if (t == b) {
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return std::nullopt;
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
    }
    Slot& slot = buffer_[b & MASK];
    T val = std::move(*slot.item);
    slot.item.reset();
    slot.ready.store(false, std::memory_order_release);
    return val;
  } else {
    bottom_.store(b + 1, std::memory_order_relaxed);
    return std::nullopt;
  }
}
```

#### The Store-Load Barrier Proof
The most critical race condition occurs when exactly one task remains in the queue ($top = 0, bottom = 1$).
- The owner wants to pop the task at index 0.
- A concurrent stealer wants to steal the task at index 0.

In Dekker-style synchronization, the owner writes to `bottom_` and reads `top_`, while the stealer reads `top_` and reads `bottom_`.
On weakly ordered architectures (ARM64), a processor may reorder a store followed by a load (Store-Load reordering).
- **Without the `seq_cst` fence**: The owner's store `bottom_ = b` could be delayed past its load of `top_`. Concurrently, the stealer could read an old value of `bottom_`. Both threads might believe they exclusively own the item without synchronizing.
- **With `std::atomic_thread_fence(memory_order_seq_cst)`**: The C++20 memory model enforces a total order across all `seq_cst` operations:
  $$\text{Store}(bottom_) \prec \text{Fence}_{\text{seq\_cst}} \prec \text{Load}(top_)$$
  Either the owner observes `top_` already advanced by the stealer, or the stealer observes `bottom_` decremented by the owner.

#### Single-Element Race Resolution
When $t == b$, exactly one element remains. Both owner and stealer attempt an atomic CAS:
`top_.compare_exchange_strong(t, t + 1, seq_cst, ...)`
- **Owner wins**: Increments `top_` from $t$ to $t + 1$. The stealer's CAS fails. The owner moves the item and restores `bottom_ = b + 1`. Both indices are now $t + 1$, queue is empty.
- **Stealer wins**: The owner's CAS fails. The owner restores `bottom_ = b + 1` and returns `std::nullopt`. The stealer successfully retrieves the item.

#### Multi-Element Fast Path ($t < b$)
When $t < b$, at least two elements exist. Since thieves only steal at index $t$ and $t < b$, no thief touches index $b$. The owner extracts `buffer_[b & MASK]` without performing any CAS, achieving zero atomic contention.

---

### 3.3 Steal Operation (Thief Threads, FIFO)

```cpp
std::optional<T> steal() {
  int64_t t = top_.load(std::memory_order_acquire);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  int64_t b = bottom_.load(std::memory_order_acquire);

  if (t < b) {
    Slot& slot = buffer_[t & MASK];
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                      std::memory_order_relaxed)) {
      return std::nullopt; // Lost race to another thief or owner pop
    }

    while (!slot.ready.load(std::memory_order_acquire)) {
      yield_cpu();
    }

    T val = std::move(*slot.item);
    slot.item.reset();
    slot.ready.store(false, std::memory_order_release);
    return val;
  }
  return std::nullopt;
}
```

* **CAS-First Protocol**:
  Unlike classic Chase-Lev where the item pointer is speculatively read before the CAS, Pyreth executes the CAS **first**. Only the winning thief accesses `slot.item`.
* **Zero Race on Move-Only Types**:
  Because only the winning thread accesses `slot.item`, `T` can be a move-only type (`std::unique_ptr<Task>`, `std::move_only_function`). Losing thieves never touch the payload.
* **Wraparound Safety**:
  After extracting the item, the winner sets `slot.ready.store(false, release)`. If the owner pushes $Capacity$ items before a delayed thief finishes moving, the owner's `push()` observes `slot.ready == true` and safely rejects the push rather than corrupting memory.

---

## 4. Worker Parking Primitives (`WorkerParker` & `AtomicNotification`)

Busy-spinning in empty work-stealing loops burns CPU power and degrades thermal headroom for inference execution. Pyreth introduces two notification primitives:

### 4.1 `WorkerParker`
Each worker thread maintains a `WorkerParker` instance implementing an idempotent 3-state token automaton:

```
         +-------------------+
         |       EMPTY       |
         +-------------------+
           /               ^
   unpark /                 \ park (consumes permit)
         v                   \
  +-------------+       +-------------+
  |   NOTIFIED  |       |   PARKED    |
  +-------------+       +-------------+
         |                     ^
         |                     | park (blocks in semaphore)
         +---------------------+
```

1. **Lost Wakeup Protection**:
   If an external submitter calls `unpark()` before an idle worker calls `park()`, `state_` transitions from `EMPTY` to `NOTIFIED`. When the worker calls `park()`, it executes:
   ```cpp
   if (state_.exchange(EMPTY, std::memory_order_acquire) == NOTIFIED) return;
   ```
   The token is consumed and `park()` returns immediately without blocking.
2. **Idempotent Multi-Notification**:
   Multiple concurrent `unpark()` calls are consolidated into the single `NOTIFIED` state. `std::binary_semaphore::release()` is called at most once, preventing semaphore capacity overflow.
3. **Timed Waiting**:
   `parkFor(duration)` allows idle workers to sleep with configurable timeouts, periodically waking to probe global queues or verify shutdown signals.

### 4.2 `AtomicNotification`
Leverages C++20 `std::atomic<uint32_t>::wait` and `notify_one()` / `notify_all()` for broadcast coordination across thread pool lifecycles.

---

## 5. Summary of Concurrency Invariants

| Operation | Atomic Read Order | Atomic Write Order | Fence | Primary Synchronization |
| :--- | :--- | :--- | :--- | :--- |
| `push()` | `top_` (acquire) | `bottom_` (release) | None | In-order task visibility before bottom bump |
| `pop()` (fast path) | `bottom_` (relaxed), `top_` (relaxed) | `bottom_` (relaxed) | `seq_cst` | Dekker barrier prevents single-element race |
| `pop()` (single item)| `top_` (relaxed) | `top_` (CAS seq_cst) | `seq_cst` | CAS on `top_` arbitrates owner vs stealer |
| `steal()` | `top_` (acquire), `bottom_` (acquire)| `top_` (CAS seq_cst) | `seq_cst` | CAS on `top_` arbitrates multiple thieves |
| `park()` / `unpark()`| `state_` (acquire) | `state_` (release) | None | Binary semaphore blocks only when state is PARKED |
