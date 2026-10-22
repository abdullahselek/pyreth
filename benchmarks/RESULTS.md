# Pyreth Performance Benchmark Results & Regression Suite

This document publishes the automated micro-benchmark and end-to-end serving benchmark results for **Pyreth**, along with the regression thresholds enforced by the benchmark suite.

---

## 1. Benchmark Architecture & Pipeline

```
+-----------------------------------------------------------------------------------+
|                        PYRETH BENCHMARKING ARCHITECTURE                           |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [A] C++ Google Benchmark Suite (./build/benchmarks/pyreth_benchmarks)            |
|                                                                                   |
|      +---------------------------+     +---------------------------------------+  |
|      | Memory Subsystem          |     | Lock-Free Concurrency                 |  |
|      | - ArenaAllocator (2.48ns) |     | - WorkStealingQueue LIFO (4.52ns)     |  |
|      | - posix_memalign (10-73ns)|     | - WorkStealingQueue FIFO (4.69ns)     |  |
|      | - MemoryPool Lease (9.9ns)|     | - Burst 256 Push/Pop (288M items/sec) |  |
|      +---------------------------+     +---------------------------------------+  |
|                    |                                       |                      |
|                    +-------------------+-------------------+                      |
|                                        |                                          |
|                                        v                                          |
|      +-------------------------------------------------------------------------+  |
|      | Zero-Copy Tensor & C++ InferenceSession Execution                       |  |
|      | - Tensor::fromBuffer Zero-Copy (36.8ns) vs Heap Creation (51.9ns)       |  |
|      | - In-Place Run (569ns) vs Arena Run (687ns) vs Heap Run (768ns)         |  |
|      +-------------------------------------------------------------------------+  |
|                                                                                   |
|  [B] Python End-to-End Serving Suite (uv run python benchmarks/benchmark_serving.py)
|                                                                                   |
|      +--------------------+   GIL Released    +--------------------------------+  |
|      | Batch Sizes 1..64  | ----------------> | 1A. Sync Session (Heap)        |  |
|      | (1,2,4,8,16,32,64) |                   | 1B. Sync Session (Arena Reuse) |  |
|      +--------------------+                   | 2.  Async ThreadPool+MemoryPool|  |
|                                               | 3.  Concurrent DynamicBatcher  |  |
|                                               +--------------------------------+  |
|                                                                |                  |
|                                                                v                  |
|                                               +--------------------------------+  |
|                                               | Automated Regression Verifier  |  |
|                                               | - P50 <= P90 <= P99 < 100ms    |  |
|                                               | - B64 vs B1 Scaling >= 5.0x    |  |
|                                               | - DynamicBatcher Coalescing    |  |
|                                               +--------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Test Environment

| Property | Value |
| :--- | :--- |
| **Platform** | macOS 26.4 (`arm64`, Apple Silicon, 18 CPU cores) |
| **L1 / L2 Cache** | L1d: 64 KiB, L1i: 128 KiB, L2 Unified: 8192 KiB (x18) |
| **Compiler** | AppleClang 21.0 (`-O3 -DNDEBUG -std=c++20`) |
| **ONNX Runtime** | v1.19.2 (`arm64`) |
| **Python Runtime** | CPython 3.14.8 + `nanobind` + `numpy` 2.x |
| **Google Benchmark** | v1.9.1 |

---

## 3. C++ Micro-Benchmarks (`Google Benchmark`)

Executed via:

```bash
./build/benchmarks/pyreth_benchmarks --benchmark_min_time=0.05s
```

### 3.1 Memory Allocation: `ArenaAllocator` vs `posix_memalign` + `free`

| Benchmark | Allocation Size | Time (ns) | CPU (ns) | Throughput (Items/sec) | Bandwidth | Speedup vs `malloc` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `BM_MallocFree_64B_Aligned_Single/256` | 256 B | 10.90 ns | 10.90 ns | 91.73M/s | 21.87 GiB/s | 1.00x |
| `BM_ArenaAllocate_64B_Aligned_Single/256` | 256 B | **2.49 ns** | **2.49 ns** | **401.70M/s** | **95.77 GiB/s** | **4.38x** |
| `BM_MallocFree_64B_Aligned_Single/1024` | 1 KiB | 18.60 ns | 18.60 ns | 53.85M/s | 51.35 GiB/s | 1.00x |
| `BM_ArenaAllocate_64B_Aligned_Single/1024` | 1 KiB | **2.48 ns** | **2.48 ns** | **403.73M/s** | **385.03 GiB/s** | **7.50x** |
| `BM_MallocFree_64B_Aligned_Single/4096` | 4 KiB | 13.50 ns | 13.50 ns | 74.23M/s | 283.17 GiB/s | 1.00x |
| `BM_ArenaAllocate_64B_Aligned_Single/4096` | 4 KiB | **2.47 ns** | **2.47 ns** | **404.54M/s** | **1.51 TiB/s** | **5.47x** |
| `BM_MallocFree_64B_Aligned_Single/16384` | 16 KiB | 17.80 ns | 17.80 ns | 56.16M/s | 856.93 GiB/s | 1.00x |
| `BM_ArenaAllocate_64B_Aligned_Single/16384` | 16 KiB | **2.48 ns** | **2.48 ns** | **403.16M/s** | **6.01 TiB/s** | **7.18x** |
| `BM_MallocFree_64B_Aligned_Single/65536` | 64 KiB | 72.90 ns | 72.90 ns | 13.72M/s | 837.55 GiB/s | 1.00x |
| `BM_ArenaAllocate_64B_Aligned_Single/65536` | 64 KiB | **2.48 ns** | **2.48 ns** | **403.07M/s** | **24.02 TiB/s** | **29.40x** |
| `BM_MallocFree_Batch64_Allocations` | 64 x 1 KiB | 1265.00 ns | 1264.00 ns | 50.64M/s | 48.30 GiB/s | 1.00x |
| `BM_ArenaAllocate_Batch64_Allocations` | 64 x 1 KiB | **157.00 ns** | **157.00 ns** | **408.22M/s** | **389.31 GiB/s** | **8.05x** |
| `BM_MemoryPool_AcquireAllocateRelease` | 1 KiB (Pooled) | **9.94 ns** | **9.93 ns** | **100.68M/s** | — | — |

### 3.2 Lock-Free Queue & ThreadPool Scheduling

| Benchmark | Operation | Time (ns) | CPU (ns) | Throughput (Items/sec) |
| :--- | :--- | ---: | ---: | ---: |
| `BM_WorkStealingQueue_PushPop_LIFO` | Owner thread `push` + `pop` (LIFO) | **4.52 ns** | **4.52 ns** | **221.34M/s** |
| `BM_WorkStealingQueue_PushSteal_FIFO` | Cross-end `push` + `steal` (FIFO) | **4.69 ns** | **4.69 ns** | **213.36M/s** |
| `BM_WorkStealingQueue_Burst256_PushPop` | 256-task burst `push` + `pop` | **889.00 ns** (3.47 ns/op) | **889.00 ns** | **288.04M/s** |
| `BM_ThreadPool_SubmitWait` | Round-trip `submit()` + `TaskHandle::get()` | **2452.00 ns** | **1389.00 ns** | **720.04k/s** |

### 3.3 Tensor Creation, Zero-Copy Wrapping & Views

| Benchmark | Storage / Operation | Time (ns) | CPU (ns) | Throughput (Items/sec) |
| :--- | :--- | ---: | ---: | ---: |
| `BM_TensorCreate_HeapOwned/1` | Heap-allocated `[1, 16]` Float32 | 48.30 ns | 48.30 ns | 20.72M/s |
| `BM_TensorCreate_HeapOwned/16` | Heap-allocated `[16, 16]` Float32 | 57.10 ns | 57.10 ns | 17.52M/s |
| `BM_TensorCreate_HeapOwned/64` | Heap-allocated `[64, 16]` Float32 | 51.90 ns | 51.90 ns | 19.27M/s |
| `BM_TensorCreate_ArenaBacked/1` | Arena-backed `[1, 16]` Float32 + `reset()` | **38.10 ns** | **38.10 ns** | **26.23M/s** |
| `BM_TensorCreate_ArenaBacked/16` | Arena-backed `[16, 16]` Float32 + `reset()` | **38.60 ns** | **38.50 ns** | **25.94M/s** |
| `BM_TensorCreate_ArenaBacked/64` | Arena-backed `[64, 16]` Float32 + `reset()` | **38.10 ns** | **38.00 ns** | **26.28M/s** |
| `BM_TensorFromBuffer_ZeroCopy` | `Tensor::fromBuffer` `[64, 16]` zero-copy | **36.80 ns** | **36.80 ns** | **27.20M/s** |
| `BM_TensorSliceAndReshape_ZeroCopy` | Strided `slice()` + `reshape()` view | **90.70 ns** | **90.60 ns** | **11.03M/s** |

### 3.4 C++ `InferenceSession::run` (Heap vs `ArenaAllocator` vs In-Place Outputs)

Model: `MatMul([B, 16] x [16, 8]) + Add([8]) -> ReLU([B, 8])`

| Benchmark | Batch Size | Time (ns) | CPU (ns) | Sample Throughput |
| :--- | ---: | ---: | ---: | ---: |
| `BM_InferenceSession_Run_HeapOutputs/1` | 1 | 768 ns | 768 ns | 1.30M samples/s |
| `BM_InferenceSession_Run_ArenaOutputs/1` | 1 | 688 ns | 687 ns | 1.46M samples/s |
| `BM_InferenceSession_Run_InPlaceOutputs/1` | 1 | **569 ns** | **569 ns** | **1.76M samples/s** |
| `BM_InferenceSession_Run_HeapOutputs/8` | 8 | 779 ns | 779 ns | 10.27M samples/s |
| `BM_InferenceSession_Run_ArenaOutputs/8` | 8 | 710 ns | 710 ns | 11.27M samples/s |
| `BM_InferenceSession_Run_InPlaceOutputs/8` | 8 | **598 ns** | **598 ns** | **13.38M samples/s** |
| `BM_InferenceSession_Run_HeapOutputs/32` | 32 | 913 ns | 912 ns | 35.08M samples/s |
| `BM_InferenceSession_Run_ArenaOutputs/32` | 32 | 841 ns | 840 ns | 38.09M samples/s |
| `BM_InferenceSession_Run_InPlaceOutputs/32` | 32 | **721 ns** | **721 ns** | **44.40M samples/s** |
| `BM_InferenceSession_Run_HeapOutputs/64` | 64 | 1069 ns | 1068 ns | 59.93M samples/s |
| `BM_InferenceSession_Run_ArenaOutputs/64` | 64 | 1027 ns | 1026 ns | 62.37M samples/s |
| `BM_InferenceSession_Run_InPlaceOutputs/64` | 64 | **895 ns** | **894 ns** | **71.57M samples/s** |

---

## 4. End-to-End Python Serving Benchmarks (Batch Sizes 1 to 64)

Executed via:

```bash
uv run python benchmarks/benchmark_serving.py
```

Model: 2-Layer MLP (`[B, 64] -> MatMul(64x128) -> Add -> ReLU -> MatMul(128x16) -> Add -> [B, 16]`, Float32).

### 4.1 Synchronous `InferenceSession.run` (Heap-Allocated Outputs)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 1.79 | 1.83 | 2.00 | 0.0020 | 536,971.9 | 536,971.9 |
| **2** | 1.83 | 1.87 | 1.92 | 0.0019 | 526,952.3 | 1,053,904.6 |
| **4** | 2.08 | 2.13 | 2.54 | 0.0025 | 467,199.1 | 1,868,796.5 |
| **8** | 2.75 | 2.79 | 2.88 | 0.0029 | 354,793.6 | 2,838,348.9 |
| **16** | 4.13 | 4.17 | 4.88 | 0.0049 | 235,605.9 | 3,769,695.2 |
| **32** | 6.87 | 6.92 | 7.04 | 0.0070 | 144,621.8 | 4,627,898.9 |
| **64** | 12.33 | 12.67 | 13.38 | 0.0134 | 80,155.0 | **5,129,917.2** |

### 4.2 Synchronous `InferenceSession.run` (`ArenaAllocator` O(1) Reset Reuse)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | **1.67** | **1.71** | **1.88** | **0.0019** | **576,922.5** | **576,922.5** |
| **2** | **1.75** | **1.79** | **1.84** | **0.0018** | **554,400.6** | **1,108,801.1** |
| **4** | **1.96** | **2.00** | **2.04** | **0.0020** | **500,625.8** | **2,002,503.1** |
| **8** | **2.62** | **2.71** | **2.97** | **0.0030** | **369,174.0** | **2,953,391.8** |
| **16** | **4.00** | **4.04** | **4.17** | **0.0042** | **247,486.5** | **3,959,783.5** |
| **32** | **6.71** | **6.75** | **6.79** | **0.0068** | **147,951.8** | **4,734,456.2** |
| **64** | **11.75** | **12.21** | **15.01** | **0.0150** | **83,556.7** | **5,347,631.6** |

### 4.3 Asynchronous `InferenceSession.run_async` (`ThreadPool` + `MemoryPool`, 4 Workers)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 178.29 | 292.18 | 317.86 | 0.3179 | 475,529.3 | 475,529.3 |
| **2** | 188.92 | 297.63 | 319.17 | 0.3192 | 498,235.0 | 996,470.0 |
| **4** | 155.33 | 264.38 | 286.10 | 0.2861 | 515,575.5 | 2,062,302.2 |
| **8** | 150.54 | 311.88 | 344.75 | 0.3447 | 454,330.7 | 3,634,645.4 |
| **16** | 208.25 | 357.31 | 392.79 | 0.3928 | 412,299.7 | 6,596,795.6 |
| **32** | 272.04 | 478.14 | 521.00 | 0.5210 | 339,438.1 | 10,862,020.1 |
| **64** | 395.94 | 753.62 | 821.84 | 0.8218 | 222,551.8 | **14,243,318.2** |

### 4.4 Concurrent `DynamicBatcher` Serving (8 Python Client Threads, Individual `B=1` Requests)

Each Python thread submits individual `B=1` requests concurrently while the C++ `DynamicBatcher` coalesces them into batched tensor runs (`max_batch_size` = 1..64) and demultiplexes outputs zero-copy back to each caller:

| Max Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Avg Coalesced Batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 441.12 | 925.42 | 1210.49 | 1.2105 | 58,199.4 | 1.00 |
| **2** | 362.73 | 834.67 | 1360.24 | 1.3602 | 64,607.7 | 2.00 |
| **4** | 381.75 | 743.23 | 1090.33 | 1.0903 | 69,244.6 | 4.00 |
| **8** | 321.79 | 595.42 | 977.12 | 0.9771 | 83,834.8 | 8.00 |
| **16** | 385.60 | 608.75 | 845.71 | 0.8457 | 85,011.0 | 16.00 |
| **32** | 524.38 | 1031.08 | 1219.80 | 1.2198 | 67,112.3 | 28.44 |
| **64** | 488.35 | 650.12 | **823.34** | **0.8233** | **86,791.9** | **64.00** |

---

## 5. Automated Regression Checks & Reproduction

Both benchmark suites are wired into automated testing to prevent performance or correctness regressions:

1. **C++ Google Benchmark Smoke Test (CTest)**:
   ```bash
   cmake -B build && cmake --build build --parallel
   ctest --test-dir build -R PyrethMicrobenchmarksSmoke --output-on-failure
   ```
2. **Python Serving Benchmark & Regression Verifier**:
   ```bash
   uv run python benchmarks/benchmark_serving.py --json-out build/benchmark_results.json
   ```
   The script automatically verifies:
   - All batch sizes `(1, 2, 4, 8, 16, 32, 64)` execute cleanly across all 4 execution modes.
   - Monotonic percentile ordering (`0 < P50 <= P90 <= P99`) and P99 latency ceiling (`< 100 ms`).
   - Sample throughput scaling (`Batch=64` achieves `>= 5.0x` higher `samples/sec` than `Batch=1`).
   - Multi-threaded GIL release and dynamic batch coalescing (`avg_coalesced_batch > 1.0` under concurrent load).
