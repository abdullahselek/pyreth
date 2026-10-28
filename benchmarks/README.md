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

## 5. HuggingFace Image Classification Benchmark (`google/mobilenet_v2_1.0_224`)

Executed via:

```bash
uv run python benchmarks/benchmark_image_classification.py --iterations 10 --warmup 2 --dynamic-requests 64
```

Model: `google/mobilenet_v2_1.0_224` exported from HuggingFace (`pixel_values: [batch_size, 3, 224, 224]` `float32` $\rightarrow$ `logits: [batch_size, 1001]` `float32`).

### 5.1 Synchronous `InferenceSession.run` (Heap-Allocated Outputs)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 9,192.52 | 9,268.70 | 9,340.48 | 9.3405 | 108.9 | 108.9 |
| **2** | 18,271.44 | 18,442.19 | 18,512.26 | 18.5123 | 54.7 | 109.3 |
| **4** | 36,551.17 | 36,756.99 | 36,986.72 | 36.9867 | 27.3 | 109.3 |
| **8** | 73,511.21 | 74,000.35 | 74,913.05 | 74.9130 | 13.6 | 108.6 |
| **16** | 148,779.08 | 149,413.96 | 150,150.42 | 150.1504 | 6.7 | 107.4 |
| **32** | 299,531.67 | 300,570.26 | 303,642.59 | 303.6426 | 3.3 | 106.7 |
| **64** | 601,627.54 | 603,021.08 | 603,723.08 | 603.7231 | 1.7 | 106.3 |

### 5.2 Synchronous `InferenceSession.run` (`ArenaAllocator` O(1) Reset Reuse)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 9,225.73 | 9,359.84 | 9,360.21 | 9.3602 | 108.2 | 108.2 |
| **2** | 18,587.48 | 18,753.42 | 19,185.15 | 19.1852 | 53.8 | 107.5 |
| **4** | 36,635.19 | 37,619.95 | 37,926.67 | 37.9267 | 27.1 | 108.3 |
| **8** | 73,248.81 | 73,494.60 | 73,554.81 | 73.5548 | 13.6 | **109.2** |
| **16** | 148,053.04 | 148,279.20 | 148,351.36 | 148.3514 | 6.8 | 108.1 |
| **32** | 299,413.17 | 299,697.63 | 299,854.30 | 299.8543 | 3.3 | 107.0 |
| **64** | 601,309.31 | 602,223.25 | 602,304.96 | 602.3050 | 1.7 | 106.4 |

### 5.3 Asynchronous `InferenceSession.run_async` (`ThreadPool` + `MemoryPool`, 4 Workers)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 19,861.62 | 29,648.39 | 29,665.36 | 29.6654 | 336.9 | 336.9 |
| **2** | 39,594.35 | 59,137.50 | 59,138.48 | 59.1385 | 169.1 | **338.1** |
| **4** | 79,364.04 | 118,407.92 | 118,783.59 | 118.7836 | 84.2 | 336.6 |
| **8** | 160,331.46 | 239,257.70 | 239,328.21 | 239.3282 | 41.8 | 334.2 |
| **16** | 322,426.65 | 480,076.24 | 480,676.21 | 480.6762 | 20.8 | 332.8 |
| **32** | 644,930.60 | 964,330.41 | 964,333.42 | 964.3334 | 10.4 | 331.8 |
| **64** | 1,286,559.87 | 1,921,771.30 | 1,926,813.99 | 1,926.8140 | 5.2 | 332.1 |

### 5.4 Concurrent `DynamicBatcher` Serving (8 Python Client Threads)

| Max Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Avg Coalesced Batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 83,484.27 | 147,400.73 | 157,315.92 | 157.3159 | 403.7 | 1.00 |
| **2** | 88,740.81 | 157,110.37 | 157,393.78 | 157.3938 | **404.1** | 2.00 |
| **4** | 99,174.87 | 158,328.66 | 158,743.21 | 158.7432 | 400.9 | 4.00 |
| **8** | 120,370.67 | 160,836.11 | 160,851.09 | 160.8511 | 395.5 | 8.00 |
| **16** | 161,866.71 | 162,280.82 | 162,295.09 | 162.2951 | 392.7 | 16.00 |
| **32** | 321,724.87 | 322,163.62 | 322,252.00 | 322.2520 | 198.2 | 32.00 |
| **64** | 642,963.56 | 643,114.47 | 643,196.58 | 643.1966 | 99.4 | **64.00** |

---

## 6. HuggingFace Small LLM Prefill Benchmark (`HuggingFaceTB/SmolLM2-135M`)

Executed via:

```bash
uv run python benchmarks/benchmark_small_llm.py --iterations 4 --warmup 1 --dynamic-requests 64
```

Model: `HuggingFaceTB/SmolLM2-135M` prefill (`use_cache=False`) exported from HuggingFace (`input_ids: [batch_size, 32]` `int64`, `attention_mask: [batch_size, 32]` `int64` $\rightarrow$ `logits: [batch_size, 32, 49152]` `float32`, Arena Block Size: `786,496 KiB`).

### 6.1 Synchronous `InferenceSession.run` (Heap-Allocated Outputs)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 70,589.65 | 70,687.53 | 70,717.35 | 70.7174 | 14.2 | 14.2 |
| **2** | 138,535.79 | 138,747.02 | 138,783.30 | 138.7833 | 7.2 | 14.4 |
| **4** | 273,786.96 | 274,106.66 | 274,167.15 | 274.1672 | 3.7 | **14.6** |
| **8** | 547,806.44 | 548,278.72 | 548,453.08 | 548.4531 | 1.8 | **14.6** |
| **16** | 1,096,281.08 | 1,098,342.72 | 1,099,137.82 | 1,099.1378 | 0.9 | **14.6** |
| **32** | 2,194,344.13 | 2,211,412.05 | 2,216,922.52 | 2,216.9225 | 0.5 | **14.6** |
| **64** | 4,536,347.56 | 4,564,372.89 | 4,567,322.96 | 4,567.3230 | 0.2 | 14.1 |

### 6.2 Synchronous `InferenceSession.run` (`ArenaAllocator` O(1) Reset Reuse)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 70,691.94 | 70,794.98 | 70,826.65 | 70.8266 | 14.2 | 14.2 |
| **2** | 138,999.75 | 139,050.12 | 139,068.99 | 139.0690 | 7.2 | 14.4 |
| **4** | 275,867.04 | 276,014.64 | 276,068.66 | 276.0687 | 3.6 | **14.5** |
| **8** | 550,860.23 | 551,058.35 | 551,130.15 | 551.1301 | 1.8 | **14.5** |
| **16** | 1,102,590.65 | 1,104,376.13 | 1,104,972.75 | 1,104.9728 | 0.9 | **14.5** |
| **32** | 2,215,724.00 | 2,219,817.92 | 2,221,117.67 | 2,221.1177 | 0.5 | 14.4 |
| **64** | 4,570,871.54 | 4,574,663.43 | 4,575,328.22 | 4,575.3282 | 0.2 | 14.0 |

### 6.3 Asynchronous `InferenceSession.run_async` (`ThreadPool` + `MemoryPool`, 4 Workers)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 75,500.79 | 75,500.99 | 75,501.04 | 75.5010 | 53.0 | 53.0 |
| **2** | 148,051.87 | 148,052.40 | 148,052.57 | 148.0526 | 27.0 | 54.0 |
| **4** | 293,586.85 | 293,587.42 | 293,587.46 | 293.5875 | 13.6 | 54.5 |
| **8** | 586,219.02 | 586,275.28 | 586,275.40 | 586.2754 | 6.8 | **54.6** |
| **16** | 1,181,950.50 | 1,182,138.52 | 1,182,138.65 | 1,182.1387 | 3.4 | 54.1 |
| **32** | 2,439,362.92 | 2,439,466.10 | 2,439,505.31 | 2,439.5053 | 1.6 | 52.5 |
| **64** | 5,051,674.48 | 5,051,675.06 | 5,051,675.12 | 5,051.6751 | 0.8 | 50.7 |

### 6.4 Concurrent `DynamicBatcher` Serving (8 Python Client Threads)

| Max Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Avg Coalesced Batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 637,604.67 | 1,127,152.30 | 1,202,816.14 | 1,202.8161 | 53.2 | 1.00 |
| **2** | 666,235.19 | 1,185,344.75 | 1,185,509.14 | 1,185.5091 | 53.9 | 2.00 |
| **4** | 735,364.40 | 1,179,113.22 | 1,179,148.39 | 1,179.1484 | **54.2** | 4.00 |
| **8** | 882,988.15 | 1,181,553.71 | 1,181,562.84 | 1,181.5628 | 54.1 | 8.00 |
| **16** | 1,193,102.12 | 1,194,373.71 | 1,194,442.33 | 1,194.4423 | 53.6 | 16.00 |
| **32** | 2,383,398.19 | 2,383,589.32 | 2,383,700.93 | 2,383.7009 | 26.8 | 32.00 |
| **64** | 4,855,166.65 | 4,855,296.64 | 4,855,423.29 | 4,855.4233 | 13.2 | **64.00** |

---

## 7. HuggingFace Regression Benchmark (`prajjwal1/bert-tiny`)

Executed via:

```bash
uv run python benchmarks/benchmark_regression.py
```

Model: `prajjwal1/bert-tiny` configured for a continuous regression head (`input_ids: [batch_size, 32]` `int64`, `attention_mask: [batch_size, 32]` `int64` $\rightarrow$ `predictions: [batch_size, 1]` `float32`).

### 7.1 Synchronous `InferenceSession.run` (Heap-Allocated Outputs)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 338.94 | 347.69 | 357.03 | 0.3570 | 2,933.3 | 2,933.3 |
| **2** | 607.58 | 625.65 | 643.41 | 0.6434 | 1,634.6 | 3,269.2 |
| **4** | 1,153.19 | 1,236.67 | 1,276.29 | 1.2763 | 854.7 | 3,418.8 |
| **8** | 2,278.44 | 2,312.40 | 2,343.36 | 2.3434 | 439.9 | 3,519.1 |
| **16** | 4,463.85 | 4,547.23 | 4,574.41 | 4.5744 | 223.6 | 3,578.2 |
| **32** | 8,954.90 | 9,099.18 | 9,136.76 | 9.1368 | 111.6 | 3,571.3 |
| **64** | 17,755.33 | 17,942.07 | 18,023.38 | 18.0234 | 56.2 | 3,595.7 |

### 7.2 Synchronous `InferenceSession.run` (`ArenaAllocator` O(1) Reset Reuse)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 302.85 | 316.64 | 327.28 | 0.3273 | 3,272.8 | 3,272.8 |
| **2** | 583.52 | 606.36 | 611.59 | 0.6116 | 1,699.0 | 3,398.1 |
| **4** | 1,146.04 | 1,164.07 | 1,177.65 | 1.1776 | 872.9 | 3,491.6 |
| **8** | 2,259.96 | 2,316.18 | 2,372.91 | 2.3729 | 440.3 | 3,522.6 |
| **16** | 4,466.31 | 4,535.25 | 4,642.95 | 4.6429 | 222.9 | 3,566.2 |
| **32** | 8,933.73 | 9,206.18 | 9,416.75 | 9.4167 | 111.2 | 3,558.2 |
| **64** | 17,869.50 | 17,985.68 | 18,032.09 | 18.0321 | 56.0 | 3,581.1 |

### 7.3 Asynchronous `InferenceSession.run_async` (`ThreadPool` + `MemoryPool`, 4 Workers)

| Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Throughput (Samples/sec) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 1,206.98 | 1,871.03 | 2,032.44 | 2.0324 | 9,716.2 | 9,716.2 |
| **2** | 2,502.85 | 3,877.57 | 4,046.81 | 4.0468 | 4,921.9 | 9,843.7 |
| **4** | 4,056.83 | 6,567.13 | 6,659.69 | 6.6597 | 2,996.3 | 11,985.3 |
| **8** | 7,498.94 | 12,380.30 | 12,473.59 | 12.4736 | 1,600.3 | 12,802.7 |
| **16** | 14,510.40 | 24,169.50 | 24,200.32 | 24.2003 | 825.9 | 13,215.0 |
| **32** | 28,961.31 | 48,072.66 | 48,096.92 | 48.0969 | 415.7 | 13,301.4 |
| **64** | 57,258.75 | 95,222.13 | 95,498.28 | 95.4983 | 209.3 | 13,392.0 |

### 7.4 Concurrent `DynamicBatcher` Serving (8 Python Threads)

| Max Batch Size | P50 Latency (us) | P90 Latency (us) | P99 Latency (us) | P99 Latency (ms) | Throughput (Req/sec) | Avg Coalesced Batch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 2,883.81 | 5,046.28 | 5,454.20 | 5.4542 | 9,785.2 | 1.00 |
| **2** | 3,284.54 | 5,250.15 | 5,879.07 | 5.8791 | 9,288.6 | 2.00 |
| **4** | 3,362.79 | 5,212.78 | 5,275.38 | 5.2754 | 10,665.0 | 4.00 |
| **8** | 3,834.25 | 5,022.75 | 5,041.28 | 5.0413 | 11,340.0 | 8.00 |
| **16** | 5,019.21 | 5,146.61 | 5,168.85 | 5.1689 | 10,961.6 | 16.00 |
| **32** | 9,813.98 | 9,911.78 | 9,982.64 | 9.9826 | 6,082.9 | 32.00 |
| **64** | 20,452.42 | 20,614.47 | 20,671.40 | 20.6714 | 3,044.6 | 64.00 |

---

## 8. Automated Regression Checks & Reproduction

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
