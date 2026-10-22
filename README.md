```bash
██████╗ ██╗   ██╗██████╗ ███████╗████████╗██╗  ██╗
██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝╚══██╔══╝██║  ██║
██████╔╝ ╚████╔╝ ██████╔╝█████╗     ██║   ███████║
██╔═══╝   ╚██╔╝  ██╔══██╗██╔══╝     ██║   ██╔══██║
██║        ██║   ██║  ██║███████╗   ██║   ██║  ██║
╚═╝        ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝
```

**High throughput, low latency parallel processing AI model inference engine in C++20 and Python.**

---

## Overview

**Pyreth** is a production-grade AI inference runtime engineered to eliminate memory allocation bottlenecks, lock contention, and Python Global Interpreter Lock (GIL) serialization on the inference hot path. Built around a zero-copy C++20 core (`EngineCore`) with `nanobind` Python bindings, Pyreth pairs a lock-free work-stealing scheduler and 64-byte SIMD-aligned memory arenas with ONNX Runtime graph execution and dynamic request batching.

### Architecture

```
+-----------------------------------------------------------------------------------+
|                        CLIENT APPLICATIONS (C++20 / Python)                       |
|      np.ndarray <--(Zero-Copy DLPack/Buffer)--> pyreth::Tensor (64B Aligned)      |
+-----------------------------------------+-----------------------------------------+
                                          |
                          [nb::gil_scoped_release in Python]
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                                  ENGINE CORE                                      |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | DynamicBatcher (Latency-Bounded Request Coalescing & Zero-Copy Demux)       |  |
|  | - Coalesces requests up to max_batch_size or max_batch_delay_ms/us          |  |
|  | - Concatenates batch in pooled 64B Arena & slices outputs in O(1)           |  |
|  +--------------------------------------+--------------------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +-----------------------------------------------------------------------------+  |
|  | ThreadPool (Lock-Free Chase-Lev Work-Stealing Deques + WorkerParker)        |  |
|  | - Owner LIFO push/pop (4.5ns) | Thief FIFO steal (4.7ns) | Zero-spin park   |  |
|  +------------------+---------------------------------------+------------------+  |
|                     |                                       |                     |
|                     v                                       v                     |
|  +-------------------------------------+ +-------------------------------------+  |
|  | Memory Hierarchy                    | | InferenceSession (ONNX Runtime)     |  |
|  | - ArenaAllocator (2.5ns bump, O(1)) | | - Cached ModelInfo & TensorInfo     |  |
|  | - MemoryPool (RAII Lease recycling) | | - Zero-copy Ort::Value tensor bridge|  |
|  | - Atomic pin-count lifetime safety  | | - CPU / CoreML / CUDA / TensorRT    |  |
|  +-------------------------------------+ +-------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### Key Highlights

| Subsystem | Design | Measured Performance |
| :--- | :--- | :--- |
| **Memory Subsystem** | 64-byte SIMD-aligned [`ArenaAllocator`](docs/memory.md) & [`MemoryPool`](docs/memory.md) with atomic pin counting and $O(1)$ pointer-bump reset | **2.48 ns** per allocation (**up to 29.4x faster** than `posix_memalign`/`free`) |
| **Lock-Free Concurrency** | Bounded Chase-Lev [`WorkStealingQueue`](docs/parallelism.md), 3-state semaphore [`WorkerParker`](docs/parallelism.md), and [`ThreadPool`](docs/parallelism.md) | **4.52 ns** LIFO push/pop (**221M ops/s**); **288M ops/s** on 256-task bursts |
| **Zero-Copy Tensor & NumPy Bridge** | [`pyreth::Tensor`](docs/tensor.md) with `OWNED`, `ARENA`, `POOLED`, and `EXTERNAL` storage; bidirectional zero-copy [`nanobind::ndarray`](docs/python_api.md) bridge | **36.8 ns** zero-copy buffer wrap; **0 byte copies** across Python <-> C++ <-> ORT |
| **Inference & Dynamic Batching** | [`InferenceSession`](docs/inference.md) & [`DynamicBatcher`](docs/dynamic_batching.md) with full Python GIL release (`nb::gil_scoped_release`) | **576k req/s** (`1.67 us` P50) sync; **14.2M samples/s** async; **86.8k req/s** concurrent dynamic batching |

Full benchmark methodology and latency/throughput tables across batch sizes 1 to 64 are published in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

---

## Project Layout

```
├── benchmarks/       # C++ Google Benchmark suite & Python serving benchmarks
├── cmake/            # CMake discovery and ONNX Runtime provisioning modules
├── docs/             # Subsystem architecture & API guides
├── include/pyreth/   # Public C++20 API headers
├── python/           # nanobind C++ extension (pyreth_core.cpp) & pyreth package
├── sandbox/          # C++ development runner & integration smoke executable
├── src/core/         # Core C++ engine implementation
├── tests/python/     # Python pytest integration & concurrency test suite
├── unittest/         # C++ GoogleTest unit test suite
├── CMakeLists.txt    # Root CMake build configuration
└── pyproject.toml    # Python packaging configuration (uv / scikit-build-core)
```

---

## Installation Guide

### Prerequisites

- **CMake** >= 3.25
- **C++20 Compiler**:
  - **macOS (Apple Silicon)**: macOS 26+ with Xcode Command Line Tools (`AppleClang` >= 17)
  - **Linux (Ubuntu / Debian x86_64 & ARM64)**: `clang` >= 16 or `g++` >= 12, `build-essential`, `ninja-build`
  - **Windows (x64)**: Visual Studio 2022 (MSVC v143)
- **Python Ecosystem**: [uv](https://docs.astral.sh/uv/) (automatically manages Python >= 3.13, `nanobind`, `scikit-build-core`, and `numpy`)

> **Note**: Prebuilt ONNX Runtime v1.19.2 libraries, GoogleTest v1.15.2, and Google Benchmark v1.9.1 are fetched automatically by CMake during configuration—no manual system installation is required.

### 1. Python Installation (`uv` / `pip`)

#### Using `uv` (Recommended for Development & Serving)

```sh
# Clone the repository
git clone https://github.com/abdullahselek/pyreth.git
cd pyreth

# Create the virtual environment, compile the C++20 nanobind extension, and install dependencies
uv sync

# Verify installation
uv run python -c "import pyreth; print('Pyreth:', pyreth.__version__, '| ORT:', pyreth.get_runtime_version(), '| Best Provider:', pyreth.get_best_backend())"
```

#### Building a Self-Contained Python Wheel

```sh
uv build --wheel
uv pip install dist/pyreth-*.whl
```

### 2. C++ Installation & Building from Source

```sh
# 1. Configure Release build (fetches ONNX Runtime, GoogleTest, and Google Benchmark)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# 2. Build all targets in parallel (libEngineCore.a, Sandbox, pyreth_tests, pyreth_benchmarks)
cmake --build build --parallel

# 3. Run the C++ unit test suite (124 tests)
ctest --test-dir build --output-on-failure
```

#### CMake Configuration Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `BUILD_TESTING` | `ON` | Build the GoogleTest unit test suite (`pyreth_tests`) |
| `BUILD_BENCHMARKS` | `ON` | Build the Google Benchmark micro-benchmark suite (`pyreth_benchmarks`) |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the `nanobind` `_core` Python extension module (`ON` automatically under `uv`/`pip`) |
| `ENABLE_ASAN` | `OFF` | Instrument targets with Clang/GCC/MSVC AddressSanitizer (`-fsanitize=address`) |
| `ENABLE_UBSAN` | `OFF` | Instrument targets with UndefinedBehaviorSanitizer (`-fsanitize=undefined`) |
| `ENABLE_TSAN` | `OFF` | Instrument targets with ThreadSanitizer (`-fsanitize=thread`) |

---

## Quick Start

### Python Quick Start

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pyreth import (
    ArenaAllocator,
    DynamicBatcher,
    DynamicBatcherConfig,
    ExecutionProvider,
    InferenceSession,
    SessionConfig,
    Tensor,
)

# 1. Configure and load an ONNX model (from file path or bytes)
config = SessionConfig(
    intra_op_num_threads=1,
    providers=[ExecutionProvider.CPU],
)
session = InferenceSession("model.onnx", config)
print(session.model_info)

# 2. Zero-copy synchronous inference with NumPy and ArenaAllocator reuse
arena = ArenaAllocator(1024 * 1024)
x = np.random.randn(4, 64).astype(np.float32)

outputs = session.run(x, arena=arena)
y_np = outputs[0].numpy()  # Zero-copy NumPy view backed by the arena
del outputs, y_np
arena.reset()  # O(1) pointer rewind once views are released

# 3. High-throughput concurrent serving with DynamicBatcher (releases GIL)
batcher_cfg = DynamicBatcherConfig(
    max_batch_size=32,
    max_batch_delay_ms=2,
    num_workers=4,
)

with DynamicBatcher(session, batcher_cfg) as batcher:
    samples = [np.random.randn(1, 64).astype(np.float32) for _ in range(64)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(batcher.run_numpy, samples))
    print(batcher.metrics().to_dict())
```

### C++20 Quick Start

```cpp
#include <iostream>
#include <pyreth/pyreth.hpp>

int main() {
  using namespace pyreth;

  // 1. Create an InferenceSession
  SessionConfig session_cfg;
  session_cfg.setIntraOpNumThreads(1);
  auto session_res = InferenceSession::create("model.onnx", session_cfg);
  if (!session_res.isOk()) {
    std::cerr << "Load failed: " << session_res.error().message() << "\n";
    return 1;
  }
  InferenceSession session = std::move(session_res).value();

  // 2. Allocate a 64-byte aligned input tensor inside an ArenaAllocator
  ArenaAllocator arena(1024 * 1024);
  Tensor x = Tensor::create(Shape{1, 64}, DataType::Float32, arena).value();
  x.fill<float>(1.0f);

  // 3. Dispatch through DynamicBatcher
  DynamicBatcherConfig batcher_cfg;
  batcher_cfg.setMaxBatchSize(16).setMaxBatchDelayMs(2);
  DynamicBatcher batcher(session, batcher_cfg);

  auto future_res = batcher.submit({x});
  auto outputs_res = future_res.value().get();
  if (outputs_res.isOk()) {
    const Tensor& y = outputs_res.value()[0];
    std::cout << "Output shape: " << y.shape() << "\n";
  }
  return 0;
}
```

---

## Testing & Sanitizers

### Running Unit & Integration Tests

```sh
# Run C++ unit tests via CTest
ctest --test-dir build --output-on-failure

# Run Python integration & concurrency tests via uv
uv run pytest -v
```

### Running Clang Sanitizers (`ASan`, `UBSan`, `TSan`)

Pyreth's lock-free queues, worker parker, memory pools, and dynamic batcher are verified under Clang's sanitizers in CI across Linux (`ubuntu-latest`) and macOS (`macos-latest`):

```sh
# 1. AddressSanitizer (ASan) + UndefinedBehaviorSanitizer (UBSan)
cmake -B build-asan -S . -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure

# 2. ThreadSanitizer (TSan)
cmake -B build-tsan -S . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_TSAN=ON -DBUILD_BENCHMARKS=OFF
cmake --build build-tsan --parallel
ctest --test-dir build-tsan --output-on-failure
```

---

## Performance Benchmarking

```sh
# Run C++ Google Benchmark micro-benchmarks (Arena vs malloc, WorkStealingQueue, Tensor, InferenceSession)
./build/benchmarks/pyreth_benchmarks --benchmark_min_time=0.05s

# Run end-to-end Python serving benchmark suite (Batch sizes 1..64 + regression checks)
uv run python benchmarks/benchmark_serving.py
```

See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for detailed latency percentiles (`P50`, `P90`, `P99`) and throughput (`requests/sec` and `samples/sec`).

---

## Code Style & Static Analysis

Repository formatting and static analysis standards are enforced in CI on every push and pull request:

### C++ (`clang-format` & `clang-tidy`)

C++ formatting is governed by [`.clang-format`](.clang-format) and static analysis by [`.clang-tidy`](.clang-tidy).

```sh
# Check formatting across all C++ headers and sources (dry run)
find include src unittest sandbox python benchmarks \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format --dry-run --Werror

# Format all C++ files in-place
find include src unittest sandbox python benchmarks \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i

# Run clang-tidy static analysis
cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
clang-tidy -p build src/core/*.cpp
```

### Python (`ruff`)

```sh
# Check Python linting and formatting
uv run ruff check .
uv run ruff format --check .

# Auto-format and fix
uv run ruff format .
uv run ruff check --fix .
```

---

## Documentation

- [`docs/tensor.md`](docs/tensor.md) — `Tensor`, `Shape`, `DataType`, 64-byte SIMD alignment, and storage backends
- [`docs/memory.md`](docs/memory.md) — `ArenaAllocator`, `MemoryPool`, `ScopedArena`, and pin-counted lifetime safety
- [`docs/parallelism.md`](docs/parallelism.md) — Lock-free `WorkStealingQueue`, `WorkerParker`, and `ThreadPool` architecture
- [`docs/inference.md`](docs/inference.md) — `Runtime`, `Backend`, `ModelInfo`, `InferenceSession`, and zero-copy ORT execution
- [`docs/dynamic_batching.md`](docs/dynamic_batching.md) — `DynamicBatcher` latency-bounded request coalescing and demultiplexing
- [`docs/python_api.md`](docs/python_api.md) — Python `nanobind` API reference, zero-copy NumPy ownership, and GIL release architecture
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) — Micro-benchmark and end-to-end serving benchmark report

---

## License

See [LICENSE](LICENSE).
