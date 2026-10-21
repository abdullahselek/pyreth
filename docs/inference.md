# Inference Engine & Session Architecture

Pyreth provides a C++20 inference session and runtime wrapper on top of ONNX Runtime (ORT), designed for deterministic latency, zero-copy tensor passing, and safe concurrency in high-throughput server deployments.

---

## 1. Architectural Overview

```
                      +-----------------------------+
                      |      Runtime Singleton      |
                      |   (Thread-Safe Ort::Env)    |
                      +--------------+--------------+
                                     |
              +----------------------+----------------------+
              |                                             |
              v                                             v
    +-------------------+                         +-------------------+
    | InferenceSession  |                         | InferenceSession  |
    | (Model A: ResNet) |                         | (Model B: BERT)   |
    +---------+---------+                         +---------+---------+
              |                                             |
     Session Options                               Session Options
   - intra_op: 1                                 - intra_op: 1
   - inter_op: 1                                 - inter_op: 1
   - GraphOpt: All                               - GraphOpt: All
   - Provider: CoreML / CUDA                     - Provider: CPU
```

### The `Runtime` Singleton
`Ort::Env` manages global state across the entire process: logging channels, platform facilities, and global thread pool metadata. Creating multiple `Ort::Env` instances in a single process introduces thread contention and undefined telemetry state.

`pyreth::Runtime` provides a thread-safe singleton (`Runtime::instance()`) that guarantees:
- **Centralized Initialization**: A single, process-wide `Ort::Env` instance is lazily allocated upon first access or explicitly configured via `Runtime::instance().init(log_level, log_id)`.
- **Thread Safety**: Concurrent calls to `Runtime::instance().getEnv()` across worker threads are safe and synchronized.
- **Controlled Teardown**: Clean lifecycle management via `Runtime::instance().shutdown()`.

---

## 2. Concurrency Model & Thread Budgeting

A common failure mode in multi-threaded inference servers is **thread oversubscription**:
- Pyreth maintains a high-throughput, lock-free work-stealing `ThreadPool` handling request scheduling, input preprocessing, and dynamic batch construction.
- If each ONNX Runtime session concurrently spins up its own internal thread pool (e.g. defaulting to `std::thread::hardware_concurrency()`), hundreds of threads will contend for CPU cores, causing excessive context switching and tail latency spikes.

### Pyreth's Rule of Concurrency
1. **Pyreth ThreadPool**: Serves as the primary orchestrator across CPU cores.
2. **ORT Intra-Op Threads**: Configured via `SessionConfig::intra_op_num_threads`. Defaults to **1** for batched or concurrent serving pipelines. If running isolated single-request inference where full core utilization is desired, `intra_op_num_threads` can be scaled explicitly.
3. **ORT Inter-Op Threads**: Configured via `SessionConfig::inter_op_num_threads`. Defaults to **1** to maintain sequential node evaluation within individual request tasks.

---

## 3. Session Configuration (`SessionConfig`)

`SessionConfig` offers builder-style configuration for graph execution and hardware options:

```cpp
#include <pyreth/inference_session.hpp>

pyreth::SessionConfig config;
config.setIntraOpNumThreads(1)
      .setInterOpNumThreads(1)
      .setGraphOptimizationLevel(pyreth::OptimizationLevel::All)
      .setExecutionMode(pyreth::SessionExecutionMode::Sequential)
      .setEnableCpuMemArena(true)
      .setEnableMemPattern(true)
      .setLogSeverityLevel(pyreth::LogLevel::Warning)
      .setProviders({pyreth::ExecutionProvider::COREML, pyreth::ExecutionProvider::CPU})
      .setAllowProviderFallback(true);
```

### Graph Optimization Levels
| Level | Enum Value | Description |
| :--- | :--- | :--- |
| **Disable** | `OptimizationLevel::Disable` | Preserves verbatim computation graph; useful for debugging and tracing. |
| **Basic** | `OptimizationLevel::Basic` | Constant folding, redundant node elimination, and dead code pruning. |
| **Extended** | `OptimizationLevel::Extended` | Kernel fusion (e.g., Conv + BatchNorm + ReLU fusion), memory layout optimization. |
| **All** | `OptimizationLevel::All` | Level 1 + Level 2 + hardware-specific graph transformations (e.g., CoreML / TensorRT fusions). |

### Execution Modes
- `SessionExecutionMode::Sequential`: Evaluates graph operators in topological sequence. Ideal for batched serving under Pyreth's thread pool.
- `SessionExecutionMode::Parallel`: Executes independent operators concurrently across ORT inter-op threads.

---

## 4. Execution Provider Registration & Fallback

Pyreth supports hardware acceleration backends (CoreML, CUDA, TensorRT, ROCm) with deterministic fallback:

1. **Explicit Priority Order**: When `providers` is populated with multiple entries (e.g. `{COREML, CPU}`), Pyreth evaluates each provider sequentially against `Backend::isProviderAvailable()`.
2. **Graceful Fallback (`allow_provider_fallback = true`)**: If a requested hardware accelerator is unavailable (e.g. running on non-Apple hardware without CUDA), Pyreth logs the absence, skips the provider, and continues down the priority chain until a valid backend (such as CPU) is registered.
3. **Strict Validation (`allow_provider_fallback = false`)**: If a requested provider cannot be registered and fallback is disabled, `InferenceSession::create()` halts immediately and returns `StatusCode::BACKEND_UNAVAILABLE` or `StatusCode::BACKEND_ERROR`.

---

## 5. C++ API & Error Handling

To maintain zero-overhead deterministic execution, `InferenceSession` does not throw C++ exceptions across its public factory interface. All creation and validation errors return `pyreth::Result<InferenceSession>`.

### Loading from Filesystem
```cpp
#include <pyreth/inference_session.hpp>
#include <iostream>

pyreth::SessionConfig config;
config.setIntraOpNumThreads(1)
      .setGraphOptimizationLevel(pyreth::OptimizationLevel::All);

auto session_res = pyreth::InferenceSession::create("models/resnet50.onnx", config);
if (!session_res.isOk()) {
  std::cerr << "Failed to load model: " << session_res.error().message() 
            << " (Code: " << static_cast<int>(session_res.error().code()) << ")\n";
  return;
}

auto session = std::move(session_res).value();
std::cout << "Model loaded successfully from: " << session.getModelPath() << "\n";
std::cout << "Input count: " << session.getInputCount() << "\n";
```

### Loading from In-Memory Buffer
```cpp
#include <pyreth/inference_session.hpp>
#include <span>

std::vector<uint8_t> model_bytes = fetchModelFromNetworkOrPackage();

auto session_res = pyreth::InferenceSession::create(
    std::span<const uint8_t>(model_bytes.data(), model_bytes.size()));

if (session_res.isOk()) {
  auto& session = session_res.value();
  for (const auto& name : session.getInputNames()) {
    std::cout << "Input name: " << name << "\n";
  }
}
```

### Error Conditions
- `StatusCode::MODEL_NOT_FOUND`: Target file path does not exist on the filesystem.
- `StatusCode::INVALID_ARGUMENT`: Input byte buffer or model parameters are empty or malformed.
- `StatusCode::MODEL_LOAD_FAILED`: ORT failed to parse the protobuf graph (corrupted model, unsupported opset / IR version).
- `StatusCode::BACKEND_UNAVAILABLE`: A requested execution provider was unavailable when fallback was disabled.

---

## 6. Move Semantics & Thread Safety

- **Move-Only**: `InferenceSession` is strictly move-only (`InferenceSession(const InferenceSession&) = delete`). Transfer of session ownership safely relocates underlying ORT resources without reallocation.
- **Concurrent Inference**: A single `InferenceSession` instance is safe for concurrent inference runs across multiple worker threads once initialized.
