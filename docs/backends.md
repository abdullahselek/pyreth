# Hardware Backends & Execution Providers

Pyreth delegates low-level operator kernel evaluation to **ONNX Runtime (ORT)** execution providers while managing parallel scheduling, dynamic batching, and memory pools in C++20. 

To ensure portable, zero-configuration execution across heterogeneous environments, Pyreth implements an automated **hardware discovery and fallback chain**.

---

## 1. Provider Priority & Fallback Hierarchy

When configuring an inference session without an explicit backend override, Pyreth evaluates available execution providers using the following priority order:

```
[Incoming Request]
        │
        ├──► 1. TensorrtExecutionProvider (NVIDIA TensorRT, max throughput)
        │         │ (if unavailable)
        ├──► 2. CUDAExecutionProvider     (NVIDIA CUDA kernels)
        │         │ (if unavailable)
        ├──► 3. CoreMLExecutionProvider   (Apple Neural Engine / Metal on macOS)
        │         │ (if unavailable)
        ├──► 4. DmlExecutionProvider      (DirectML on Windows)
        │         │ (if unavailable)
        ├──► 5. ROCMExecutionProvider     (AMD ROCm on Linux)
        │         │ (if unavailable)
        └──► 6. CPUExecutionProvider      (Guaranteed universal fallback)
```

| Execution Provider | Primary Platform | Hardware Acceleration |
| :--- | :--- | :--- |
| `TensorrtExecutionProvider` | Linux / Windows | NVIDIA GPU (TensorRT FP16/INT8 engines) |
| `CUDAExecutionProvider` | Linux / Windows | NVIDIA GPU (CUDA compute kernels) |
| `CoreMLExecutionProvider` | macOS (Apple Silicon) | Apple Neural Engine (ANE) & Metal GPU |
| `DmlExecutionProvider` | Windows | DirectX 12 compatible GPUs (Intel, AMD, NVIDIA) |
| `ROCMExecutionProvider` | Linux | AMD Radeon / Instinct GPUs |
| `CPUExecutionProvider` | Universal | Multi-threaded host CPU (AVX2, AVX-512, ARM NEON) |

---

## 2. CMake Prebuilt Binary Integration

Rather than compiling ONNX Runtime from source—which requires extensive compilation times and gigabytes of submodules—Pyreth uses a hybrid discovery model managed by [`cmake/OnnxRuntime.cmake`](../cmake/OnnxRuntime.cmake):

1. **Local System Discovery**:
   CMake searches standard system paths:
   * `$ENV{ONNXRUNTIME_ROOT}` or `ONNXRUNTIME_ROOT`
   * `/opt/homebrew/opt/onnxruntime` (macOS Homebrew)
   * `/usr/local` / `/usr`
2. **Automated Prebuilt Fetching**:
   If no local installation is found, CMake queries the host architecture and downloads the official prebuilt binary tarball/zip from Microsoft:
   * **macOS Apple Silicon**: `onnxruntime-osx-arm64-<version>.tgz` (includes CoreML)
   * **macOS Intel**: `onnxruntime-osx-x86_64-<version>.tgz`
   * **Linux x86_64**: `onnxruntime-linux-x64-<version>.tgz`
   * **Linux aarch64**: `onnxruntime-linux-aarch64-<version>.tgz`
   * **Windows x64 / arm64**: `onnxruntime-win-<arch>-<version>.zip`
3. **Imported Target**:
   Defines `onnxruntime::onnxruntime` and sets RPATH automatically so executables locate the dynamic library (`.dylib` / `.so`) without manual path configuration.

---

## 3. C++ API Reference

The [`pyreth::Backend`](../include/pyreth/backend.hpp) utility exposes static discovery methods:

```cpp
#include <pyreth/backend.hpp>
#include <iostream>

using namespace pyreth;

// Query ORT version
std::cout << "ONNX Runtime Version: " << Backend::getRuntimeVersion() << "\n";

// List all registered execution providers
for (const auto& provider_name : Backend::getAvailableProviders()) {
  std::cout << " - " << provider_name << "\n";
}

// Check for specific provider availability
if (Backend::isProviderAvailable(ExecutionProvider::COREML)) {
  std::cout << "CoreML acceleration is active.\n";
}

// Query the best available hardware accelerator
ExecutionProvider best = Backend::getBestAvailableProvider();
std::cout << "Selected Backend: " << executionProviderToString(best) << "\n";

// Query whether any non-CPU accelerator is active
if (Backend::hasHardwareAcceleration()) {
  std::cout << "Hardware acceleration enabled.\n";
} else {
  std::cout << "Falling back to CPU execution.\n";
}
```

---

## 4. Hardware Acceleration Notes

* **Apple Silicon (M1/M2/M3/M4/M5 and newer)**: The official prebuilt macOS arm64 binary includes native CoreML support. Models can execute operations directly on the Apple Neural Engine (ANE) with low power consumption and high efficiency.
* **Linux / NVIDIA**: On systems with CUDA and TensorRT installed, prebuilt or custom ORT packages with GPU flags can be supplied via `ONNXRUNTIME_ROOT=/path/to/ort-gpu`.
* **Universal CPU**: `CPUExecutionProvider` is always guaranteed to be present and serves as the baseline fallback across all operating systems.
