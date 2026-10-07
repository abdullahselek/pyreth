#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <pyreth/common.hpp>
#include <pyreth/result.hpp>

namespace pyreth {

/**
 * @enum ExecutionProvider
 * @brief Strongly typed hardware execution providers supported by Pyreth.
 */
enum class ExecutionProvider : uint8_t {
  CPU = 0,
  COREML,
  CUDA,
  TENSORRT,
  DIRECTML,
  ROCM,
  UNKNOWN
};

/**
 * @brief Convert an ExecutionProvider enum to its canonical string name.
 */
[[nodiscard]] constexpr std::string_view
executionProviderToString(ExecutionProvider ep) noexcept {
  switch (ep) {
  case ExecutionProvider::CPU:
    return "CPUExecutionProvider";
  case ExecutionProvider::COREML:
    return "CoreMLExecutionProvider";
  case ExecutionProvider::CUDA:
    return "CUDAExecutionProvider";
  case ExecutionProvider::TENSORRT:
    return "TensorrtExecutionProvider";
  case ExecutionProvider::DIRECTML:
    return "DmlExecutionProvider";
  case ExecutionProvider::ROCM:
    return "ROCMExecutionProvider";
  case ExecutionProvider::UNKNOWN:
  default:
    return "Unknown";
  }
}

/**
 * @brief Parse an execution provider string name into the strongly-typed enum.
 */
[[nodiscard]] PYRETH_API ExecutionProvider
parseExecutionProvider(std::string_view name) noexcept;

/**
 * @class Backend
 * @brief Utility for querying runtime execution providers and hardware
 * acceleration status.
 */
class PYRETH_API Backend {
public:
  /**
   * @brief Returns the version string of the underlying ONNX Runtime.
   */
  [[nodiscard]] static std::string getRuntimeVersion();

  /**
   * @brief Queries all execution providers currently registered and available
   * in the runtime.
   */
  [[nodiscard]] static std::vector<std::string> getAvailableProviders();

  /**
   * @brief Queries all execution providers mapped to strongly-typed enum
   * values.
   */
  [[nodiscard]] static std::vector<ExecutionProvider>
  getAvailableExecutionProviders();

  /**
   * @brief Checks if a specific provider name is available in the current
   * runtime.
   */
  [[nodiscard]] static bool isProviderAvailable(std::string_view provider_name);

  /**
   * @brief Checks if a strongly-typed provider is available in the current
   * runtime.
   */
  [[nodiscard]] static bool isProviderAvailable(ExecutionProvider provider);

  /**
   * @brief Returns the best hardware acceleration provider available on the
   * current machine.
   *
   * Priority hierarchy:
   * - TensorRT / CUDA
   * - CoreML
   * - DirectML
   * - ROCm
   * - Fallback: CPU
   */
  [[nodiscard]] static ExecutionProvider getBestAvailableProvider();

  /**
   * @brief Returns true if any hardware accelerator (non-CPU) is available.
   */
  [[nodiscard]] static bool hasHardwareAcceleration();
};

} // namespace pyreth
