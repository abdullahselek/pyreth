#pragma once

#include <cstdint>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <string>
#include <string_view>

namespace pyreth::server {

/**
 * @enum ExecutionMode
 * @brief Serving concurrency and execution paradigm for inference requests.
 */
enum class ExecutionMode : uint8_t {
  Sync = 0,        ///< Synchronous execution on the calling event loop or thread.
  Async,           ///< Asynchronous offloading to Pyreth work-stealing ThreadPool.
  DynamicBatching  ///< Concurrent batching via DynamicBatcher.
};

/**
 * @brief Converts ExecutionMode to canonical string representation ("sync", "async", "batch").
 */
[[nodiscard]] constexpr std::string_view executionModeToString(ExecutionMode mode) noexcept {
  switch (mode) {
    case ExecutionMode::Sync:
      return "sync";
    case ExecutionMode::Async:
      return "async";
    case ExecutionMode::DynamicBatching:
      return "batch";
    default:
      return "unknown";
  }
}

/**
 * @brief Parses ExecutionMode from string representation.
 */
[[nodiscard]] inline Result<ExecutionMode> stringToExecutionMode(std::string_view str) noexcept {
  if (str == "sync" || str == "Sync" || str == "SYNC") {
    return Result<ExecutionMode>::success(ExecutionMode::Sync);
  }
  if (str == "async" || str == "Async" || str == "ASYNC") {
    return Result<ExecutionMode>::success(ExecutionMode::Async);
  }
  if (str == "batch" || str == "Batch" || str == "BATCH" || str == "dynamic_batching" ||
      str == "DynamicBatching") {
    return Result<ExecutionMode>::success(ExecutionMode::DynamicBatching);
  }
  return Result<ExecutionMode>::failure(StatusCode::INVALID_ARGUMENT,
                                        "Unsupported execution mode: " + std::string(str));
}

inline constexpr std::string_view kExecutionModeHeader = "X-Pyreth-Execution-Mode";

}  // namespace pyreth::server
