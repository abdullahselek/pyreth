#pragma once

#include <cstdint>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <string>
#include <string_view>

namespace Ort {
struct Env;
}
namespace ort = Ort;

namespace pyreth {

/**
 * @enum LogLevel
 * @brief Discrete enumeration of logging severity levels for the engine and ORT.
 */
enum class LogLevel : uint8_t { Verbose = 0, Info, Warning, Error, Fatal };

/**
 * @brief Convert LogLevel to its canonical string representation.
 */
[[nodiscard]] constexpr std::string_view logLevelToString(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Verbose:
      return "Verbose";
    case LogLevel::Info:
      return "Info";
    case LogLevel::Warning:
      return "Warning";
    case LogLevel::Error:
      return "Error";
    case LogLevel::Fatal:
      return "Fatal";
    default:
      return "Unknown";
  }
}

/**
 * @class Runtime
 * @brief Thread-safe singleton managing the shared ONNX Runtime environment (Ort::Env).
 *
 * Ort::Env is a process-wide resource managing global thread pools, logging, and
 * platform state. Multiple Ort::Env instances in the same process are discouraged
 * by ONNX Runtime. Runtime provides centralized, thread-safe access to this shared
 * environment across sessions and worker threads.
 */
class PYRETH_API Runtime {
public:
  /**
   * @brief Returns the process-wide thread-safe Runtime singleton instance.
   */
  static Runtime& instance();

  /**
   * @brief Explicitly initializes or reconfigures the shared Ort::Env.
   *
   * If already initialized, replaces the active environment with the specified
   * logging configuration.
   *
   * @param log_level Initial logging severity filter.
   * @param log_id Identifier string prefix for runtime log records.
   * @return Status indicating success or internal initialization failure.
   */
  Status init(LogLevel log_level = LogLevel::Warning, std::string_view log_id = "PyrethRuntime");

  /**
   * @brief Checks whether the shared Ort::Env is currently allocated and ready.
   */
  [[nodiscard]] bool isInitialized() const noexcept;

  /**
   * @brief Returns the configured log level of the shared environment.
   */
  [[nodiscard]] LogLevel getLogLevel() const noexcept;

  /**
   * @brief Returns the configured log identifier of the shared environment.
   */
  [[nodiscard]] std::string getLogId() const;

  /**
   * @brief Retrieves a reference to the shared Ort::Env instance.
   *
   * If the environment has not been explicitly initialized, lazily initializes
   * it with default parameters (LogLevel::Warning, "PyrethRuntime").
   *
   * @note Callers using this method directly must include `<onnxruntime_cxx_api.h>`.
   */
  [[nodiscard]] ort::Env& getEnv();

  /**
   * @brief Retrieves a const reference to the shared Ort::Env instance.
   *
   * Lazily initializes with defaults if uninitialized.
   *
   * @note Callers using this method directly must include `<onnxruntime_cxx_api.h>`.
   */
  [[nodiscard]] const ort::Env& getEnv() const;

  /**
   * @brief Releases the shared Ort::Env instance.
   *
   * Useful for testing lifecycle transitions and clean teardown. Subsequent calls
   * to getEnv() will lazily re-instantiate the environment.
   */
  void shutdown();

  // Non-copyable, non-movable singleton contract
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  Runtime(Runtime&&) = delete;
  Runtime& operator=(Runtime&&) = delete;

private:
  Runtime();
  ~Runtime();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pyreth
