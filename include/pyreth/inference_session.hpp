#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pyreth/backend.hpp>
#include <pyreth/common.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/result.hpp>
#include <pyreth/runtime.hpp>
#include <pyreth/thread_pool.hpp>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Ort {
struct Session;
}

namespace pyreth {

/**
 * @enum OptimizationLevel
 * @brief ONNX Runtime graph optimization levels.
 */
enum class OptimizationLevel : uint8_t { Disable = 0, Basic, Extended, All };

/**
 * @brief Convert OptimizationLevel to its canonical string representation.
 */
[[nodiscard]] constexpr std::string_view optimizationLevelToString(
    OptimizationLevel level) noexcept {
  switch (level) {
    case OptimizationLevel::Disable:
      return "Disable";
    case OptimizationLevel::Basic:
      return "Basic";
    case OptimizationLevel::Extended:
      return "Extended";
    case OptimizationLevel::All:
      return "All";
    default:
      return "Unknown";
  }
}

/**
 * @enum SessionExecutionMode
 * @brief Execution mode defining sequential operator evaluation vs parallel graph execution.
 */
enum class SessionExecutionMode : uint8_t { Sequential = 0, Parallel };

/**
 * @brief Convert SessionExecutionMode to its canonical string representation.
 */
[[nodiscard]] constexpr std::string_view sessionExecutionModeToString(
    SessionExecutionMode mode) noexcept {
  switch (mode) {
    case SessionExecutionMode::Sequential:
      return "Sequential";
    case SessionExecutionMode::Parallel:
      return "Parallel";
    default:
      return "Unknown";
  }
}

/**
 * @struct SessionConfig
 * @brief Fine-grained configuration options for an InferenceSession.
 *
 * Controls intra/inter-op thread budgets, graph optimization level, execution providers,
 * and memory patterns.
 *
 * @note To prevent thread oversubscription when Pyreth's ThreadPool is orchestrating
 * parallel inference requests or dynamic batches, `intra_op_num_threads` defaults to 1.
 */
struct PYRETH_API SessionConfig {
  int intra_op_num_threads{1};
  int inter_op_num_threads{1};
  OptimizationLevel graph_optimization_level{OptimizationLevel::All};
  SessionExecutionMode execution_mode{SessionExecutionMode::Sequential};
  bool enable_cpu_mem_arena{true};
  bool enable_mem_pattern{true};
  LogLevel log_severity_level{LogLevel::Warning};
  std::string log_id{"PyrethSession"};
  std::vector<ExecutionProvider> providers{ExecutionProvider::CPU};
  bool allow_provider_fallback{true};
  bool enable_profiling{false};
  std::string profile_file_prefix{""};

  SessionConfig& setIntraOpNumThreads(int threads) noexcept {
    intra_op_num_threads = threads;
    return *this;
  }

  SessionConfig& setInterOpNumThreads(int threads) noexcept {
    inter_op_num_threads = threads;
    return *this;
  }

  SessionConfig& setGraphOptimizationLevel(OptimizationLevel level) noexcept {
    graph_optimization_level = level;
    return *this;
  }

  SessionConfig& setExecutionMode(SessionExecutionMode mode) noexcept {
    execution_mode = mode;
    return *this;
  }

  SessionConfig& setEnableCpuMemArena(bool enable) noexcept {
    enable_cpu_mem_arena = enable;
    return *this;
  }

  SessionConfig& setEnableMemPattern(bool enable) noexcept {
    enable_mem_pattern = enable;
    return *this;
  }

  SessionConfig& setLogSeverityLevel(LogLevel level) noexcept {
    log_severity_level = level;
    return *this;
  }

  SessionConfig& setLogId(std::string id) {
    log_id = std::move(id);
    return *this;
  }

  SessionConfig& setProviders(std::vector<ExecutionProvider> provs) {
    providers = std::move(provs);
    return *this;
  }

  SessionConfig& addProvider(ExecutionProvider provider) {
    providers.push_back(provider);
    return *this;
  }

  SessionConfig& setAllowProviderFallback(bool allow) noexcept {
    allow_provider_fallback = allow;
    return *this;
  }

  SessionConfig& enableProfiling(std::string prefix) {
    enable_profiling = true;
    profile_file_prefix = std::move(prefix);
    return *this;
  }

  SessionConfig& disableProfiling() noexcept {
    enable_profiling = false;
    profile_file_prefix.clear();
    return *this;
  }
};

/**
 * @class InferenceSession
 * @brief High-performance wrapper around Ort::Session with deterministic error handling
 * and resource lifecycle guarantees.
 *
 * Provides factory methods returning Result<InferenceSession> rather than throwing
 * C++ exceptions, enabling deterministic latency in server serving loops.
 */
class PYRETH_API InferenceSession {
public:
  /**
   * @brief Constructs an InferenceSession by loading an ONNX model from the filesystem.
   *
   * @param model_path Path to the `.onnx` model file.
   * @param config Session configuration options and execution provider preferences.
   * @return Result containing the initialized InferenceSession, or an Error status
   *         (e.g., MODEL_NOT_FOUND, MODEL_LOAD_FAILED, BACKEND_UNAVAILABLE).
   */
  static Result<InferenceSession> create(std::string_view model_path,
                                         const SessionConfig& config = SessionConfig{});

  /**
   * @brief Constructs an InferenceSession by parsing in-memory ONNX model protobuf bytes.
   *
   * @param model_bytes Contiguous byte buffer containing serialized ONNX model data.
   * @param config Session configuration options and execution provider preferences.
   * @return Result containing the initialized InferenceSession, or an Error status.
   */
  static Result<InferenceSession> create(std::span<const uint8_t> model_bytes,
                                         const SessionConfig& config = SessionConfig{});

  /**
   * @brief Default constructor creates an empty, uninitialized (invalid) session.
   */
  InferenceSession() noexcept;

  /**
   * @brief Move constructor transfers underlying session ownership.
   */
  InferenceSession(InferenceSession&& other) noexcept;

  /**
   * @brief Move assignment operator transfers session ownership.
   */
  InferenceSession& operator=(InferenceSession&& other) noexcept;

  /**
   * @brief Destructor releases underlying session resources.
   */
  ~InferenceSession();

  // Non-copyable
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession& operator=(const InferenceSession&) = delete;

  /**
   * @brief Returns true if this session holds a valid, initialized Ort::Session.
   */
  [[nodiscard]] bool isValid() const noexcept;

  /**
   * @brief Returns the configuration options applied when creating this session.
   */
  [[nodiscard]] const SessionConfig& getConfig() const noexcept;

  /**
   * @brief Returns the list of execution providers actively enabled for this session.
   */
  [[nodiscard]] const std::vector<ExecutionProvider>& getActiveProviders() const noexcept;

  /**
   * @brief Returns the file path of the model if loaded from disk, or empty if loaded from memory.
   */
  [[nodiscard]] std::string_view getModelPath() const noexcept;

  /**
   * @brief Returns the number of model inputs.
   */
  [[nodiscard]] size_t getInputCount() const noexcept;

  /**
   * @brief Returns the number of model outputs.
   */
  [[nodiscard]] size_t getOutputCount() const noexcept;

  /**
   * @brief Returns the names of all model inputs in index order.
   */
  [[nodiscard]] std::vector<std::string> getInputNames() const;

  /**
   * @brief Returns the names of all model outputs in index order.
   */
  [[nodiscard]] std::vector<std::string> getOutputNames() const;

  /**
   * @brief Returns the cached ModelInfo metadata and IO schema for this session.
   */
  [[nodiscard]] const ModelInfo& getModelInfo() const noexcept;

  /**
   * @brief Returns cached metadata for all model inputs in positional order.
   */
  [[nodiscard]] const std::vector<TensorInfo>& getInputs() const noexcept;

  /**
   * @brief Returns cached metadata for all model outputs in positional order.
   */
  [[nodiscard]] const std::vector<TensorInfo>& getOutputs() const noexcept;

  /**
   * @brief Retrieves the TensorInfo for a specific model input by index.
   */
  [[nodiscard]] Result<TensorInfo> getInputInfo(size_t index) const noexcept;

  /**
   * @brief Retrieves the TensorInfo for a specific model input by name.
   */
  [[nodiscard]] Result<TensorInfo> getInputInfo(std::string_view name) const noexcept;

  /**
   * @brief Retrieves the TensorInfo for a specific model output by index.
   */
  [[nodiscard]] Result<TensorInfo> getOutputInfo(size_t index) const noexcept;

  /**
   * @brief Retrieves the TensorInfo for a specific model output by name.
   */
  [[nodiscard]] Result<TensorInfo> getOutputInfo(std::string_view name) const noexcept;

  /**
   * @brief Zero-overhead validation of runtime input tensors against the cached ModelInfo schema.
   */
  [[nodiscard]] Status validateInputs(std::span<const Tensor> inputs) const noexcept;

  /**
   * @brief Executes synchronous zero-copy inference on positional input tensors.
   *
   * Input `Tensor` storage is pinned for the entire execution duration and wrapped
   * via `CreateTensorWithDataAsOrtValue` without copying. Returned output `Tensor`s
   * take zero-copy ownership of the resulting `Ort::Value` buffers.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::span<const Tensor> inputs);

  /**
   * @brief Convenience overload for initializer-list input tensors.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::initializer_list<Tensor> inputs) {
    return run(std::span<const Tensor>(inputs.begin(), inputs.size()));
  }

  /**
   * @brief Executes synchronous inference allocating outputs directly inside `output_arena`.
   *
   * When output dimensions are static or determinable from symbolic input dimensions,
   * output tensors are pre-allocated in `output_arena` before execution so ONNX Runtime
   * writes directly into the 64-byte aligned arena with zero intermediate copy.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::span<const Tensor> inputs,
                                                ArenaAllocator& output_arena);

  /**
   * @brief Executes synchronous inference allocating outputs from a leased arena in `output_pool`.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::span<const Tensor> inputs,
                                                MemoryPool& output_pool);

  /**
   * @brief Executes synchronous in-place zero-copy inference into pre-allocated output tensors.
   *
   * Both `inputs` and `outputs` remain pinned for the entire execution duration and are
   * bound directly via `CreateTensorWithDataAsOrtValue`.
   */
  [[nodiscard]] Status run(std::span<const Tensor> inputs, std::span<Tensor> outputs);

  /**
   * @brief Executes synchronous zero-copy inference using named input tensors.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(
      const std::unordered_map<std::string, Tensor>& named_inputs);

  /**
   * @brief Dispatches zero-copy inference asynchronously onto a Pyreth ThreadPool.
   *
   * Shallow `Tensor` handles are captured immediately on the calling thread, ensuring
   * all underlying `TensorStorage` / `ArenaAllocator` instances remain pinned until the
   * worker thread completes execution.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      ThreadPool& pool, std::span<const Tensor> inputs);

  /**
   * @brief Convenience overload for asynchronous inference with initializer-list inputs.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      ThreadPool& pool, std::initializer_list<Tensor> inputs) {
    return runAsync(pool, std::span<const Tensor>(inputs.begin(), inputs.size()));
  }

  /**
   * @brief Dispatches asynchronous inference onto `pool` with outputs backed by `output_pool`.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      ThreadPool& pool, std::span<const Tensor> inputs, MemoryPool& output_pool);

  /**
   * @brief Dispatches asynchronous in-place zero-copy inference into pre-allocated output tensors.
   */
  [[nodiscard]] Result<TaskHandle<Status>> runAsync(ThreadPool& pool,
                                                    std::span<const Tensor> inputs,
                                                    std::span<Tensor> outputs);

  /**
   * @brief Direct mutable access to the underlying Ort::Session.
   * @note Requires `#include <onnxruntime_cxx_api.h>` by the caller.
   */
  [[nodiscard]] Ort::Session& getOrtSession();

  /**
   * @brief Direct const access to the underlying Ort::Session.
   * @note Requires `#include <onnxruntime_cxx_api.h>` by the caller.
   */
  [[nodiscard]] const Ort::Session& getOrtSession() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit InferenceSession(std::unique_ptr<Impl> impl) noexcept;
};

}  // namespace pyreth
