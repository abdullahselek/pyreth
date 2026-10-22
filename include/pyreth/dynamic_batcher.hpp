#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>
#include <pyreth/thread_pool.hpp>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace pyreth {

/**
 * @struct DynamicBatcherConfig
 * @brief Configuration parameters for DynamicBatcher request scheduling and batching.
 *
 * Controls the maximum batch size (`max_batch_size`), the maximum latency window
 * (`max_batch_delay_ms`) before flushing a partial batch, queue capacity, and memory
 * pool arena sizing.
 */
struct PYRETH_API DynamicBatcherConfig {
  size_t max_batch_size{16};                    ///< Maximum samples along axis 0 per batch
  uint64_t max_batch_delay_ms{5};               ///< Max wait window in milliseconds
  uint64_t max_batch_delay_us{0};               ///< Optional microsecond override (if > 0)
  size_t max_queue_capacity{1024};              ///< Maximum pending requests in queue
  size_t num_workers{0};                        ///< Internal ThreadPool workers (0 = auto)
  size_t arena_block_size{DEFAULT_BLOCK_SIZE};  ///< Arena block size for internal MemoryPool

  DynamicBatcherConfig& setMaxBatchSize(size_t size) noexcept {
    max_batch_size = size;
    return *this;
  }

  DynamicBatcherConfig& setMaxBatchDelayMs(uint64_t delay_ms) noexcept {
    max_batch_delay_ms = delay_ms;
    max_batch_delay_us = 0;
    return *this;
  }

  DynamicBatcherConfig& setMaxBatchDelayUs(uint64_t delay_us) noexcept {
    max_batch_delay_us = delay_us;
    return *this;
  }

  DynamicBatcherConfig& setMaxQueueCapacity(size_t capacity) noexcept {
    max_queue_capacity = capacity;
    return *this;
  }

  DynamicBatcherConfig& setNumWorkers(size_t workers) noexcept {
    num_workers = workers;
    return *this;
  }

  DynamicBatcherConfig& setArenaBlockSize(size_t block_size) noexcept {
    arena_block_size = block_size;
    return *this;
  }

  [[nodiscard]] constexpr std::chrono::microseconds effectiveBatchDelay() const noexcept {
    if (max_batch_delay_us > 0) {
      return std::chrono::microseconds(max_batch_delay_us);
    }
    return std::chrono::microseconds(max_batch_delay_ms * 1000ULL);
  }
};

using BatcherConfig = DynamicBatcherConfig;

/**
 * @struct DynamicBatcherMetrics
 * @brief Runtime throughput, batching efficiency, and latency telemetry for DynamicBatcher.
 */
struct PYRETH_API DynamicBatcherMetrics {
  uint64_t requests_submitted{0};
  uint64_t requests_completed{0};
  uint64_t requests_rejected{0};
  uint64_t batches_dispatched{0};
  uint64_t full_batches{0};
  uint64_t timeout_batches{0};
  uint64_t total_batched_samples{0};
  uint64_t total_batch_wait_us{0};
  uint64_t total_execution_us{0};

  [[nodiscard]] constexpr double averageBatchSize() const noexcept {
    return batches_dispatched > 0 ? static_cast<double>(total_batched_samples) /
                                        static_cast<double>(batches_dispatched)
                                  : 0.0;
  }

  [[nodiscard]] constexpr double averageBatchWaitUs() const noexcept {
    return batches_dispatched > 0
               ? static_cast<double>(total_batch_wait_us) / static_cast<double>(batches_dispatched)
               : 0.0;
  }

  [[nodiscard]] constexpr double averageExecutionUs() const noexcept {
    return batches_dispatched > 0
               ? static_cast<double>(total_execution_us) / static_cast<double>(batches_dispatched)
               : 0.0;
  }
};

using BatcherMetrics = DynamicBatcherMetrics;

/**
 * @class DynamicBatcher
 * @brief High-throughput dynamic batching scheduler and inference engine.
 *
 * Queues incoming asynchronous or synchronous inference requests within a configurable
 * maximum latency window (`max_batch_delay_ms`) or up to `max_batch_size`, concatenates
 * individual request input tensors into a single contiguous 64-byte aligned arena tensor,
 * dispatches a single batched `InferenceSession::run()` call on the `ThreadPool`, and
 * demultiplexes the output tensors back to each caller with zero-copy slicing.
 */
class PYRETH_API DynamicBatcher {
public:
  /**
   * @brief Factory method constructing a DynamicBatcher with internal ThreadPool and MemoryPool.
   */
  static Result<DynamicBatcher> create(InferenceSession& session,
                                       const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Factory method constructing a DynamicBatcher using an external ThreadPool.
   */
  static Result<DynamicBatcher> create(InferenceSession& session, ThreadPool& pool,
                                       const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Factory method constructing a DynamicBatcher using external ThreadPool and MemoryPool.
   */
  static Result<DynamicBatcher> create(InferenceSession& session, ThreadPool& pool,
                                       MemoryPool& memory_pool,
                                       const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Default constructor creates an empty, uninitialized batcher.
   */
  DynamicBatcher() noexcept;

  /**
   * @brief Constructs a DynamicBatcher managing its own internal ThreadPool and MemoryPool.
   */
  explicit DynamicBatcher(InferenceSession& session,
                          const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Constructs a DynamicBatcher dispatching onto an external ThreadPool.
   */
  DynamicBatcher(InferenceSession& session, ThreadPool& pool,
                 const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Constructs a DynamicBatcher using an external ThreadPool and MemoryPool.
   */
  DynamicBatcher(InferenceSession& session, ThreadPool& pool, MemoryPool& memory_pool,
                 const DynamicBatcherConfig& config = DynamicBatcherConfig{});

  /**
   * @brief Move constructor transfers ownership of the scheduler and worker resources.
   */
  DynamicBatcher(DynamicBatcher&& other) noexcept;

  /**
   * @brief Move assignment operator transfers ownership of the scheduler and worker resources.
   */
  DynamicBatcher& operator=(DynamicBatcher&& other) noexcept;

  /**
   * @brief Destructor drains pending requests, stops the scheduler thread, and awaits in-flight
   * batches.
   */
  ~DynamicBatcher();

  // Non-copyable
  DynamicBatcher(const DynamicBatcher&) = delete;
  DynamicBatcher& operator=(const DynamicBatcher&) = delete;

  /**
   * @brief Enqueues an asynchronous inference request with positional input tensors.
   *
   * @param inputs Ordered input tensors for a single request (e.g., batch size >= 1 along axis 0).
   * @return Result containing a TaskHandle that resolves to this request's demultiplexed outputs.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> submit(
      std::span<const Tensor> inputs);

  /**
   * @brief Convenience overload for initializer-list input tensors.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> submit(
      std::initializer_list<Tensor> inputs) {
    return submit(std::span<const Tensor>(inputs.begin(), inputs.size()));
  }

  /**
   * @brief Enqueues an asynchronous inference request using named input tensors.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> submit(
      const std::unordered_map<std::string, Tensor>& named_inputs);

  /**
   * @brief Alias for submit() matching InferenceSession::runAsync().
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      std::span<const Tensor> inputs) {
    return submit(inputs);
  }

  /**
   * @brief Alias for submit() with initializer-list inputs.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      std::initializer_list<Tensor> inputs) {
    return submit(std::span<const Tensor>(inputs.begin(), inputs.size()));
  }

  /**
   * @brief Alias for submit() with named input tensors.
   */
  [[nodiscard]] Result<TaskHandle<Result<std::vector<Tensor>>>> runAsync(
      const std::unordered_map<std::string, Tensor>& named_inputs) {
    return submit(named_inputs);
  }

  /**
   * @brief Submits an inference request to the dynamic batcher and blocks until its batch
   * completes and outputs are demultiplexed.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::span<const Tensor> inputs);

  /**
   * @brief Synchronous batched execution overload for initializer-list inputs.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(std::initializer_list<Tensor> inputs) {
    return run(std::span<const Tensor>(inputs.begin(), inputs.size()));
  }

  /**
   * @brief Synchronous batched execution overload for named input tensors.
   */
  [[nodiscard]] Result<std::vector<Tensor>> run(
      const std::unordered_map<std::string, Tensor>& named_inputs);

  /**
   * @brief Immediately flushes all queued requests into batches and blocks until all in-flight
   * batches complete.
   */
  void flush();

  /**
   * @brief Drains any remaining queued requests, stops the scheduler thread, and waits for all
   * in-flight batch executions to finish.
   */
  void shutdown() noexcept;

  /**
   * @brief Returns true if the batcher has been shut down.
   */
  [[nodiscard]] bool isShutdown() const noexcept;

  /**
   * @brief Returns true if the batcher is initialized and holds a valid InferenceSession.
   */
  [[nodiscard]] bool isValid() const noexcept;

  /**
   * @brief Returns the active configuration for this batcher.
   */
  [[nodiscard]] const DynamicBatcherConfig& getConfig() const noexcept;

  /**
   * @brief Returns the number of requests currently waiting in the batch queue.
   */
  [[nodiscard]] size_t pendingCount() const noexcept;

  /**
   * @brief Returns a snapshot of runtime batching and throughput telemetry.
   */
  [[nodiscard]] DynamicBatcherMetrics metrics() const noexcept;

  /**
   * @brief Resets all runtime telemetry counters to zero.
   */
  void resetMetrics() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  explicit DynamicBatcher(std::unique_ptr<Impl> impl) noexcept;
};

}  // namespace pyreth
