#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <pyreth/work_stealing_queue.hpp>
#include <pyreth/worker_parker.hpp>
#include <queue>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyreth {

/**
 * @class MoveOnlyTask
 * @brief Type-erased, move-only task wrapper optimized with Small Object
 * Optimization (SOO).
 *
 * Avoids dynamic heap allocation for callables fitting within 48 bytes (such as
 * lambdas capturing pointers or std::shared_ptr<std::packaged_task>).
 */
class MoveOnlyTask {
public:
  static constexpr size_t INLINE_CAPACITY = 48;

  MoveOnlyTask() noexcept = default;

  template <typename F>
    requires(!std::is_same_v<std::decay_t<F>, MoveOnlyTask> && std::is_invocable_v<F>)
  MoveOnlyTask(F&& f) {
    using Decayed = std::decay_t<F>;
    if constexpr (sizeof(Decayed) <= INLINE_CAPACITY &&
                  alignof(Decayed) <= alignof(std::max_align_t)) {
      new (storage_) Decayed(std::forward<F>(f));
      invoker_ = [](void* s) { (*reinterpret_cast<Decayed*>(s))(); };
      destroyer_ = [](void* s) { reinterpret_cast<Decayed*>(s)->~Decayed(); };
      mover_ = [](void* src, void* dest) noexcept {
        new (dest) Decayed(std::move(*reinterpret_cast<Decayed*>(src)));
        reinterpret_cast<Decayed*>(src)->~Decayed();
      };
    } else {
      auto ptr = std::make_unique<Decayed>(std::forward<F>(f));
      new (storage_) std::unique_ptr<Decayed>(std::move(ptr));
      invoker_ = [](void* s) { (**reinterpret_cast<std::unique_ptr<Decayed>*>(s))(); };
      destroyer_ = [](void* s) {
        reinterpret_cast<std::unique_ptr<Decayed>*>(s)->~unique_ptr<Decayed>();
      };
      mover_ = [](void* src, void* dest) noexcept {
        new (dest)
            std::unique_ptr<Decayed>(std::move(*reinterpret_cast<std::unique_ptr<Decayed>*>(src)));
        reinterpret_cast<std::unique_ptr<Decayed>*>(src)->~unique_ptr<Decayed>();
      };
    }
  }

  ~MoveOnlyTask() { reset(); }

  MoveOnlyTask(const MoveOnlyTask&) = delete;
  MoveOnlyTask& operator=(const MoveOnlyTask&) = delete;

  MoveOnlyTask(MoveOnlyTask&& other) noexcept { moveFrom(std::move(other)); }

  MoveOnlyTask& operator=(MoveOnlyTask&& other) noexcept {
    if (this != &other) {
      reset();
      moveFrom(std::move(other));
    }
    return *this;
  }

  void operator()() {
    if (invoker_) {
      invoker_(storage_);
      reset();
    }
  }

  [[nodiscard]] explicit operator bool() const noexcept { return invoker_ != nullptr; }

  void reset() noexcept {
    if (destroyer_) {
      destroyer_(storage_);
      destroyer_ = nullptr;
      invoker_ = nullptr;
      mover_ = nullptr;
    }
  }

private:
  void moveFrom(MoveOnlyTask&& other) noexcept {
    if (other.mover_) {
      other.mover_(other.storage_, storage_);
      invoker_ = other.invoker_;
      destroyer_ = other.destroyer_;
      mover_ = other.mover_;
      other.invoker_ = nullptr;
      other.destroyer_ = nullptr;
      other.mover_ = nullptr;
    }
  }

  using InvokerFn = void (*)(void* storage);
  using DestroyerFn = void (*)(void* storage);
  using MoverFn = void (*)(void* src, void* dest) noexcept;

  InvokerFn invoker_{nullptr};
  DestroyerFn destroyer_{nullptr};
  MoverFn mover_{nullptr};
  alignas(std::max_align_t) std::byte storage_[INLINE_CAPACITY];
};

/**
 * @class TaskHandle
 * @brief Move-only asynchronous task handle wrapping future results.
 *
 * @tparam R Return type of the submitted task.
 */
template <typename R>
class TaskHandle {
public:
  TaskHandle() = default;
  explicit TaskHandle(std::future<R> fut) : future_(std::move(fut)) {}

  TaskHandle(TaskHandle&&) noexcept = default;
  TaskHandle& operator=(TaskHandle&&) noexcept = default;

  TaskHandle(const TaskHandle&) = delete;
  TaskHandle& operator=(const TaskHandle&) = delete;

  [[nodiscard]] bool valid() const noexcept { return future_.valid(); }

  [[nodiscard]] bool isReady() const noexcept {
    return future_.valid() &&
           future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  void wait() const {
    if (future_.valid()) {
      future_.wait();
    }
  }

  template <typename Rep, typename Period>
  [[nodiscard]] bool waitFor(const std::chrono::duration<Rep, Period>& duration) const {
    if (!future_.valid()) {
      return false;
    }
    return future_.wait_for(duration) == std::future_status::ready;
  }

  R get() { return future_.get(); }

private:
  std::future<R> future_;
};

/**
 * @brief Specialization of TaskHandle for void-returning tasks.
 */
template <>
class TaskHandle<void> {
public:
  TaskHandle() = default;
  explicit TaskHandle(std::future<void> fut) : future_(std::move(fut)) {}

  TaskHandle(TaskHandle&&) noexcept = default;
  TaskHandle& operator=(TaskHandle&&) noexcept = default;

  TaskHandle(const TaskHandle&) = delete;
  TaskHandle& operator=(const TaskHandle&) = delete;

  [[nodiscard]] bool valid() const noexcept { return future_.valid(); }

  [[nodiscard]] bool isReady() const noexcept {
    return future_.valid() &&
           future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
  }

  void wait() const {
    if (future_.valid()) {
      future_.wait();
    }
  }

  template <typename Rep, typename Period>
  [[nodiscard]] bool waitFor(const std::chrono::duration<Rep, Period>& duration) const {
    if (!future_.valid()) {
      return false;
    }
    return future_.wait_for(duration) == std::future_status::ready;
  }

  void get() { future_.get(); }

private:
  std::future<void> future_;
};

/**
 * @struct ThreadBudget
 * @brief Thread partition budgeting to avoid CPU oversubscription with ONNX
 * Runtime.
 */
struct ThreadBudget {
  size_t pyreth_worker_threads{0};
  size_t ort_intra_op_threads{1};
  size_t ort_inter_op_threads{1};
};

/**
 * @struct ThreadPoolConfig
 * @brief Configuration parameters for ThreadPool creation.
 */
struct ThreadPoolConfig {
  size_t num_threads{0};              ///< 0 = auto-detect hardware concurrency
  size_t queue_capacity{1024};        ///< Local queue capacity per worker
  size_t reserve_threads_for_ort{1};  ///< Cores budgeted for ONNX Runtime intra-op
};

/**
 * @struct ThreadPoolMetrics
 * @brief Runtime execution statistics for the ThreadPool.
 */
struct ThreadPoolMetrics {
  uint64_t tasks_submitted{0};
  uint64_t tasks_executed{0};
  uint64_t tasks_stolen{0};
};

// Forward declaration of internal worker structure
struct WorkerData;

/**
 * @class ThreadPool
 * @brief High-throughput lock-free work-stealing thread pool.
 *
 * Implements cooperative cancellation using std::jthread, per-worker Chase-Lev
 * work-stealing queues, and adaptive worker parking to minimize scheduling
 * latency.
 */
class PYRETH_API ThreadPool {
public:
  /**
   * @brief Construct ThreadPool with explicit worker thread count.
   *
   * @param num_threads Number of worker threads (0 = hardware concurrency).
   */
  explicit ThreadPool(size_t num_threads = 0);

  /**
   * @brief Construct ThreadPool with custom configuration.
   *
   * @param config Configuration parameters.
   */
  explicit ThreadPool(const ThreadPoolConfig& config);

  /**
   * @brief Destructor. Cooperatively signals shutdown and joins all workers.
   */
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  /**
   * @brief Submit a callable task returning a Result<TaskHandle<R>>.
   *
   * @tparam F Callable type.
   * @tparam Args Argument types.
   * @param f Function or callable to execute asynchronously.
   * @param args Arguments to pass to f.
   * @return Result containing TaskHandle on success, or Error if shutdown/full.
   */
  template <typename F, typename... Args>
  auto submit(F&& f, Args&&... args)
      -> Result<TaskHandle<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>> {
    using ReturnType = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;

    if (isShutdown()) {
      return Result<TaskHandle<ReturnType>>::failure(StatusCode::INTERNAL_ERROR,
                                                     "ThreadPool has been shut down");
    }

    auto bound = [func = std::forward<F>(f),
                  ... captured_args = std::forward<Args>(args)]() mutable -> ReturnType {
      return std::invoke(func, std::move(captured_args)...);
    };

    auto ptask = std::make_shared<std::packaged_task<ReturnType()>>(std::move(bound));
    auto fut = ptask->get_future();

    MoveOnlyTask task([ptask]() { (*ptask)(); });

    if (!pushTask(std::move(task))) {
      return Result<TaskHandle<ReturnType>>::failure(StatusCode::QUEUE_FULL,
                                                     "ThreadPool task queue is full");
    }

    return Result<TaskHandle<ReturnType>>::success(TaskHandle<ReturnType>(std::move(fut)));
  }

  /**
   * @brief Submit a fire-and-forget callable task.
   *
   * @param f Callable to execute.
   * @return Status::ok() on success, or error on failure.
   */
  template <typename F>
  Status submitDetached(F&& f) {
    if (isShutdown()) {
      return Error(StatusCode::INTERNAL_ERROR, "ThreadPool has been shut down");
    }

    MoveOnlyTask task(std::forward<F>(f));
    if (!pushTask(std::move(task))) {
      return Error::queueFull("ThreadPool task queue is full");
    }

    return Status::ok();
  }

  /**
   * @brief Cooperatively request all worker threads to stop and unpark them.
   */
  void shutdown() noexcept;

  /**
   * @brief Check whether the thread pool has been shut down.
   */
  [[nodiscard]] bool isShutdown() const noexcept;

  /**
   * @brief Query the number of active worker threads.
   */
  [[nodiscard]] size_t numWorkers() const noexcept;

  /**
   * @brief Query runtime execution metrics.
   */
  [[nodiscard]] ThreadPoolMetrics metrics() const noexcept;

  /**
   * @brief Calculate recommended thread budget partitioned between Pyreth and
   * ORT.
   *
   * @param reserved_for_ort Number of hardware cores reserved for ORT intra-op
   * threads.
   * @return Partitioned ThreadBudget.
   */
  [[nodiscard]] static ThreadBudget computeThreadBudget(size_t reserved_for_ort = 1) noexcept;

private:
  bool pushTask(MoveOnlyTask task);
  void notifyOneWorker() noexcept;
  std::optional<MoveOnlyTask> findTask(size_t worker_id);
  void workerLoop(size_t worker_id, std::stop_token stop_token);

  std::vector<std::unique_ptr<WorkerData>> workers_data_;
  std::vector<std::jthread> workers_;

  std::atomic<bool> shutdown_{false};
  std::atomic<size_t> next_unpark_idx_{0};
  std::atomic<uint64_t> tasks_submitted_{0};

  // Global injection queue for external non-worker submissions
  std::mutex injection_mutex_;
  std::queue<MoveOnlyTask> injection_queue_;
};

}  // namespace pyreth
