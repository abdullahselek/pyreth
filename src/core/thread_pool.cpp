#include <algorithm>
#include <chrono>
#include <pyreth/thread_pool.hpp>

namespace pyreth {

struct WorkerData {
  size_t id{0};
  WorkStealingQueue<MoveOnlyTask, 1024> queue;
  WorkerParker parker;
  std::atomic<uint64_t> tasks_executed{0};
  std::atomic<uint64_t> tasks_stolen{0};
};

static thread_local WorkerData* t_current_worker = nullptr;

ThreadPool::ThreadPool(size_t num_threads)
    : ThreadPool([num_threads]() {
        ThreadPoolConfig config;
        config.num_threads = num_threads;
        return config;
      }()) {}

ThreadPool::ThreadPool(const ThreadPoolConfig& config) {
  size_t count = config.num_threads;
  if (count == 0) {
    auto budget = computeThreadBudget(config.reserve_threads_for_ort);
    count = budget.pyreth_worker_threads;
  }
  if (count == 0) {
    count = 1;
  }

  workers_data_.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    auto wd = std::make_unique<WorkerData>();
    wd->id = i;
    workers_data_.push_back(std::move(wd));
  }

  workers_.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    workers_.emplace_back([this, i](std::stop_token st) { workerLoop(i, st); });
  }
}

ThreadPool::~ThreadPool() {
  shutdown();
}

void ThreadPool::shutdown() noexcept {
  if (shutdown_.exchange(true, std::memory_order_release)) {
    return;
  }

  for (auto& w : workers_) {
    w.request_stop();
  }

  for (auto& wd : workers_data_) {
    wd->parker.unpark();
  }

  // std::jthread automatically joins upon destruction/clearing
  workers_.clear();
}

bool ThreadPool::isShutdown() const noexcept {
  return shutdown_.load(std::memory_order_acquire);
}

size_t ThreadPool::numWorkers() const noexcept {
  return workers_data_.size();
}

ThreadPoolMetrics ThreadPool::metrics() const noexcept {
  ThreadPoolMetrics m;
  m.tasks_submitted = tasks_submitted_.load(std::memory_order_relaxed);

  for (const auto& wd : workers_data_) {
    m.tasks_executed += wd->tasks_executed.load(std::memory_order_relaxed);
    m.tasks_stolen += wd->tasks_stolen.load(std::memory_order_relaxed);
  }

  return m;
}

ThreadBudget ThreadPool::computeThreadBudget(size_t reserved_for_ort) noexcept {
  unsigned int hw = std::thread::hardware_concurrency();
  if (hw == 0) {
    hw = 4;
  }

  ThreadBudget budget;
  budget.ort_intra_op_threads = std::max<size_t>(1, reserved_for_ort);
  budget.ort_inter_op_threads = 1;

  if (hw > budget.ort_intra_op_threads) {
    budget.pyreth_worker_threads = hw - budget.ort_intra_op_threads;
  } else {
    budget.pyreth_worker_threads = 1;
  }

  return budget;
}

bool ThreadPool::pushTask(MoveOnlyTask task) {
  tasks_submitted_.fetch_add(1, std::memory_order_relaxed);

  // Fast path: if submitting from an existing worker, push to local deque
  // (LIFO)
  if (t_current_worker != nullptr) {
    if (t_current_worker->queue.push(std::move(task))) {
      notifyOneWorker();
      return true;
    }
  }

  // Global injection path (external submitters or local queue overflow)
  {
    std::lock_guard<std::mutex> lock(injection_mutex_);
    injection_queue_.push(std::move(task));
  }

  notifyOneWorker();
  return true;
}

void ThreadPool::notifyOneWorker() noexcept {
  if (workers_data_.empty()) {
    return;
  }
  size_t idx = next_unpark_idx_.fetch_add(1, std::memory_order_relaxed) % workers_data_.size();
  workers_data_[idx]->parker.unpark();
}

std::optional<MoveOnlyTask> ThreadPool::findTask(size_t worker_id) {
  // 1. Check local work-stealing queue (LIFO for cache hotness)
  auto task = workers_data_[worker_id]->queue.pop();
  if (task) {
    return task;
  }

  // 2. Try lock on global injection queue
  {
    std::unique_lock<std::mutex> lock(injection_mutex_, std::try_to_lock);
    if (lock.owns_lock() && !injection_queue_.empty()) {
      auto t = std::move(injection_queue_.front());
      injection_queue_.pop();
      return t;
    }
  }

  // 3. Steal from peer workers in round-robin order (FIFO)
  const size_t n = workers_data_.size();
  for (size_t i = 1; i < n; ++i) {
    size_t victim = (worker_id + i) % n;
    auto stolen = workers_data_[victim]->queue.steal();
    if (stolen) {
      workers_data_[worker_id]->tasks_stolen.fetch_add(1, std::memory_order_relaxed);
      return stolen;
    }
  }

  // 4. Double check injection queue with blocking lock before parking
  {
    std::lock_guard<std::mutex> lock(injection_mutex_);
    if (!injection_queue_.empty()) {
      auto t = std::move(injection_queue_.front());
      injection_queue_.pop();
      return t;
    }
  }

  return std::nullopt;
}

void ThreadPool::workerLoop(size_t worker_id, std::stop_token stop_token) {
  t_current_worker = workers_data_[worker_id].get();

  while (!stop_token.stop_requested()) {
    auto task = findTask(worker_id);
    if (task) {
      (*task)();
      workers_data_[worker_id]->tasks_executed.fetch_add(1, std::memory_order_relaxed);
    } else {
      if (stop_token.stop_requested()) {
        break;
      }
      // Cooperatively park for a short window to eliminate 100% CPU spinning
      workers_data_[worker_id]->parker.parkFor(std::chrono::milliseconds(2));
    }
  }

  // Drain remaining tasks during shutdown
  while (auto task = findTask(worker_id)) {
    (*task)();
    workers_data_[worker_id]->tasks_executed.fetch_add(1, std::memory_order_relaxed);
  }

  t_current_worker = nullptr;
}

}  // namespace pyreth
