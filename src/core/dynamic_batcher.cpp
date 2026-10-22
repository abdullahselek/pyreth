#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <pyreth/dynamic_batcher.hpp>
#include <thread>
#include <utility>
#include <vector>

namespace pyreth {

namespace {

struct PendingRequest {
  std::vector<Tensor> inputs;
  size_t batch_size{1};
  std::chrono::steady_clock::time_point arrival_time;
  std::promise<Result<std::vector<Tensor>>> promise;
};

[[nodiscard]] bool hasMatchingTrailingDims(const PendingRequest& a,
                                           const PendingRequest& b) noexcept {
  if (a.inputs.size() != b.inputs.size()) {
    return false;
  }
  for (size_t i = 0; i < a.inputs.size(); ++i) {
    const Shape& sa = a.inputs[i].shape();
    const Shape& sb = b.inputs[i].shape();
    if (sa.rank() != sb.rank()) {
      return false;
    }
    for (size_t d = 1; d < sa.rank(); ++d) {
      if (sa[d] != sb[d]) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

struct DynamicBatcher::Impl {
  InferenceSession* session{nullptr};
  DynamicBatcherConfig config;

  std::unique_ptr<ThreadPool> owned_pool;
  ThreadPool* pool{nullptr};

  std::unique_ptr<MemoryPool> owned_memory_pool;
  MemoryPool* memory_pool{nullptr};

  size_t effective_max_batch_size{16};

  mutable std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable done_cv;
  std::deque<PendingRequest> pending_queue;
  size_t pending_samples{0};
  size_t in_flight_batches{0};
  bool flush_requested{false};
  std::atomic<bool> shutdown_flag{false};

  // Atomic telemetry counters
  std::atomic<uint64_t> requests_submitted{0};
  std::atomic<uint64_t> requests_completed{0};
  std::atomic<uint64_t> requests_rejected{0};
  std::atomic<uint64_t> batches_dispatched{0};
  std::atomic<uint64_t> full_batches{0};
  std::atomic<uint64_t> timeout_batches{0};
  std::atomic<uint64_t> total_batched_samples{0};
  std::atomic<uint64_t> total_batch_wait_us{0};
  std::atomic<uint64_t> total_execution_us{0};

  std::jthread scheduler_thread;

  Impl(InferenceSession& sess, ThreadPool* ext_pool, MemoryPool* ext_mem_pool,
       const DynamicBatcherConfig& cfg)
      : session(&sess), config(cfg) {
    if (config.max_batch_size == 0) {
      config.max_batch_size = 1;
    }
    if (config.max_queue_capacity == 0) {
      config.max_queue_capacity = 1;
    }
    if (config.arena_block_size == 0) {
      config.arena_block_size = DEFAULT_BLOCK_SIZE;
    }

    effective_max_batch_size = config.max_batch_size;
    if (session->isValid() && !session->getInputs().empty()) {
      const auto& first_in = session->getInputs()[0];
      if (!first_in.shape.empty() && first_in.shape[0] > 0) {
        effective_max_batch_size =
            std::min(effective_max_batch_size, static_cast<size_t>(first_in.shape[0]));
      }
    }

    if (ext_pool != nullptr) {
      pool = ext_pool;
    } else {
      ThreadPoolConfig pool_cfg;
      pool_cfg.num_threads = config.num_workers;
      pool_cfg.reserve_threads_for_ort =
          static_cast<size_t>(std::max(1, session->getConfig().intra_op_num_threads));
      owned_pool = std::make_unique<ThreadPool>(pool_cfg);
      pool = owned_pool.get();
    }

    if (ext_mem_pool != nullptr) {
      memory_pool = ext_mem_pool;
    } else {
      const size_t initial_arenas = std::max<size_t>(4, pool->numWorkers() * 2);
      owned_memory_pool = std::make_unique<MemoryPool>(initial_arenas, config.arena_block_size);
      memory_pool = owned_memory_pool.get();
    }

    scheduler_thread = std::jthread([this](std::stop_token st) { schedulerLoop(std::move(st)); });
  }

  ~Impl() { stopAndDrain(); }

  void stopAndDrain() noexcept {
    bool expected = false;
    if (!shutdown_flag.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
      // Already shut down; ensure thread and in-flight tasks have finished
      if (scheduler_thread.joinable()) {
        scheduler_thread.join();
      }
      std::unique_lock<std::mutex> lock(mutex);
      done_cv.wait(lock, [this] { return pending_queue.empty() && in_flight_batches == 0; });
      return;
    }

    {
      std::lock_guard<std::mutex> lock(mutex);
      cv.notify_all();
    }

    if (scheduler_thread.joinable()) {
      scheduler_thread.request_stop();
      scheduler_thread.join();
    }

    std::unique_lock<std::mutex> lock(mutex);
    done_cv.wait(lock, [this] { return pending_queue.empty() && in_flight_batches == 0; });
  }

  [[nodiscard]] bool canFlushImmediatelyLocked() const noexcept {
    if (pending_queue.empty()) {
      return false;
    }
    if (shutdown_flag.load(std::memory_order_acquire) || flush_requested ||
        config.effectiveBatchDelay().count() == 0) {
      return true;
    }

    const PendingRequest& first = pending_queue.front();
    size_t accumulated = first.batch_size;
    if (accumulated >= effective_max_batch_size) {
      return true;
    }

    for (size_t i = 1; i < pending_queue.size(); ++i) {
      const PendingRequest& req = pending_queue[i];
      if (!hasMatchingTrailingDims(first, req)) {
        return true;
      }
      if (accumulated + req.batch_size > effective_max_batch_size) {
        return true;
      }
      accumulated += req.batch_size;
      if (accumulated >= effective_max_batch_size) {
        return true;
      }
    }

    return false;
  }

  void schedulerLoop([[maybe_unused]] std::stop_token stop_token) {
    while (true) {
      std::vector<PendingRequest> batch;
      size_t batch_samples = 0;
      bool reached_full_batch = false;

      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] {
          return !pending_queue.empty() || shutdown_flag.load(std::memory_order_acquire);
        });

        if (pending_queue.empty() && shutdown_flag.load(std::memory_order_acquire)) {
          break;
        }

        if (!canFlushImmediatelyLocked()) {
          const auto deadline = pending_queue.front().arrival_time + config.effectiveBatchDelay();
          cv.wait_until(lock, deadline, [this] { return canFlushImmediatelyLocked(); });
        }

        if (pending_queue.empty()) {
          continue;
        }

        batch.reserve(std::min(pending_queue.size(), effective_max_batch_size));
        batch_samples = pending_queue.front().batch_size;
        batch.push_back(std::move(pending_queue.front()));
        pending_queue.pop_front();

        while (!pending_queue.empty()) {
          const PendingRequest& next = pending_queue.front();
          if (!hasMatchingTrailingDims(batch.front(), next)) {
            break;
          }
          if (batch_samples + next.batch_size > effective_max_batch_size) {
            reached_full_batch = true;
            break;
          }
          batch_samples += next.batch_size;
          batch.push_back(std::move(pending_queue.front()));
          pending_queue.pop_front();
          if (batch_samples >= effective_max_batch_size) {
            reached_full_batch = true;
            break;
          }
        }

        if (batch_samples >= effective_max_batch_size) {
          reached_full_batch = true;
        }

        pending_samples -= batch_samples;
        ++in_flight_batches;
      }

      dispatchBatch(std::move(batch), batch_samples, reached_full_batch);
    }
  }

  void dispatchBatch(std::vector<PendingRequest>&& batch, size_t batch_samples,
                     bool is_full_batch) {
    auto shared_batch = std::make_shared<std::vector<PendingRequest>>(std::move(batch));
    auto status = pool->submitDetached([this, shared_batch, batch_samples, is_full_batch]() {
      executeBatch(*shared_batch, batch_samples, is_full_batch);
    });

    if (!status.isOk()) {
      // Fallback to inline execution if the worker pool queue is full or shut down
      executeBatch(*shared_batch, batch_samples, is_full_batch);
    }
  }

  void failBatch(std::vector<PendingRequest>& batch, const Error& err, uint64_t wait_us) noexcept {
    for (auto& req : batch) {
      req.inputs.clear();
    }
    batches_dispatched.fetch_add(1, std::memory_order_release);
    total_batch_wait_us.fetch_add(wait_us, std::memory_order_release);
    requests_completed.fetch_add(batch.size(), std::memory_order_release);
    for (auto& req : batch) {
      req.promise.set_value(Result<std::vector<Tensor>>::failure(err));
    }
    finishInFlightBatch();
  }

  void finishInFlightBatch() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    --in_flight_batches;
    if (pending_queue.empty() && in_flight_batches == 0) {
      done_cv.notify_all();
    }
  }

  void executeBatch(std::vector<PendingRequest>& batch, size_t total_batch_samples,
                    bool is_full_batch) noexcept {
    const auto exec_start = std::chrono::steady_clock::now();
    const uint64_t wait_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(exec_start - batch[0].arrival_time)
            .count());

    const size_t num_inputs = batch[0].inputs.size();
    auto input_lease = memory_pool->acquire();
    std::vector<Tensor> batched_inputs;
    batched_inputs.reserve(num_inputs);

    // 1. Concatenate individual request input tensors into a single contiguous arena tensor
    for (size_t in_idx = 0; in_idx < num_inputs; ++in_idx) {
      const Tensor& proto = batch[0].inputs[in_idx];
      std::vector<int64_t> batched_dims = proto.shape().dims();
      batched_dims[0] = static_cast<int64_t>(total_batch_samples);
      Shape batched_shape(std::move(batched_dims));

      auto batched_res = Tensor::create(batched_shape, proto.dtype(), *input_lease);
      if (!batched_res.isOk()) {
        failBatch(batch, batched_res.error(), wait_us);
        return;
      }

      Tensor batched_tensor = std::move(batched_res).value();
      auto* dst_bytes = static_cast<std::byte*>(batched_tensor.data());
      size_t byte_offset = 0;

      for (const auto& req : batch) {
        const Tensor& src = req.inputs[in_idx];
        const size_t req_bytes = src.totalBytes();
        if (req_bytes == 0) {
          continue;
        }
        if (src.isContiguous()) {
          std::memcpy(dst_bytes + byte_offset, src.data(), req_bytes);
        } else {
          auto contig_res = src.toContiguous(*input_lease);
          if (!contig_res.isOk()) {
            failBatch(batch, contig_res.error(), wait_us);
            return;
          }
          std::memcpy(dst_bytes + byte_offset, contig_res.value().data(), req_bytes);
        }
        byte_offset += req_bytes;
      }

      batched_inputs.push_back(std::move(batched_tensor));
    }

    // Release per-request input references now that they are concatenated into batched_inputs
    for (auto& req : batch) {
      req.inputs.clear();
    }

    // 2. Dispatch a single batched inference run with outputs backed by the MemoryPool
    auto run_res = session->run(std::span<const Tensor>(batched_inputs), *memory_pool);
    batched_inputs.clear();
    {
      auto drop_lease = std::move(input_lease);
      (void)drop_lease;
    }

    const auto exec_end = std::chrono::steady_clock::now();
    const uint64_t exec_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(exec_end - exec_start).count());

    if (!run_res.isOk()) {
      total_execution_us.fetch_add(exec_us, std::memory_order_release);
      failBatch(batch, run_res.error(), wait_us);
      return;
    }

    // 3. Demultiplex batched output tensors back to individual request callers (zero-copy slice)
    const std::vector<Tensor>& batched_outputs = run_res.value();
    const size_t num_outputs = batched_outputs.size();
    int64_t row_offset = 0;

    std::vector<Result<std::vector<Tensor>>> request_results;
    request_results.reserve(batch.size());

    for (auto& req : batch) {
      const auto req_batch = static_cast<int64_t>(req.batch_size);
      std::vector<Tensor> req_outputs;
      req_outputs.reserve(num_outputs);
      Error slice_err = Error::ok();

      for (size_t out_idx = 0; out_idx < num_outputs; ++out_idx) {
        const Tensor& bout = batched_outputs[out_idx];
        if (bout.rank() >= 1 && bout.shape()[0] == static_cast<int64_t>(total_batch_samples)) {
          auto slice_res = bout.slice(0, row_offset, row_offset + req_batch, 1);
          if (!slice_res.isOk()) {
            slice_err = slice_res.error();
            break;
          }
          req_outputs.push_back(std::move(slice_res).value());
        } else {
          req_outputs.push_back(bout);
        }
      }

      row_offset += req_batch;
      if (slice_err.isOk()) {
        request_results.push_back(Result<std::vector<Tensor>>::success(std::move(req_outputs)));
      } else {
        request_results.push_back(Result<std::vector<Tensor>>::failure(slice_err));
      }
    }

    // 4. Record batch telemetry BEFORE fulfilling request promises so callers waking from
    // TaskHandle::get() / wait() immediately observe consistent metrics.
    batches_dispatched.fetch_add(1, std::memory_order_release);
    if (is_full_batch) {
      full_batches.fetch_add(1, std::memory_order_release);
    } else {
      timeout_batches.fetch_add(1, std::memory_order_release);
    }
    total_batched_samples.fetch_add(total_batch_samples, std::memory_order_release);
    requests_completed.fetch_add(batch.size(), std::memory_order_release);
    total_batch_wait_us.fetch_add(wait_us, std::memory_order_release);
    total_execution_us.fetch_add(exec_us, std::memory_order_release);

    for (size_t i = 0; i < batch.size(); ++i) {
      batch[i].promise.set_value(std::move(request_results[i]));
    }

    finishInFlightBatch();
  }
};

DynamicBatcher::DynamicBatcher() noexcept : impl_(nullptr) {}

DynamicBatcher::DynamicBatcher(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

DynamicBatcher::DynamicBatcher(InferenceSession& session, const DynamicBatcherConfig& config)
    : impl_(session.isValid() ? std::make_unique<Impl>(session, nullptr, nullptr, config)
                              : nullptr) {}

DynamicBatcher::DynamicBatcher(InferenceSession& session, ThreadPool& pool,
                               const DynamicBatcherConfig& config)
    : impl_(session.isValid() ? std::make_unique<Impl>(session, &pool, nullptr, config) : nullptr) {
}

DynamicBatcher::DynamicBatcher(InferenceSession& session, ThreadPool& pool, MemoryPool& memory_pool,
                               const DynamicBatcherConfig& config)
    : impl_(session.isValid() ? std::make_unique<Impl>(session, &pool, &memory_pool, config)
                              : nullptr) {}

DynamicBatcher::DynamicBatcher(DynamicBatcher&& other) noexcept = default;

DynamicBatcher& DynamicBatcher::operator=(DynamicBatcher&& other) noexcept = default;

DynamicBatcher::~DynamicBatcher() = default;

Result<DynamicBatcher> DynamicBatcher::create(InferenceSession& session,
                                              const DynamicBatcherConfig& config) {
  if (!session.isValid()) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "InferenceSession is uninitialized or invalid");
  }
  if (config.max_batch_size == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_batch_size must be >= 1");
  }
  if (config.max_queue_capacity == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_queue_capacity must be >= 1");
  }
  auto impl = std::make_unique<Impl>(session, nullptr, nullptr, config);
  return Result<DynamicBatcher>::success(DynamicBatcher(std::move(impl)));
}

Result<DynamicBatcher> DynamicBatcher::create(InferenceSession& session, ThreadPool& pool,
                                              const DynamicBatcherConfig& config) {
  if (!session.isValid()) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "InferenceSession is uninitialized or invalid");
  }
  if (pool.isShutdown()) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "ThreadPool has already been shut down");
  }
  if (config.max_batch_size == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_batch_size must be >= 1");
  }
  if (config.max_queue_capacity == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_queue_capacity must be >= 1");
  }
  auto impl = std::make_unique<Impl>(session, &pool, nullptr, config);
  return Result<DynamicBatcher>::success(DynamicBatcher(std::move(impl)));
}

Result<DynamicBatcher> DynamicBatcher::create(InferenceSession& session, ThreadPool& pool,
                                              MemoryPool& memory_pool,
                                              const DynamicBatcherConfig& config) {
  if (!session.isValid()) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "InferenceSession is uninitialized or invalid");
  }
  if (pool.isShutdown()) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "ThreadPool has already been shut down");
  }
  if (config.max_batch_size == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_batch_size must be >= 1");
  }
  if (config.max_queue_capacity == 0) {
    return Result<DynamicBatcher>::failure(StatusCode::INVALID_ARGUMENT,
                                           "DynamicBatcher max_queue_capacity must be >= 1");
  }
  auto impl = std::make_unique<Impl>(session, &pool, &memory_pool, config);
  return Result<DynamicBatcher>::success(DynamicBatcher(std::move(impl)));
}

Result<TaskHandle<Result<std::vector<Tensor>>>> DynamicBatcher::submit(
    std::span<const Tensor> inputs) {
  if (!isValid()) {
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_ARGUMENT, "DynamicBatcher is uninitialized or invalid");
  }

  if (impl_->shutdown_flag.load(std::memory_order_acquire)) {
    impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INTERNAL_ERROR, "DynamicBatcher has been shut down");
  }

  auto val_status = impl_->session->validateInputs(inputs);
  if (!val_status.isOk()) {
    impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(val_status.error());
  }

  if (inputs.empty() || inputs[0].rank() == 0) {
    impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_DIMENSIONS,
        "DynamicBatcher requires input tensors with rank >= 1 (leading batch dimension)");
  }

  const int64_t req_batch = inputs[0].shape()[0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i].rank() == 0 || inputs[i].shape()[0] != req_batch) {
      impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
          StatusCode::INVALID_DIMENSIONS,
          "All inputs in a request must have the same batch size along axis 0");
    }
  }

  if (req_batch <= 0 || static_cast<size_t>(req_batch) > impl_->effective_max_batch_size) {
    impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_DIMENSIONS, "Request batch size exceeds DynamicBatcher max_batch_size");
  }

  std::promise<Result<std::vector<Tensor>>> promise;
  auto fut = promise.get_future();

  PendingRequest req;
  req.inputs.assign(inputs.begin(), inputs.end());
  req.batch_size = static_cast<size_t>(req_batch);
  req.arrival_time = std::chrono::steady_clock::now();
  req.promise = std::move(promise);

  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->shutdown_flag.load(std::memory_order_relaxed)) {
      impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
          StatusCode::INTERNAL_ERROR, "DynamicBatcher has been shut down");
    }

    if (impl_->pending_queue.size() >= impl_->config.max_queue_capacity) {
      impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
          StatusCode::QUEUE_FULL, "DynamicBatcher request queue is full");
    }

    const bool was_empty = impl_->pending_queue.empty();
    impl_->pending_samples += req.batch_size;
    impl_->pending_queue.push_back(std::move(req));
    impl_->requests_submitted.fetch_add(1, std::memory_order_relaxed);

    if (was_empty || impl_->canFlushImmediatelyLocked()) {
      impl_->cv.notify_one();
    }
  }

  return Result<TaskHandle<Result<std::vector<Tensor>>>>::success(
      TaskHandle<Result<std::vector<Tensor>>>(std::move(fut)));
}

Result<TaskHandle<Result<std::vector<Tensor>>>> DynamicBatcher::submit(
    const std::unordered_map<std::string, Tensor>& named_inputs) {
  if (!isValid()) {
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_ARGUMENT, "DynamicBatcher is uninitialized or invalid");
  }

  const auto& model_info = impl_->session->getModelInfo();
  if (named_inputs.size() != model_info.inputs.size()) {
    impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Number of named input tensors does not match model input count");
  }

  std::vector<Tensor> ordered_inputs(model_info.inputs.size());
  for (const auto& [name, tensor] : named_inputs) {
    auto idx_res = model_info.getInputIndex(name);
    if (!idx_res.isOk()) {
      impl_->requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(idx_res.error());
    }
    ordered_inputs[idx_res.value()] = tensor;
  }

  return submit(std::span<const Tensor>(ordered_inputs));
}

Result<std::vector<Tensor>> DynamicBatcher::run(std::span<const Tensor> inputs) {
  auto handle_res = submit(inputs);
  if (!handle_res.isOk()) {
    return Result<std::vector<Tensor>>::failure(handle_res.error());
  }
  return handle_res.value().get();
}

Result<std::vector<Tensor>> DynamicBatcher::run(
    const std::unordered_map<std::string, Tensor>& named_inputs) {
  auto handle_res = submit(named_inputs);
  if (!handle_res.isOk()) {
    return Result<std::vector<Tensor>>::failure(handle_res.error());
  }
  return handle_res.value().get();
}

void DynamicBatcher::flush() {
  if (!isValid()) {
    return;
  }
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (impl_->pending_queue.empty() && impl_->in_flight_batches == 0) {
    return;
  }
  impl_->flush_requested = true;
  impl_->cv.notify_all();
  impl_->done_cv.wait(
      lock, [this] { return impl_->pending_queue.empty() && impl_->in_flight_batches == 0; });
  impl_->flush_requested = false;
}

void DynamicBatcher::shutdown() noexcept {
  if (impl_) {
    impl_->stopAndDrain();
  }
}

bool DynamicBatcher::isShutdown() const noexcept {
  return !impl_ || impl_->shutdown_flag.load(std::memory_order_acquire);
}

bool DynamicBatcher::isValid() const noexcept {
  return impl_ != nullptr && impl_->session != nullptr && impl_->session->isValid();
}

const DynamicBatcherConfig& DynamicBatcher::getConfig() const noexcept {
  static const DynamicBatcherConfig empty_config{};
  return impl_ ? impl_->config : empty_config;
}

size_t DynamicBatcher::pendingCount() const noexcept {
  if (!impl_) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->pending_queue.size();
}

DynamicBatcherMetrics DynamicBatcher::metrics() const noexcept {
  DynamicBatcherMetrics m{};
  if (!impl_) {
    return m;
  }
  m.requests_submitted = impl_->requests_submitted.load(std::memory_order_acquire);
  m.requests_completed = impl_->requests_completed.load(std::memory_order_acquire);
  m.requests_rejected = impl_->requests_rejected.load(std::memory_order_acquire);
  m.batches_dispatched = impl_->batches_dispatched.load(std::memory_order_acquire);
  m.full_batches = impl_->full_batches.load(std::memory_order_acquire);
  m.timeout_batches = impl_->timeout_batches.load(std::memory_order_acquire);
  m.total_batched_samples = impl_->total_batched_samples.load(std::memory_order_acquire);
  m.total_batch_wait_us = impl_->total_batch_wait_us.load(std::memory_order_acquire);
  m.total_execution_us = impl_->total_execution_us.load(std::memory_order_acquire);
  return m;
}

void DynamicBatcher::resetMetrics() noexcept {
  if (!impl_) {
    return;
  }
  impl_->requests_submitted.store(0, std::memory_order_relaxed);
  impl_->requests_completed.store(0, std::memory_order_relaxed);
  impl_->requests_rejected.store(0, std::memory_order_relaxed);
  impl_->batches_dispatched.store(0, std::memory_order_relaxed);
  impl_->full_batches.store(0, std::memory_order_relaxed);
  impl_->timeout_batches.store(0, std::memory_order_relaxed);
  impl_->total_batched_samples.store(0, std::memory_order_relaxed);
  impl_->total_batch_wait_us.store(0, std::memory_order_relaxed);
  impl_->total_execution_us.store(0, std::memory_order_relaxed);
}

}  // namespace pyreth
