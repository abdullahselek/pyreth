#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <pyreth/common.hpp>
#include <semaphore>

namespace pyreth {

/**
 * @class WorkerParker
 * @brief High-performance thread parking and notification primitive.
 *
 * Provides cooperative worker thread parking (sleep/wakeup) to eliminate
 * busy-spinning when idle. Implements a 3-state token machine (EMPTY, PARKED, NOTIFIED)
 * backed by std::binary_semaphore to guarantee:
 *  1. No lost wakeups: unpark() prior to park() permits park() to return immediately.
 *  2. Idempotent unpark(): multiple concurrent unparks do not overflow semaphore permits.
 *  3. Timed parking: parkFor() with arbitrary std::chrono durations.
 *  4. Zero dynamic allocation and lock-free state transitions.
 */
class WorkerParker {
public:
  WorkerParker() : sem_(0), state_(EMPTY) {}
  ~WorkerParker() = default;

  // Non-copyable and non-movable
  WorkerParker(const WorkerParker&) = delete;
  WorkerParker& operator=(const WorkerParker&) = delete;
  WorkerParker(WorkerParker&&) = delete;
  WorkerParker& operator=(WorkerParker&&) = delete;

  /**
   * @brief Unpark the worker thread.
   *
   * Idempotent: can be called multiple times safely. If the worker is already
   * parked, it is awakened. If the worker is not yet parked, a token is stored
   * so its next call to park() returns immediately without blocking.
   */
  void unpark() noexcept {
    auto prev = state_.exchange(NOTIFIED, std::memory_order_release);
    if (prev == PARKED) {
      sem_.release();
    }
  }

  /**
   * @brief Park the calling worker thread indefinitely until unparked.
   *
   * If a permit is already available (via a preceding unpark() call),
   * consumes the permit and returns immediately without blocking.
   */
  void park() noexcept {
    if (state_.exchange(EMPTY, std::memory_order_acquire) == NOTIFIED) {
      return;
    }
    int8_t expected = EMPTY;
    if (state_.compare_exchange_strong(expected, PARKED, std::memory_order_relaxed)) {
      sem_.acquire();
      state_.store(EMPTY, std::memory_order_relaxed);
    } else {
      // Concurrently unparked
      state_.store(EMPTY, std::memory_order_relaxed);
    }
  }

  /**
   * @brief Park the calling worker thread with a timeout.
   *
   * @tparam Rep Duration representation type.
   * @tparam Period Duration period type.
   * @param duration Maximum duration to block.
   * @return true if unparked; false if timed out.
   */
  template <typename Rep, typename Period>
  bool parkFor(const std::chrono::duration<Rep, Period>& duration) noexcept {
    if (state_.exchange(EMPTY, std::memory_order_acquire) == NOTIFIED) {
      return true;
    }
    int8_t expected = EMPTY;
    if (state_.compare_exchange_strong(expected, PARKED, std::memory_order_relaxed)) {
      if (sem_.try_acquire_for(duration)) {
        state_.store(EMPTY, std::memory_order_relaxed);
        return true;
      }
      // Timed out: attempt transition back from PARKED to EMPTY
      expected = PARKED;
      if (!state_.compare_exchange_strong(expected, EMPTY, std::memory_order_relaxed)) {
        // Unpark happened right at timeout deadline; consume released permit
        sem_.acquire();
        state_.store(EMPTY, std::memory_order_relaxed);
        return true;
      }
      return false;
    } else {
      // Concurrently unparked
      state_.store(EMPTY, std::memory_order_relaxed);
      return true;
    }
  }

  /**
   * @brief Reset notification state to EMPTY.
   */
  void reset() noexcept { state_.store(EMPTY, std::memory_order_relaxed); }

  /**
   * @brief Check if the worker is currently in the PARKED state.
   */
  [[nodiscard]] bool isParked() const noexcept {
    return state_.load(std::memory_order_relaxed) == PARKED;
  }

private:
  static constexpr int8_t EMPTY = 0;
  static constexpr int8_t PARKED = 1;
  static constexpr int8_t NOTIFIED = 2;

  std::binary_semaphore sem_{0};
  std::atomic<int8_t> state_{EMPTY};
};

/**
 * @class AtomicNotification
 * @brief Zero-overhead notification primitive leveraging C++20 atomic wait/notify.
 *
 * Useful for event-based worker wakeups and broadcast shutdown notifications.
 */
class AtomicNotification {
public:
  AtomicNotification() : count_{0} {}
  ~AtomicNotification() = default;

  AtomicNotification(const AtomicNotification&) = delete;
  AtomicNotification& operator=(const AtomicNotification&) = delete;
  AtomicNotification(AtomicNotification&&) = delete;
  AtomicNotification& operator=(AtomicNotification&&) = delete;

  /**
   * @brief Wake up one thread currently waiting on this notification.
   */
  void notifyOne() noexcept {
    count_.fetch_add(1, std::memory_order_release);
    count_.notify_one();
  }

  /**
   * @brief Wake up all threads currently waiting on this notification.
   */
  void notifyAll() noexcept {
    count_.fetch_add(1, std::memory_order_release);
    count_.notify_all();
  }

  /**
   * @brief Wait until count_ differs from expected value.
   *
   * @param expected The expected value that blocks the caller until changed.
   */
  void wait(uint32_t expected) noexcept { count_.wait(expected, std::memory_order_acquire); }

  /**
   * @brief Query current notification counter value.
   */
  [[nodiscard]] uint32_t get() const noexcept { return count_.load(std::memory_order_acquire); }

private:
  std::atomic<uint32_t> count_{0};
};

}  // namespace pyreth
