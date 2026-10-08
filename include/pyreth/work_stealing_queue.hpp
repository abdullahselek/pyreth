#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <pyreth/common.hpp>
#include <utility>

namespace pyreth {

/**
 * @class WorkStealingQueue
 * @brief Lock-free Single-Producer Multi-Consumer (SPMC) bounded work-stealing
 * deque.
 *
 * Implements the Chase-Lev work-stealing algorithm with C++20 memory ordering
 * based on the formal memory model proofs by Lê et al. (PPoPP 2013).
 *
 * Key Characteristics:
 *  - Single Producer (Worker Thread):
 *      - `push()`: Inserts item at the bottom (LIFO).
 *      - `pop()`: Extracts item from the bottom (LIFO), exploiting CPU cache
 * locality.
 *  - Multiple Consumers (Thief Threads):
 *      - `steal()`: Extracts item from the top (FIFO), taking the oldest task
 * chunks.
 *  - Bounded Circular Buffer:
 *      - Fixed power-of-two capacity avoiding dynamic memory reclamation or
 * hazard pointers.
 *      - Wraparound guard ensuring safe reuse of slots without data races.
 *  - Supports move-only types (e.g., std::unique_ptr, move-only tasks).
 *  - False sharing prevention: `top_` and `bottom_` aligned to separate 64-byte
 * cache lines.
 *
 * @tparam T Element type stored in the queue.
 * @tparam Capacity Power-of-two capacity of the circular buffer (default:
 * 1024).
 */
template <typename T, size_t Capacity = 1024>
class WorkStealingQueue {
  static_assert(Capacity > 0, "Capacity must be greater than zero");
  static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");

  static constexpr int64_t MASK = static_cast<int64_t>(Capacity - 1);

  struct Slot {
    std::atomic<bool> ready{false};
    std::optional<T> item{std::nullopt};
  };

public:
  using ValueType = T;

  WorkStealingQueue() : buffer_(new Slot[Capacity]) {}

  ~WorkStealingQueue() { clear(); }

  // Non-copyable and non-movable (pinned concurrent data structure)
  WorkStealingQueue(const WorkStealingQueue&) = delete;
  WorkStealingQueue& operator=(const WorkStealingQueue&) = delete;
  WorkStealingQueue(WorkStealingQueue&&) = delete;
  WorkStealingQueue& operator=(WorkStealingQueue&&) = delete;

  /**
   * @brief Push an item onto the bottom of the deque (Producer/Owner only).
   *
   * @param item Value or rvalue to push.
   * @return true if pushed successfully; false if queue is full.
   */
  bool push(const T& item) { return emplace(item); }

  /**
   * @brief Push an rvalue item onto the bottom of the deque (Producer/Owner
   * only).
   *
   * @param item Rvalue to move into the queue.
   * @return true if pushed successfully; false if queue is full.
   */
  bool push(T&& item) { return emplace(std::move(item)); }

  /**
   * @brief Emplace an item onto the bottom of the deque (Producer/Owner only).
   *
   * @tparam Args Constructor argument types for T.
   * @param args Arguments forwarded to T's constructor.
   * @return true if pushed successfully; false if queue is full.
   */
  template <typename... Args>
  bool emplace(Args&&... args) {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_acquire);

    if (b - t >= static_cast<int64_t>(Capacity)) {
      return false;  // Queue is full
    }

    Slot& slot = buffer_[static_cast<size_t>(b) & static_cast<size_t>(MASK)];

    // Wraparound safety check: verify slot was emptied by previous thief/pop
    if (slot.ready.load(std::memory_order_acquire)) {
      return false;  // Slot still held by an active thief
    }

    slot.item.emplace(std::forward<Args>(args)...);
    slot.ready.store(true, std::memory_order_release);

    // Release fence ensures slot content is visible before bottom_ is
    // incremented
    bottom_.store(b + 1, std::memory_order_release);
    return true;
  }

  /**
   * @brief Pop an item from the bottom of the deque in LIFO order
   * (Producer/Owner only).
   *
   * @return std::optional containing the popped item, or std::nullopt if queue
   * is empty or if the last element was lost to a concurrent stealer.
   */
  std::optional<T> pop() {
    int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);

    // Seq_cst fence prevents Store-Load reordering between bottom_ store and
    // top_ load
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top_.load(std::memory_order_relaxed);

    if (t <= b) {
      // Non-empty queue
      if (t == b) {
        // Last element: owner races with concurrent thieves claiming slot t
        if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                          std::memory_order_relaxed)) {
          // Lost race to a thief
          bottom_.store(b + 1, std::memory_order_relaxed);
          return std::nullopt;
        }
        // Won the race
        bottom_.store(b + 1, std::memory_order_relaxed);
      }

      Slot& slot = buffer_[static_cast<size_t>(b) & static_cast<size_t>(MASK)];
      T val = std::move(*slot.item);
      slot.item.reset();
      slot.ready.store(false, std::memory_order_release);
      return val;
    } else {
      // Queue is empty
      bottom_.store(b + 1, std::memory_order_relaxed);
      return std::nullopt;
    }
  }

  /**
   * @brief Steal an item from the top of the deque in FIFO order
   * (Consumers/Thieves).
   *
   * @return std::optional containing the stolen item, or std::nullopt if queue
   * is empty or contention was lost to another thief or owner pop.
   */
  std::optional<T> steal() {
    int64_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t b = bottom_.load(std::memory_order_acquire);

    if (t < b) {
      Slot& slot = buffer_[static_cast<size_t>(t) & static_cast<size_t>(MASK)];

      // Compete with other thieves and owner pop
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst,
                                        std::memory_order_relaxed)) {
        return std::nullopt;  // Contention lost
      }

      // Winning thief extracts the item
      while (!slot.ready.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
        __builtin_ia32_pause();
#endif
#elif defined(__aarch64__)
        asm volatile("yield");
#endif
      }

      T val = std::move(*slot.item);
      slot.item.reset();
      slot.ready.store(false, std::memory_order_release);
      return val;
    }

    return std::nullopt;
  }

  /**
   * @brief Query the approximate number of items currently in the deque.
   *
   * Accurate for the owner thread when no concurrent stealing occurs.
   * Approximate lock-free snapshot for other threads.
   */
  [[nodiscard]] size_t size() const noexcept {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return (b > t) ? static_cast<size_t>(b - t) : 0;
  }

  /**
   * @brief Query whether the deque is currently empty.
   */
  [[nodiscard]] bool empty() const noexcept {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_relaxed);
    return b <= t;
  }

  /**
   * @brief Maximum capacity of the bounded deque.
   */
  [[nodiscard]] static constexpr size_t capacity() noexcept { return Capacity; }

  /**
   * @brief Drain all items from the deque (Producer/Owner only).
   */
  void clear() noexcept {
    while (pop().has_value()) {
      // Drains and destructs all remaining items
    }
  }

private:
  // Top pointer (stolen by thieves, read/updated during last-item race by
  // owner)
  alignas(64) std::atomic<int64_t> top_{0};

  // Bottom pointer (exclusively updated by owner, read by thieves)
  alignas(64) std::atomic<int64_t> bottom_{0};

  // Circular slot array
  alignas(64) std::unique_ptr<Slot[]> buffer_;
};

}  // namespace pyreth
