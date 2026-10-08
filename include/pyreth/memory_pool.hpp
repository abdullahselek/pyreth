#pragma once

#include <cstddef>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <utility>
#include <vector>

#include <pyreth/common.hpp>
#include <pyreth/result.hpp>

namespace pyreth {

inline constexpr size_t DEFAULT_ALIGNMENT = 64;
inline constexpr size_t DEFAULT_BLOCK_SIZE = 1024 * 1024; // 1 MB initial block

/**
 * @class ArenaAllocator
 * @brief High-performance 64-byte aligned SIMD bump allocator with O(1) reset.
 *
 * Designed for low-latency tensor allocations and request-scoped inference
 * scratchpads. Guarantees that all allocations are 64-byte aligned for optimal
 * SIMD vectorization (AVX-512, ARM NEON, Apple Silicon unified cache lines).
 *
 * Implements std::pmr::memory_resource to enable seamless integration with
 * standard polymorphic memory resource containers (std::pmr::vector, etc.).
 */
class PYRETH_API ArenaAllocator : public std::pmr::memory_resource {
public:
  /**
   * @brief Construct an ArenaAllocator with a specified default block size.
   *
   * @param block_size Size of the initial contiguous memory block (default 1
   * MB).
   */
  explicit ArenaAllocator(size_t block_size = DEFAULT_BLOCK_SIZE);

  /**
   * @brief Destructor. Releases all allocated backing blocks back to the OS.
   */
  ~ArenaAllocator() override;

  ArenaAllocator(const ArenaAllocator &) = delete;
  ArenaAllocator &operator=(const ArenaAllocator &) = delete;

  ArenaAllocator(ArenaAllocator &&other) noexcept;
  ArenaAllocator &operator=(ArenaAllocator &&other) noexcept;

  /**
   * @brief Allocate a 64-byte aligned buffer of specified size.
   *
   * @param bytes Number of bytes to allocate.
   * @param alignment Alignment boundary (minimum and default: 64 bytes).
   * @return Pointer to 64-byte aligned memory. Throws std::bad_alloc on
   * failure.
   */
  [[nodiscard]] void *allocate(size_t bytes,
                               size_t alignment = DEFAULT_ALIGNMENT);

  /**
   * @brief Exception-free monadic allocation for low-latency hot paths.
   *
   * @param bytes Number of bytes to allocate.
   * @param alignment Alignment boundary (minimum and default: 64 bytes).
   * @return Result containing 64-byte aligned pointer or
   * StatusCode::ALLOCATION_FAILED.
   */
  [[nodiscard]] Result<void *>
  tryAllocate(size_t bytes, size_t alignment = DEFAULT_ALIGNMENT) noexcept;

  /**
   * @brief Individual deallocations are no-ops in ArenaAllocator.
   *
   * Memory is reclaimed in bulk via reset() or destruction.
   */
  void deallocate(void *p, size_t bytes,
                  size_t alignment = DEFAULT_ALIGNMENT) noexcept;

  /**
   * @brief Construct an object of type T directly in arena memory.
   *
   * @tparam T Type to construct.
   * @tparam Args Constructor argument types.
   * @param args Arguments forwarded to T's constructor.
   * @return Pointer to newly constructed object.
   */
  template <typename T, typename... Args> T *create(Args &&...args) {
    size_t align = std::max(alignof(T), DEFAULT_ALIGNMENT);
    void *mem = allocate(sizeof(T), align);
    return new (mem) T(std::forward<Args>(args)...);
  }

  /**
   * @brief Allocate a contiguous typed array in arena memory.
   *
   * @tparam T Element type.
   * @param count Number of elements.
   * @param alignment Alignment boundary (minimum 64 bytes).
   * @return Pointer to contiguous array memory.
   */
  template <typename T>
  T *allocateArray(size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
    size_t align = std::max(alignof(T), alignment);
    return static_cast<T *>(allocate(sizeof(T) * count, align));
  }

  /**
   * @brief Reset the arena in O(1) time.
   *
   * Rewinds the bump pointer to the beginning of the memory block without
   * freeing backing OS memory. Subsequent allocations reuse existing buffers
   * without heap allocation overhead.
   */
  void reset() noexcept;

  /**
   * @brief Total payload bytes currently allocated from this arena.
   */
  [[nodiscard]] size_t bytesAllocated() const noexcept;

  /**
   * @brief Total capacity in bytes currently reserved across all backing
   * blocks.
   */
  [[nodiscard]] size_t bytesReserved() const noexcept;

  /**
   * @brief Number of active memory blocks currently managed by this arena.
   */
  [[nodiscard]] size_t blockCount() const noexcept;

protected:
  void *do_allocate(size_t bytes, size_t alignment) override;
  void do_deallocate(void *p, size_t bytes, size_t alignment) override;
  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override;

private:
  struct Block;

  std::unique_ptr<Block> allocateBlock(size_t capacity);
  void *allocateFromNextBlock(size_t bytes, size_t alignment);

  size_t default_block_size_{DEFAULT_BLOCK_SIZE};
  std::unique_ptr<Block> head_{nullptr};
  Block *current_{nullptr};
  size_t bytes_allocated_{0};
  size_t bytes_reserved_{0};
  size_t block_count_{0};
};

/**
 * @class MemoryPool
 * @brief Thread-safe pool managing reusable ArenaAllocators for inference
 * serving.
 *
 * Provides:
 *  1. Thread-local arenas for zero-contention worker thread scratchpads.
 *  2. Pooled arena leases with automatic O(1) reset upon return.
 */
class PYRETH_API MemoryPool {
public:
  /**
   * @class Lease
   * @brief RAII lease for an ArenaAllocator borrowed from a MemoryPool.
   *
   * Automatically resets the arena in O(1) and returns it to the pool upon
   * destruction.
   */
  class PYRETH_API Lease {
  public:
    Lease() = default;
    Lease(MemoryPool *pool, std::unique_ptr<ArenaAllocator> arena);
    ~Lease();

    Lease(Lease &&other) noexcept;
    Lease &operator=(Lease &&other) noexcept;

    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;

    ArenaAllocator &get() noexcept { return *arena_; }
    ArenaAllocator *operator->() noexcept { return arena_.get(); }
    ArenaAllocator &operator*() noexcept { return *arena_; }
    [[nodiscard]] bool valid() const noexcept { return arena_ != nullptr; }

  private:
    MemoryPool *pool_{nullptr};
    std::unique_ptr<ArenaAllocator> arena_{nullptr};
  };

  /**
   * @brief Construct MemoryPool with pre-allocated arenas.
   *
   * @param initial_count Number of arenas to pre-allocate.
   * @param block_size Default block size for each arena.
   */
  explicit MemoryPool(size_t initial_count = 4,
                      size_t block_size = DEFAULT_BLOCK_SIZE);

  ~MemoryPool();

  MemoryPool(const MemoryPool &) = delete;
  MemoryPool &operator=(const MemoryPool &) = delete;
  MemoryPool(MemoryPool &&) = delete;
  MemoryPool &operator=(MemoryPool &&) = delete;

  /**
   * @brief Acquire an arena lease from the pool.
   *
   * Reuses an available idle arena or creates a new one if pool is exhausted.
   */
  [[nodiscard]] Lease acquire();

  /**
   * @brief Access the persistent thread-local arena for the calling thread.
   *
   * Provides lock-free, zero-contention scratchpad memory for worker threads.
   */
  [[nodiscard]] static ArenaAllocator &getThreadLocalArena();

  /**
   * @brief Number of idle arenas currently available in the pool.
   */
  [[nodiscard]] size_t availableCount() const noexcept;

private:
  friend class Lease;
  void returnArena(std::unique_ptr<ArenaAllocator> arena);

  size_t default_block_size_;
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<ArenaAllocator>> arenas_;
};

/**
 * @class ScopedArena
 * @brief RAII helper managing a scoped per-request arena lease.
 *
 * Automatically resets the arena in O(1) when leaving the inference request
 * scope.
 */
class PYRETH_API ScopedArena {
public:
  explicit ScopedArena(
      ArenaAllocator &arena = MemoryPool::getThreadLocalArena());
  ~ScopedArena();

  ScopedArena(const ScopedArena &) = delete;
  ScopedArena &operator=(const ScopedArena &) = delete;
  ScopedArena(ScopedArena &&) = delete;
  ScopedArena &operator=(ScopedArena &&) = delete;

  ArenaAllocator &arena() noexcept { return arena_; }
  ArenaAllocator *operator->() noexcept { return &arena_; }
  ArenaAllocator &operator*() noexcept { return arena_; }

  [[nodiscard]] void *allocate(size_t bytes,
                               size_t alignment = DEFAULT_ALIGNMENT) {
    return arena_.allocate(bytes, alignment);
  }

  template <typename T, typename... Args> T *create(Args &&...args) {
    return arena_.create<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  T *allocateArray(size_t count, size_t alignment = DEFAULT_ALIGNMENT) {
    return arena_.allocateArray<T>(count, alignment);
  }

private:
  ArenaAllocator &arena_;
};

} // namespace pyreth
