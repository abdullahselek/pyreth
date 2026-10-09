#include <algorithm>
#include <new>
#include <pyreth/memory_pool.hpp>

namespace pyreth {

struct ArenaAllocator::Block {
  std::byte* data{nullptr};
  size_t capacity{0};
  size_t offset{0};
  std::unique_ptr<Block> next{nullptr};

  explicit Block(size_t cap) : capacity(cap), offset(0) {
    data = static_cast<std::byte*>(::operator new(capacity, std::align_val_t(DEFAULT_ALIGNMENT)));
  }

  ~Block() {
    if (data) {
      ::operator delete(data, std::align_val_t(DEFAULT_ALIGNMENT));
    }
  }
};

ArenaAllocator::ArenaAllocator(size_t block_size) : default_block_size_(block_size) {
  head_ = allocateBlock(default_block_size_);
  current_ = head_.get();
}

ArenaAllocator::~ArenaAllocator() = default;

ArenaAllocator::ArenaAllocator(ArenaAllocator&& other) noexcept
    : default_block_size_(other.default_block_size_),
      head_(std::move(other.head_)),
      current_(other.current_),
      bytes_allocated_(other.bytes_allocated_),
      bytes_reserved_(other.bytes_reserved_),
      block_count_(other.block_count_) {
  pinned_count_.store(other.pinned_count_.load(std::memory_order_relaxed),
                      std::memory_order_relaxed);
  other.pinned_count_.store(0, std::memory_order_relaxed);
  other.current_ = nullptr;
  other.bytes_allocated_ = 0;
  other.bytes_reserved_ = 0;
  other.block_count_ = 0;
}

ArenaAllocator& ArenaAllocator::operator=(ArenaAllocator&& other) noexcept {
  if (this != &other) {
    head_ = std::move(other.head_);
    current_ = other.current_;
    default_block_size_ = other.default_block_size_;
    bytes_allocated_ = other.bytes_allocated_;
    bytes_reserved_ = other.bytes_reserved_;
    block_count_ = other.block_count_;
    pinned_count_.store(other.pinned_count_.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);

    other.pinned_count_.store(0, std::memory_order_relaxed);
    other.current_ = nullptr;
    other.bytes_allocated_ = 0;
    other.bytes_reserved_ = 0;
    other.block_count_ = 0;
  }
  return *this;
}

void* ArenaAllocator::allocate(size_t bytes, size_t alignment) {
  if (alignment < alignof(std::max_align_t)) {
    alignment = alignof(std::max_align_t);
  }
  if (alignment < DEFAULT_ALIGNMENT) {
    alignment = DEFAULT_ALIGNMENT;
  }

  uintptr_t curr = reinterpret_cast<uintptr_t>(current_->data + current_->offset);
  uintptr_t aligned = (curr + (alignment - 1)) & ~(static_cast<uintptr_t>(alignment) - 1);
  size_t padding = aligned - curr;
  size_t needed = padding + bytes;

  if (current_->offset + needed <= current_->capacity) {
    current_->offset += needed;
    bytes_allocated_ += bytes;
    return reinterpret_cast<void*>(aligned);
  }

  return allocateFromNextBlock(bytes, alignment);
}

Result<void*> ArenaAllocator::tryAllocate(size_t bytes, size_t alignment) noexcept {
  try {
    void* ptr = allocate(bytes, alignment);
    return Result<void*>::success(ptr);
  } catch (...) {
    return Result<void*>::failure(StatusCode::ALLOCATION_FAILED,
                                  "ArenaAllocator allocation failed");
  }
}

void ArenaAllocator::deallocate(void* /*p*/, size_t /*bytes*/, size_t /*alignment*/) noexcept {
  // Arena allocations are bulk-deallocated on reset() or destruction.
}

void ArenaAllocator::pin() noexcept {
  pinned_count_.fetch_add(1, std::memory_order_relaxed);
}

void ArenaAllocator::unpin() noexcept {
  pinned_count_.fetch_sub(1, std::memory_order_release);
}

bool ArenaAllocator::isPinned() const noexcept {
  return pinned_count_.load(std::memory_order_acquire) > 0;
}

size_t ArenaAllocator::pinCount() const noexcept {
  return pinned_count_.load(std::memory_order_acquire);
}

bool ArenaAllocator::canReset() const noexcept {
  return !isPinned();
}

void ArenaAllocator::reset() noexcept {
  if (isPinned()) {
    return;
  }
  forceReset();
}

Result<void> ArenaAllocator::tryReset() noexcept {
  if (isPinned()) {
    return Result<void>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Cannot reset ArenaAllocator: active references are pinning memory");
  }
  forceReset();
  return Result<void>::success();
}

void ArenaAllocator::forceReset() noexcept {
  Block* b = head_.get();
  while (b) {
    b->offset = 0;
    b = b->next.get();
  }
  current_ = head_.get();
  bytes_allocated_ = 0;
}

size_t ArenaAllocator::bytesAllocated() const noexcept {
  return bytes_allocated_;
}

size_t ArenaAllocator::bytesReserved() const noexcept {
  return bytes_reserved_;
}

size_t ArenaAllocator::blockCount() const noexcept {
  return block_count_;
}

void* ArenaAllocator::do_allocate(size_t bytes, size_t alignment) {
  return allocate(bytes, alignment);
}

void ArenaAllocator::do_deallocate(void* p, size_t bytes, size_t alignment) {
  deallocate(p, bytes, alignment);
}

bool ArenaAllocator::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}

std::unique_ptr<ArenaAllocator::Block> ArenaAllocator::allocateBlock(size_t capacity) {
  bytes_reserved_ += capacity;
  block_count_++;
  return std::make_unique<Block>(capacity);
}

void* ArenaAllocator::allocateFromNextBlock(size_t bytes, size_t alignment) {
  if (current_->next) {
    current_ = current_->next.get();
    current_->offset = 0;
    return allocate(bytes, alignment);
  }

  size_t next_size = std::max(default_block_size_, bytes + alignment);
  current_->next = allocateBlock(next_size);
  current_ = current_->next.get();
  return allocate(bytes, alignment);
}

// ---------------------------------------------------------------------------
// MemoryPool::Lease Implementation
// ---------------------------------------------------------------------------

MemoryPool::Lease::Lease(MemoryPool* pool, std::unique_ptr<ArenaAllocator> arena)
    : pool_(pool), arena_(std::move(arena)) {}

MemoryPool::Lease::~Lease() {
  if (pool_ && arena_) {
    arena_->reset();
    pool_->returnArena(std::move(arena_));
  }
}

MemoryPool::Lease::Lease(Lease&& other) noexcept
    : pool_(other.pool_), arena_(std::move(other.arena_)) {
  other.pool_ = nullptr;
}

MemoryPool::Lease& MemoryPool::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    if (pool_ && arena_) {
      arena_->reset();
      pool_->returnArena(std::move(arena_));
    }
    pool_ = other.pool_;
    arena_ = std::move(other.arena_);
    other.pool_ = nullptr;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// MemoryPool Implementation
// ---------------------------------------------------------------------------

MemoryPool::MemoryPool(size_t initial_count, size_t block_size) : default_block_size_(block_size) {
  for (size_t i = 0; i < initial_count; ++i) {
    arenas_.push_back(std::make_unique<ArenaAllocator>(default_block_size_));
  }
}

MemoryPool::~MemoryPool() = default;

MemoryPool::Lease MemoryPool::acquire() {
  std::unique_ptr<ArenaAllocator> arena;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!arenas_.empty()) {
      arena = std::move(arenas_.back());
      arenas_.pop_back();
    }
  }

  if (!arena) {
    arena = std::make_unique<ArenaAllocator>(default_block_size_);
  }

  return Lease(this, std::move(arena));
}

ArenaAllocator& MemoryPool::getThreadLocalArena() {
  thread_local ArenaAllocator tl_arena;
  return tl_arena;
}

size_t MemoryPool::availableCount() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return arenas_.size();
}

void MemoryPool::returnArena(std::unique_ptr<ArenaAllocator> arena) {
  std::lock_guard<std::mutex> lock(mutex_);
  arenas_.push_back(std::move(arena));
}

// ---------------------------------------------------------------------------
// ScopedArena Implementation
// ---------------------------------------------------------------------------

ScopedArena::ScopedArena(ArenaAllocator& arena) : arena_(arena) {}

ScopedArena::~ScopedArena() {
  arena_.reset();
}

}  // namespace pyreth
