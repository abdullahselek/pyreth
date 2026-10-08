#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory_resource>
#include <pyreth/memory_pool.hpp>
#include <string>
#include <thread>
#include <vector>

TEST(MemoryPoolTest, AlignmentVerificationAcrossSizes) {
  pyreth::ArenaAllocator arena(1024 * 64);

  // Test across a diverse range of allocation sizes (odd, prime, power-of-two)
  const size_t test_sizes[] = {1,  3,   7,   8,   15,  16,   31,   32,   63,
                               64, 127, 128, 255, 256, 1023, 1024, 4096, 16384};

  for (size_t size : test_sizes) {
    void* ptr = arena.allocate(size);
    ASSERT_NE(ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u) << "Failed for allocation size: " << size;
  }
}

TEST(MemoryPoolTest, MonadicTryAllocate) {
  pyreth::ArenaAllocator arena(1024 * 64);

  auto res = arena.tryAllocate(512);
  ASSERT_TRUE(res.isOk());
  void* ptr = res.value();
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
}

TEST(MemoryPoolTest, ConsecutiveAllocationsNoOverlap) {
  pyreth::ArenaAllocator arena(1024 * 64);
  constexpr size_t NUM_BUFFERS = 32;
  constexpr size_t BUFFER_SIZE = 128;

  std::vector<uint8_t*> buffers;
  buffers.reserve(NUM_BUFFERS);

  for (size_t i = 0; i < NUM_BUFFERS; ++i) {
    auto* buf = static_cast<uint8_t*>(arena.allocate(BUFFER_SIZE));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(buf) % 64, 0u);
    std::memset(buf, static_cast<int>(i + 1), BUFFER_SIZE);
    buffers.push_back(buf);
  }

  // Verify memory integrity and non-overlapping buffers
  for (size_t i = 0; i < NUM_BUFFERS; ++i) {
    for (size_t b = 0; b < BUFFER_SIZE; ++b) {
      EXPECT_EQ(buffers[i][b], static_cast<uint8_t>(i + 1));
    }
  }
}

TEST(MemoryPoolTest, TypedCreateAndAllocateArray) {
  pyreth::ArenaAllocator arena(1024 * 64);

  struct Point {
    float x;
    float y;
    float z;
  };

  Point* p = arena.create<Point>(1.0f, 2.0f, 3.0f);
  ASSERT_NE(p, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
  EXPECT_FLOAT_EQ(p->x, 1.0f);
  EXPECT_FLOAT_EQ(p->y, 2.0f);
  EXPECT_FLOAT_EQ(p->z, 3.0f);

  constexpr size_t ARRAY_LEN = 256;
  float* arr = arena.allocateArray<float>(ARRAY_LEN);
  ASSERT_NE(arr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(arr) % 64, 0u);

  for (size_t i = 0; i < ARRAY_LEN; ++i) {
    arr[i] = static_cast<float>(i) * 0.5f;
  }
  for (size_t i = 0; i < ARRAY_LEN; ++i) {
    EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i) * 0.5f);
  }
}

TEST(MemoryPoolTest, BlockChainingAndLargeAllocations) {
  // Arena with small 16 KB default block size
  pyreth::ArenaAllocator arena(16 * 1024);
  EXPECT_EQ(arena.blockCount(), 1u);

  // Allocate 10 KB (fits in block 1)
  void* p1 = arena.allocate(10 * 1024);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p1) % 64, 0u);

  // Allocate another 10 KB (exceeds block 1, forces block 2)
  void* p2 = arena.allocate(10 * 1024);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p2) % 64, 0u);
  EXPECT_GE(arena.blockCount(), 2u);

  // Allocate a 128 KB buffer (exceeds default block size, creates large block)
  void* p3 = arena.allocate(128 * 1024);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(p3) % 64, 0u);
  EXPECT_GE(arena.blockCount(), 3u);
}

TEST(MemoryPoolTest, O1ResetAndBlockReuse) {
  pyreth::ArenaAllocator arena(16 * 1024);

  // Allocate across multiple blocks
  for (int i = 0; i < 5; ++i) {
    (void)arena.allocate(12 * 1024);
  }
  size_t blocks_before = arena.blockCount();
  size_t reserved_before = arena.bytesReserved();
  EXPECT_GT(arena.bytesAllocated(), 0u);
  EXPECT_GE(blocks_before, 4u);

  // O(1) reset
  arena.reset();
  EXPECT_EQ(arena.bytesAllocated(), 0u);
  EXPECT_EQ(arena.bytesReserved(), reserved_before);
  EXPECT_EQ(arena.blockCount(), blocks_before);

  // Re-allocate across the existing blocks
  for (int i = 0; i < 5; ++i) {
    void* ptr = arena.allocate(12 * 1024);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0u);
  }

  // Blocks should have been reused without new OS block allocations
  EXPECT_EQ(arena.blockCount(), blocks_before);
}

TEST(MemoryPoolTest, PmrVectorCompatibility) {
  pyreth::ArenaAllocator arena(1024 * 64);

  std::pmr::vector<int> numbers(&arena);
  for (int i = 0; i < 1000; ++i) {
    numbers.push_back(i * 3);
  }

  EXPECT_EQ(numbers.size(), 1000u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(numbers.data()) % 64, 0u);

  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(numbers[i], i * 3);
  }
}

TEST(MemoryPoolTest, ThreadLocalArenaIsolation) {
  constexpr int NUM_THREADS = 4;
  std::vector<std::thread> threads;
  std::vector<uintptr_t> arena_addrs(NUM_THREADS, 0);
  std::atomic<int> ready{0};
  std::atomic<bool> start_gate{false};

  for (int t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([t, &arena_addrs, &ready, &start_gate]() {
      auto& arena = pyreth::MemoryPool::getThreadLocalArena();
      arena_addrs[t] = reinterpret_cast<uintptr_t>(&arena);

      ready.fetch_add(1, std::memory_order_release);
      while (!start_gate.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int i = 0; i < 100; ++i) {
        void* p = arena.allocate(128);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
      }
      arena.reset();
    });
  }

  while (ready.load(std::memory_order_acquire) < NUM_THREADS) {
    std::this_thread::yield();
  }
  start_gate.store(true, std::memory_order_release);

  for (auto& t : threads) {
    t.join();
  }

  // All concurrently active thread-local arena instances reside at distinct
  // memory addresses
  for (int i = 0; i < NUM_THREADS; ++i) {
    for (int j = i + 1; j < NUM_THREADS; ++j) {
      EXPECT_NE(arena_addrs[i], arena_addrs[j]);
    }
  }
}

TEST(MemoryPoolTest, MemoryPoolLeaseLifecycle) {
  pyreth::MemoryPool pool(2, 64 * 1024);
  EXPECT_EQ(pool.availableCount(), 2u);

  {
    auto lease1 = pool.acquire();
    EXPECT_TRUE(lease1.valid());
    EXPECT_EQ(pool.availableCount(), 1u);

    void* p = lease1->allocate(512);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    EXPECT_EQ(lease1->bytesAllocated(), 512u);

    {
      auto lease2 = pool.acquire();
      EXPECT_EQ(pool.availableCount(), 0u);
    }
    // lease2 released and returned to pool
    EXPECT_EQ(pool.availableCount(), 1u);
  }
  // lease1 released, reset, and returned to pool
  EXPECT_EQ(pool.availableCount(), 2u);

  // Next lease is freshly reset
  auto lease3 = pool.acquire();
  EXPECT_EQ(lease3->bytesAllocated(), 0u);
}

TEST(MemoryPoolTest, ScopedArenaLifecycle) {
  pyreth::ArenaAllocator arena(64 * 1024);

  {
    pyreth::ScopedArena scoped(arena);
    void* p = scoped.allocate(1024);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 64, 0u);
    EXPECT_EQ(arena.bytesAllocated(), 1024u);
  }

  // ScopedArena destructor automatically resets arena
  EXPECT_EQ(arena.bytesAllocated(), 0u);
}

TEST(MemoryPoolTest, BenchmarkAllocationSpeedAgainstMallocAndPmr) {
  constexpr int NUM_ALLOCS = 500000;
  constexpr size_t ALLOC_SIZE = 128;

  pyreth::ArenaAllocator arena(64 * 1024 * 1024);  // 64 MB arena

  // 1. Pyreth ArenaAllocator
  auto start_arena = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    void* p = arena.allocate(ALLOC_SIZE);
    (void)p;
  }
  auto dur_arena = std::chrono::steady_clock::now() - start_arena;
  auto ms_arena = std::chrono::duration_cast<std::chrono::microseconds>(dur_arena).count();

  // 2. std::malloc + std::free
  auto start_malloc = std::chrono::steady_clock::now();
  std::vector<void*> malloc_ptrs(NUM_ALLOCS);
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    malloc_ptrs[i] = std::malloc(ALLOC_SIZE);
  }
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    std::free(malloc_ptrs[i]);
  }
  auto dur_malloc = std::chrono::steady_clock::now() - start_malloc;
  auto ms_malloc = std::chrono::duration_cast<std::chrono::microseconds>(dur_malloc).count();

  // 3. std::pmr::new_delete_resource
  auto* pmr_res = std::pmr::new_delete_resource();
  auto start_pmr = std::chrono::steady_clock::now();
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    malloc_ptrs[i] = pmr_res->allocate(ALLOC_SIZE, 64);
  }
  for (int i = 0; i < NUM_ALLOCS; ++i) {
    pmr_res->deallocate(malloc_ptrs[i], ALLOC_SIZE, 64);
  }
  auto dur_pmr = std::chrono::steady_clock::now() - start_pmr;
  auto ms_pmr = std::chrono::duration_cast<std::chrono::microseconds>(dur_pmr).count();

  const double speedup = static_cast<double>(ms_malloc) / static_cast<double>(ms_arena);
  std::cout << "[MemoryBenchmark " << NUM_ALLOCS << " allocs (" << ALLOC_SIZE << " B)]\n";
  std::cout << "  ArenaAllocator:    " << ms_arena / 1000.0 << " ms\n";
  std::cout << "  std::malloc:       " << ms_malloc / 1000.0 << " ms\n";
  std::cout << "  std::pmr:          " << ms_pmr / 1000.0 << " ms\n";
  std::cout << "  Speedup vs malloc: " << speedup << "x\n";

  EXPECT_LT(ms_arena, ms_malloc) << "ArenaAllocator should be strictly faster than std::malloc";
}
