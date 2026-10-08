#include <gtest/gtest.h>

#include <atomic>
#include <pyreth/work_stealing_queue.hpp>
#include <string>
#include <thread>
#include <vector>

namespace {

struct DestructorTracker {
  static std::atomic<int> alive_count;
  int id{0};

  explicit DestructorTracker(int i) : id(i) { alive_count.fetch_add(1, std::memory_order_relaxed); }

  DestructorTracker(const DestructorTracker& other) : id(other.id) {
    alive_count.fetch_add(1, std::memory_order_relaxed);
  }

  DestructorTracker(DestructorTracker&& other) noexcept : id(other.id) {
    alive_count.fetch_add(1, std::memory_order_relaxed);
  }

  DestructorTracker& operator=(const DestructorTracker&) = default;
  DestructorTracker& operator=(DestructorTracker&&) noexcept = default;

  ~DestructorTracker() { alive_count.fetch_sub(1, std::memory_order_relaxed); }
};

std::atomic<int> DestructorTracker::alive_count{0};

struct MoveOnlyPayload {
  int value{0};
  std::string name;

  MoveOnlyPayload(int v, std::string n) : value(v), name(std::move(n)) {}
  MoveOnlyPayload(const MoveOnlyPayload&) = delete;
  MoveOnlyPayload& operator=(const MoveOnlyPayload&) = delete;
  MoveOnlyPayload(MoveOnlyPayload&&) noexcept = default;
  MoveOnlyPayload& operator=(MoveOnlyPayload&&) noexcept = default;
};

}  // namespace

TEST(WorkStealingQueueTest, SingleThreadedPushPopLIFO) {
  pyreth::WorkStealingQueue<int, 8> queue;

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0u);
  EXPECT_EQ(queue.capacity(), 8u);

  EXPECT_TRUE(queue.push(10));
  EXPECT_TRUE(queue.push(20));
  EXPECT_TRUE(queue.push(30));

  EXPECT_FALSE(queue.empty());
  EXPECT_EQ(queue.size(), 3u);

  // LIFO order for owner pop
  auto p1 = queue.pop();
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(*p1, 30);

  auto p2 = queue.pop();
  ASSERT_TRUE(p2.has_value());
  EXPECT_EQ(*p2, 20);

  auto p3 = queue.pop();
  ASSERT_TRUE(p3.has_value());
  EXPECT_EQ(*p3, 10);

  EXPECT_TRUE(queue.empty());
  EXPECT_EQ(queue.size(), 0u);

  // Pop from empty queue
  auto empty_pop = queue.pop();
  EXPECT_FALSE(empty_pop.has_value());
}

TEST(WorkStealingQueueTest, SingleThreadedPushStealFIFO) {
  pyreth::WorkStealingQueue<int, 8> queue;

  EXPECT_TRUE(queue.push(100));
  EXPECT_TRUE(queue.push(200));
  EXPECT_TRUE(queue.push(300));

  // FIFO order for thief steal
  auto s1 = queue.steal();
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(*s1, 100);

  auto s2 = queue.steal();
  ASSERT_TRUE(s2.has_value());
  EXPECT_EQ(*s2, 200);

  auto s3 = queue.steal();
  ASSERT_TRUE(s3.has_value());
  EXPECT_EQ(*s3, 300);

  EXPECT_TRUE(queue.empty());
  auto empty_steal = queue.steal();
  EXPECT_FALSE(empty_steal.has_value());
}

TEST(WorkStealingQueueTest, InterleavedPopAndSteal) {
  pyreth::WorkStealingQueue<int, 8> queue;

  EXPECT_TRUE(queue.push(1));
  EXPECT_TRUE(queue.push(2));
  EXPECT_TRUE(queue.push(3));
  EXPECT_TRUE(queue.push(4));

  // Steal oldest (FIFO) -> 1
  auto s1 = queue.steal();
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(*s1, 1);

  // Pop newest (LIFO) -> 4
  auto p1 = queue.pop();
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(*p1, 4);

  // Steal next oldest -> 2
  auto s2 = queue.steal();
  ASSERT_TRUE(s2.has_value());
  EXPECT_EQ(*s2, 2);

  // Pop remaining -> 3
  auto p2 = queue.pop();
  ASSERT_TRUE(p2.has_value());
  EXPECT_EQ(*p2, 3);

  EXPECT_TRUE(queue.empty());
}

TEST(WorkStealingQueueTest, BoundedCapacityRejection) {
  pyreth::WorkStealingQueue<int, 4> queue;

  EXPECT_TRUE(queue.push(1));
  EXPECT_TRUE(queue.push(2));
  EXPECT_TRUE(queue.push(3));
  EXPECT_TRUE(queue.push(4));

  // Queue is now full (capacity 4)
  EXPECT_FALSE(queue.push(5));
  EXPECT_FALSE(queue.push(6));
  EXPECT_EQ(queue.size(), 4u);

  // Popping one makes room for one more
  auto p = queue.pop();
  ASSERT_TRUE(p.has_value());
  EXPECT_EQ(*p, 4);

  EXPECT_TRUE(queue.push(5));
  EXPECT_FALSE(queue.push(6));
}

TEST(WorkStealingQueueTest, MoveOnlyPayloadSupport) {
  pyreth::WorkStealingQueue<MoveOnlyPayload, 8> queue;

  EXPECT_TRUE(queue.push(MoveOnlyPayload(101, "tensor_op")));
  EXPECT_TRUE(queue.push(MoveOnlyPayload(102, "kernel_eval")));

  auto p1 = queue.pop();
  ASSERT_TRUE(p1.has_value());
  EXPECT_EQ(p1->value, 102);
  EXPECT_EQ(p1->name, "kernel_eval");

  auto s1 = queue.steal();
  ASSERT_TRUE(s1.has_value());
  EXPECT_EQ(s1->value, 101);
  EXPECT_EQ(s1->name, "tensor_op");
}

TEST(WorkStealingQueueTest, EmplaceSupport) {
  pyreth::WorkStealingQueue<MoveOnlyPayload, 8> queue;

  EXPECT_TRUE(queue.emplace(42, "direct_emplace"));
  auto item = queue.pop();
  ASSERT_TRUE(item.has_value());
  EXPECT_EQ(item->value, 42);
  EXPECT_EQ(item->name, "direct_emplace");
}

TEST(WorkStealingQueueTest, WraparoundCycles) {
  pyreth::WorkStealingQueue<int, 4> queue;

  // Run thousands of wrap cycles to ensure power-of-two mask indexing
  // advances without arithmetic corruption
  for (int i = 0; i < 5000; ++i) {
    ASSERT_TRUE(queue.push(i));
    ASSERT_TRUE(queue.push(i + 1));

    auto stolen = queue.steal();
    ASSERT_TRUE(stolen.has_value());
    EXPECT_EQ(*stolen, i);

    auto popped = queue.pop();
    ASSERT_TRUE(popped.has_value());
    EXPECT_EQ(*popped, i + 1);
  }

  EXPECT_TRUE(queue.empty());
}

TEST(WorkStealingQueueTest, DestructorCleansRemainingElements) {
  DestructorTracker::alive_count.store(0, std::memory_order_relaxed);

  {
    pyreth::WorkStealingQueue<DestructorTracker, 8> queue;
    queue.push(DestructorTracker(1));
    queue.push(DestructorTracker(2));
    queue.push(DestructorTracker(3));
    EXPECT_EQ(DestructorTracker::alive_count.load(), 3);
  }

  // Destruction of the queue should destroy all remaining items
  EXPECT_EQ(DestructorTracker::alive_count.load(), 0);
}

TEST(WorkStealingQueueTest, MultiThreadedStealStress) {
  constexpr int NUM_ITEMS = 50000;
  constexpr int NUM_THIEVES = 4;

  pyreth::WorkStealingQueue<int, 1024> queue;

  std::atomic<bool> producer_done{false};
  std::atomic<int> stolen_count{0};
  std::atomic<int> popped_count{0};
  std::vector<std::atomic<int>> received_tracker(NUM_ITEMS);
  for (int i = 0; i < NUM_ITEMS; ++i) {
    received_tracker[i].store(0, std::memory_order_relaxed);
  }

  std::vector<std::thread> thieves;
  thieves.reserve(NUM_THIEVES);

  for (int t = 0; t < NUM_THIEVES; ++t) {
    thieves.emplace_back([&]() {
      while (!producer_done.load(std::memory_order_acquire) || !queue.empty()) {
        auto item = queue.steal();
        if (item.has_value()) {
          stolen_count.fetch_add(1, std::memory_order_relaxed);
          received_tracker[*item].fetch_add(1, std::memory_order_relaxed);
        } else {
          std::this_thread::yield();
        }
      }

      // Final drain
      while (auto item = queue.steal()) {
        stolen_count.fetch_add(1, std::memory_order_relaxed);
        received_tracker[*item].fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  // Producer thread (test runner)
  for (int i = 0; i < NUM_ITEMS; ++i) {
    while (!queue.push(i)) {
      // If queue is full, owner pops and processes
      auto item = queue.pop();
      if (item.has_value()) {
        popped_count.fetch_add(1, std::memory_order_relaxed);
        received_tracker[*item].fetch_add(1, std::memory_order_relaxed);
      }
    }

    // Periodically pop to create contention with thieves
    if (i % 4 == 0) {
      auto item = queue.pop();
      if (item.has_value()) {
        popped_count.fetch_add(1, std::memory_order_relaxed);
        received_tracker[*item].fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  // Drain remaining items by owner
  while (auto item = queue.pop()) {
    popped_count.fetch_add(1, std::memory_order_relaxed);
    received_tracker[*item].fetch_add(1, std::memory_order_relaxed);
  }

  producer_done.store(true, std::memory_order_release);

  for (auto& thief : thieves) {
    thief.join();
  }

  int total_processed = popped_count.load() + stolen_count.load();
  EXPECT_EQ(total_processed, NUM_ITEMS);

  // Verify that every single item was processed exactly once
  for (int i = 0; i < NUM_ITEMS; ++i) {
    EXPECT_EQ(received_tracker[i].load(), 1) << "Item " << i << " was not processed exactly once";
  }
}
