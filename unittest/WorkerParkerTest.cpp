#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <pyreth/worker_parker.hpp>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST(WorkerParkerTest, UnparkBeforeParkTokenPersistence) {
  pyreth::WorkerParker parker;

  // Unpark called before park
  parker.unpark();

  // Park should return immediately without blocking
  auto start = std::chrono::steady_clock::now();
  parker.park();
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_LT(elapsed, 200ms);
  EXPECT_FALSE(parker.isParked());
}

TEST(WorkerParkerTest, IdempotentMultipleUnparks) {
  pyreth::WorkerParker parker;

  // Multiple consecutive unparks
  parker.unpark();
  parker.unpark();
  parker.unpark();

  // First park consumes the single available permit
  parker.park();

  // Subsequent parkFor should timeout because permit was already consumed
  auto start = std::chrono::steady_clock::now();
  bool acquired = parker.parkFor(30ms);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_FALSE(acquired);
  EXPECT_GE(elapsed, 20ms);
}

TEST(WorkerParkerTest, TimedParkingTimeout) {
  pyreth::WorkerParker parker;

  auto start = std::chrono::steady_clock::now();
  bool acquired = parker.parkFor(40ms);
  auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_FALSE(acquired);
  EXPECT_GE(elapsed, 25ms);
}

TEST(WorkerParkerTest, TimedParkingEarlyWakeup) {
  pyreth::WorkerParker parker;

  std::thread unparker([&]() {
    auto t0 = std::chrono::steady_clock::now();
    while (!parker.isParked() && (std::chrono::steady_clock::now() - t0) < 1000ms) {
      std::this_thread::yield();
    }
    parker.unpark();
  });

  auto start = std::chrono::steady_clock::now();
  bool acquired = parker.parkFor(2000ms);
  auto elapsed = std::chrono::steady_clock::now() - start;

  unparker.join();

  EXPECT_TRUE(acquired);
  EXPECT_LT(elapsed, 1500ms);
}

TEST(WorkerParkerTest, ConcurrentParkAndUnparkStress) {
  pyreth::WorkerParker parker;

  for (int i = 0; i < 500; ++i) {
    std::atomic<bool> worker_ready{false};

    std::thread worker([&]() {
      worker_ready.store(true, std::memory_order_release);
      parker.park();
    });

    while (!worker_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }

    // Small delay to ensure worker has entered or is entering park()
    std::this_thread::sleep_for(200us);
    parker.unpark();

    worker.join();
  }
}

TEST(WorkerParkerTest, MultiThreadedUnparkConvergence) {
  pyreth::WorkerParker parker;
  constexpr int NUM_CALLERS = 8;

  std::vector<std::thread> callers;
  callers.reserve(NUM_CALLERS);

  for (int i = 0; i < NUM_CALLERS; ++i) {
    callers.emplace_back([&]() {
      for (int r = 0; r < 50; ++r) {
        parker.unpark();
      }
    });
  }

  for (auto& t : callers) {
    t.join();
  }

  // Consume permit
  parker.park();

  // Parker should now be empty and parkFor should time out
  EXPECT_FALSE(parker.parkFor(20ms));
}

TEST(WorkerParkerTest, AtomicNotificationSingleAndBroadcast) {
  pyreth::AtomicNotification notifier;

  EXPECT_EQ(notifier.get(), 0u);

  // Single notification
  std::thread t1([&]() { notifier.wait(0u); });

  std::this_thread::sleep_for(10ms);
  notifier.notifyOne();
  t1.join();
  EXPECT_EQ(notifier.get(), 1u);

  // Broadcast notification to multiple waiters
  constexpr int NUM_WAITERS = 4;
  std::atomic<int> completed{0};
  std::vector<std::thread> waiters;
  waiters.reserve(NUM_WAITERS);

  for (int i = 0; i < NUM_WAITERS; ++i) {
    waiters.emplace_back([&]() {
      notifier.wait(1u);
      completed.fetch_add(1, std::memory_order_relaxed);
    });
  }

  std::this_thread::sleep_for(15ms);
  notifier.notifyAll();

  for (auto& w : waiters) {
    w.join();
  }

  EXPECT_EQ(completed.load(), NUM_WAITERS);
  EXPECT_EQ(notifier.get(), 2u);
}
