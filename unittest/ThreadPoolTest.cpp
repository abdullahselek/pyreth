#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <pyreth/thread_pool.hpp>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST(ThreadPoolTest, ConstructAndShutdownLifecycle) {
  pyreth::ThreadPool pool(4);

  EXPECT_EQ(pool.numWorkers(), 4u);
  EXPECT_FALSE(pool.isShutdown());

  pool.shutdown();
  EXPECT_TRUE(pool.isShutdown());

  // Idempotent shutdown
  pool.shutdown();
  EXPECT_TRUE(pool.isShutdown());
}

TEST(ThreadPoolTest, SubmitWithReturnValue) {
  pyreth::ThreadPool pool(2);

  auto res1 = pool.submit([](int a, int b) { return a + b; }, 18, 24);
  ASSERT_TRUE(res1.isOk());

  auto res2 = pool.submit([](std::string s) { return s + "_processed"; }, std::string("input"));
  ASSERT_TRUE(res2.isOk());

  auto h1 = std::move(res1).value();
  auto h2 = std::move(res2).value();

  EXPECT_EQ(h1.get(), 42);
  EXPECT_EQ(h2.get(), "input_processed");
}

TEST(ThreadPoolTest, SubmitVoidTask) {
  pyreth::ThreadPool pool(2);
  std::atomic<int> counter{0};

  auto res = pool.submit([&counter]() { counter.fetch_add(10, std::memory_order_relaxed); });
  ASSERT_TRUE(res.isOk());

  auto handle = std::move(res).value();
  handle.wait();

  EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolTest, SubmitMoveOnlyType) {
  pyreth::ThreadPool pool(2);

  auto res = pool.submit([]() { return std::make_unique<int>(999); });
  ASSERT_TRUE(res.isOk());

  auto handle = std::move(res).value();
  std::unique_ptr<int> val = handle.get();
  ASSERT_NE(val, nullptr);
  EXPECT_EQ(*val, 999);
}

TEST(ThreadPoolTest, SubmitDetachedFireAndForget) {
  pyreth::ThreadPool pool(2);
  std::atomic<int> counter{0};
  constexpr int N = 100;

  for (int i = 0; i < N; ++i) {
    auto status =
        pool.submitDetached([&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
    EXPECT_TRUE(status.isOk());
  }

  while (counter.load(std::memory_order_relaxed) < N) {
    std::this_thread::yield();
  }

  EXPECT_EQ(counter.load(), N);
}

TEST(ThreadPoolTest, SubmissionRejectedPostShutdown) {
  pyreth::ThreadPool pool(2);
  pool.shutdown();

  auto res = pool.submit([]() { return 1; });
  EXPECT_TRUE(res.isError());
  EXPECT_EQ(res.error().code(), pyreth::StatusCode::INTERNAL_ERROR);

  auto status = pool.submitDetached([]() {});
  EXPECT_TRUE(status.isError());
}

TEST(ThreadPoolTest, ComputeThreadBudget) {
  auto budget = pyreth::ThreadPool::computeThreadBudget(2);

  EXPECT_GE(budget.ort_intra_op_threads, 1u);
  EXPECT_EQ(budget.ort_inter_op_threads, 1u);
  EXPECT_GE(budget.pyreth_worker_threads, 1u);
}

TEST(ThreadPoolTest, WorkStealingAcrossWorkers) {
  pyreth::ThreadPool pool(4);
  constexpr int SUBTASKS = 256;
  std::atomic<int> completed{0};

  // Submit a root task that spawns child tasks into its local work-stealing deque
  // and waits until at least one peer worker steals a task before returning.
  auto root_res = pool.submit([&pool, &completed]() {
    for (int i = 0; i < SUBTASKS; ++i) {
      (void)pool.submitDetached(
          [&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });
    }
    while (completed.load(std::memory_order_relaxed) == 0) {
      std::this_thread::yield();
    }
  });
  ASSERT_TRUE(root_res.isOk());
  root_res.value().wait();

  while (completed.load(std::memory_order_relaxed) < SUBTASKS) {
    std::this_thread::yield();
  }

  EXPECT_EQ(completed.load(), SUBTASKS);

  auto m = pool.metrics();
  EXPECT_GT(m.tasks_stolen, 0u) << "Child tasks should be stolen across peer worker threads";
}

TEST(ThreadPoolTest, Throughput100kTasksAndLatencyDistribution) {
  constexpr int NUM_TASKS = 100000;
  pyreth::ThreadPool pool(4);

  std::atomic<int> completed{0};
  std::vector<uint64_t> latencies_us(NUM_TASKS);

  auto start_time = std::chrono::steady_clock::now();

  for (int i = 0; i < NUM_TASKS; ++i) {
    auto submit_time = std::chrono::steady_clock::now();
    (void)pool.submitDetached([&completed, &latencies_us, i, submit_time]() {
      auto exec_time = std::chrono::steady_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::microseconds>(exec_time - submit_time).count();
      latencies_us[i] = static_cast<uint64_t>(dur);
      completed.fetch_add(1, std::memory_order_relaxed);
    });
  }

  while (completed.load(std::memory_order_relaxed) < NUM_TASKS) {
    std::this_thread::yield();
  }

  auto total_duration = std::chrono::steady_clock::now() - start_time;
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_duration).count();

  EXPECT_EQ(completed.load(), NUM_TASKS);

  // Compute latency percentiles
  std::sort(latencies_us.begin(), latencies_us.end());
  uint64_t p50 = latencies_us[NUM_TASKS * 50 / 100];
  uint64_t p90 = latencies_us[NUM_TASKS * 90 / 100];
  uint64_t p99 = latencies_us[NUM_TASKS * 99 / 100];

  const uint64_t throughput = total_ms > 0 ? (NUM_TASKS * 1000 / total_ms) : 0;
  std::cout << "[Throughput100k] Total Time: " << total_ms << " ms\n";
  std::cout << "  Throughput: " << throughput << " tasks/sec\n";
  std::cout << "  Latency: P50=" << p50 << "us, P90=" << p90 << "us, P99=" << p99 << "us\n";

  auto metrics = pool.metrics();
  EXPECT_EQ(metrics.tasks_executed, NUM_TASKS);
}
