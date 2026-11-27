#pragma once

#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <pyreth/server/model_registry.hpp>
#include <pyreth/thread_pool.hpp>
#include <string>
#include <string_view>

namespace pyreth::server {

/**
 * @brief Formats a Server-Sent Events (SSE) data chunk.
 *
 * @param event Event name (e.g., "step", "done", "error").
 * @param data Event payload (e.g., JSON string or token).
 * @return Formatted SSE frame string: "event: <event>\ndata: <data>\n\n".
 */
PYRETH_API std::string formatSSEEvent(std::string_view event, std::string_view data);

/**
 * @class WebSocketServerContext
 * @brief Shared context across WebSocket controller instances for routing and execution.
 */
class PYRETH_API WebSocketServerContext {
public:
  static WebSocketServerContext& instance() noexcept;

  void init(ModelRegistry* registry, MemoryPool* pool, ThreadPool* thread_pool) noexcept {
    registry_.store(registry, std::memory_order_release);
    memory_pool_.store(pool, std::memory_order_release);
    thread_pool_.store(thread_pool, std::memory_order_release);
  }

  void reset() noexcept {
    registry_.store(nullptr, std::memory_order_release);
    memory_pool_.store(nullptr, std::memory_order_release);
    thread_pool_.store(nullptr, std::memory_order_release);
  }

  [[nodiscard]] ModelRegistry* registry() const noexcept {
    return registry_.load(std::memory_order_acquire);
  }
  [[nodiscard]] MemoryPool* memoryPool() const noexcept {
    return memory_pool_.load(std::memory_order_acquire);
  }
  [[nodiscard]] ThreadPool* threadPool() const noexcept {
    return thread_pool_.load(std::memory_order_acquire);
  }

private:
  WebSocketServerContext() = default;
  std::atomic<ModelRegistry*> registry_{nullptr};
  std::atomic<MemoryPool*> memory_pool_{nullptr};
  std::atomic<ThreadPool*> thread_pool_{nullptr};
};

}  // namespace pyreth::server
