#pragma once

#include <cstdint>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <pyreth/server/endpoint.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/server/model_registry.hpp>
#include <pyreth/server/server_config.hpp>
#include <pyreth/thread_pool.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace pyreth::server {

/**
 * @class HttpServer
 * @brief High-performance non-blocking HTTP server powered by Drogon.
 */
class PYRETH_API HttpServer {
public:
  explicit HttpServer(const ServerConfig& config = ServerConfig{});
  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  HttpServer(HttpServer&&) noexcept;
  HttpServer& operator=(HttpServer&&) noexcept;

  [[nodiscard]] Result<void> startAsync();
  [[nodiscard]] Result<void> start();
  void stop();

  [[nodiscard]] bool isRunning() const noexcept;
  [[nodiscard]] uint16_t port() const noexcept;
  [[nodiscard]] const std::string& host() const noexcept;
  [[nodiscard]] const ServerConfig& config() const noexcept;
  [[nodiscard]] ThreadPool& threadPool() noexcept;
  [[nodiscard]] MemoryPool& memoryPool() noexcept;
  [[nodiscard]] const MemoryPool& memoryPool() const noexcept;

  [[nodiscard]] Result<void> registerModel(std::string name,
                                           std::shared_ptr<InferenceSession> session,
                                           std::string version = "1");
  [[nodiscard]] Result<void> registerBatcher(std::string name,
                                             std::shared_ptr<DynamicBatcher> batcher);
  [[nodiscard]] Result<void> registerBatcher(std::string name, DynamicBatcher batcher) {
    return registerBatcher(std::move(name), std::make_shared<DynamicBatcher>(std::move(batcher)));
  }
  [[nodiscard]] Result<void> unregisterBatcher(std::string_view name);
  [[nodiscard]] Result<void> unregisterModel(std::string_view name);
  [[nodiscard]] bool hasModel(std::string_view name) const;

  [[nodiscard]] ModelRegistry& modelRegistry() noexcept;
  [[nodiscard]] const ModelRegistry& modelRegistry() const noexcept;

  EndpointBuilder& route(std::string path);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pyreth::server
