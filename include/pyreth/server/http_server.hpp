#pragma once

#include <cstdint>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <pyreth/server/server_config.hpp>
#include <pyreth/thread_pool.hpp>
#include <string>

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

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pyreth::server
