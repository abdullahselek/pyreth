#pragma once

#include <algorithm>
#include <cstdint>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <string>
#include <thread>

namespace pyreth::server {

/**
 * @struct ServerConfig
 * @brief Configuration parameters for Pyreth HTTP server.
 */
struct PYRETH_API ServerConfig {
  std::string host{"0.0.0.0"};
  uint32_t port{8080};
  size_t io_threads{std::max<size_t>(1, std::thread::hardware_concurrency() / 2)};
  int64_t request_timeout_ms{30000};
  size_t max_payload_size{64 * 1024 * 1024};  // 64MB
  bool enable_cors{false};

  ServerConfig& setHost(std::string h) {
    host = std::move(h);
    return *this;
  }

  ServerConfig& setPort(uint32_t p) noexcept {
    port = p;
    return *this;
  }

  ServerConfig& setIoThreads(size_t threads) noexcept {
    io_threads = threads;
    return *this;
  }

  ServerConfig& setRequestTimeoutMs(int64_t timeout_ms) noexcept {
    request_timeout_ms = timeout_ms;
    return *this;
  }

  ServerConfig& setMaxPayloadSize(size_t payload_size) noexcept {
    max_payload_size = payload_size;
    return *this;
  }

  ServerConfig& setEnableCors(bool cors) noexcept {
    enable_cors = cors;
    return *this;
  }

  [[nodiscard]] Result<void> validate() const {
    if (port > 65535) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Server port must be between 0 and 65535");
    }
    if (request_timeout_ms < 0) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Request timeout must be non-negative");
    }
    if (io_threads == 0) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "I/O threads must be greater than zero");
    }
    if (max_payload_size == 0) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Max payload size must be greater than zero");
    }
    return Result<void>::success();
  }

  [[nodiscard]] bool isValid() const noexcept { return validate().isOk(); }
};

}  // namespace pyreth::server
