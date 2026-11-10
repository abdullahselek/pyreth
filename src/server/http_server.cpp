#include <drogon/drogon.h>
#include <json/json.h>

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <pyreth/common.hpp>
#include <pyreth/server/http_server.hpp>
#include <thread>
#include <utility>

namespace pyreth::server {

class HttpServer::Impl {
public:
  explicit Impl(const ServerConfig& config)
      : config_(config), pool_(config.io_threads > 0 ? config.io_threads : 1) {}

  ~Impl() { stop(); }

  Result<void> startAsync() {
    auto val = config_.validate();
    if (!val.isOk()) {
      return val;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (running_.load()) {
      return Result<void>::failure(StatusCode::INTERNAL_ERROR, "Server is already running");
    }

    // Configure Drogon
    drogon::app().addListener(config_.host, static_cast<uint16_t>(config_.port));
    drogon::app().setThreadNum(config_.io_threads);
    drogon::app().setClientMaxBodySize(config_.max_payload_size);
    if (config_.request_timeout_ms > 0) {
      drogon::app().setIdleConnectionTimeout(
          static_cast<size_t>(config_.request_timeout_ms / 1000));
    }

    // Register system health check endpoints
    drogon::app().registerHandler(
        "/healthz",
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value root;
          root["status"] = "ok";
          root["version"] = pyreth::PYRETH_VERSION;
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, root));
          callback(resp);
        },
        {drogon::Get});

    drogon::app().registerHandler(
        "/livez",
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value root;
          root["status"] = "alive";
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, root));
          callback(resp);
        },
        {drogon::Get});

    if (config_.enable_cors) {
      drogon::app().registerPreRoutingAdvice([](const drogon::HttpRequestPtr& req,
                                                drogon::AdviceCallback&& acb,
                                                drogon::AdviceChainCallback&& accb) {
        if (req->method() == drogon::HttpMethod::Options) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k204NoContent);
          resp->setCloseConnection(true);
          resp->addHeader("Access-Control-Allow-Origin", "*");
          resp->addHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
          resp->addHeader("Access-Control-Allow-Headers",
                          "Content-Type, Authorization, X-Requested-With, X-Pyreth-Execution-Mode");
          acb(resp);
          return;
        }
        accb();
      });
      drogon::app().registerPostHandlingAdvice(
          [](const drogon::HttpRequestPtr&, const drogon::HttpResponsePtr& resp) {
            resp->addHeader("Access-Control-Allow-Origin", "*");
          });
    }

    std::promise<uint16_t> port_promise;
    auto port_future = port_promise.get_future();

    drogon::app().registerBeginningAdvice([this, &port_promise]() {
      auto listeners = drogon::app().getListeners();
      if (!listeners.empty()) {
        bound_port_ = listeners[0].toPort();
      } else {
        bound_port_ = static_cast<uint16_t>(config_.port);
      }
      port_promise.set_value(bound_port_);
    });

    server_thread_ = std::thread([]() { drogon::app().run(); });

    bound_port_ = port_future.get();
    running_.store(true);
    return Result<void>::success();
  }

  Result<void> start() {
    auto res = startAsync();
    if (!res.isOk()) {
      return res;
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    return Result<void>::success();
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_.exchange(false)) {
      return;
    }
    if (drogon::app().isRunning()) {
      drogon::app().getLoop()->queueInLoop([]() { drogon::app().quit(); });
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    bound_port_ = 0;
  }

  [[nodiscard]] bool isRunning() const noexcept { return running_.load(); }

  [[nodiscard]] uint16_t port() const noexcept {
    if (running_.load() && bound_port_ != 0) {
      return bound_port_;
    }
    return static_cast<uint16_t>(config_.port);
  }

  [[nodiscard]] const std::string& host() const noexcept { return config_.host; }

  [[nodiscard]] const ServerConfig& config() const noexcept { return config_; }

  [[nodiscard]] ThreadPool& threadPool() noexcept { return pool_; }

  ServerConfig config_;
  ThreadPool pool_;
  std::thread server_thread_;
  std::mutex mutex_;
  std::atomic<bool> running_{false};
  uint16_t bound_port_{0};
};

HttpServer::HttpServer(const ServerConfig& config) : impl_(std::make_unique<Impl>(config)) {}

HttpServer::~HttpServer() = default;

HttpServer::HttpServer(HttpServer&&) noexcept = default;
HttpServer& HttpServer::operator=(HttpServer&&) noexcept = default;

Result<void> HttpServer::startAsync() {
  return impl_->startAsync();
}

Result<void> HttpServer::start() {
  return impl_->start();
}

void HttpServer::stop() {
  impl_->stop();
}

bool HttpServer::isRunning() const noexcept {
  return impl_->isRunning();
}

uint16_t HttpServer::port() const noexcept {
  return impl_->port();
}

const std::string& HttpServer::host() const noexcept {
  return impl_->host();
}

const ServerConfig& HttpServer::config() const noexcept {
  return impl_->config();
}

ThreadPool& HttpServer::threadPool() noexcept {
  return impl_->threadPool();
}

}  // namespace pyreth::server
