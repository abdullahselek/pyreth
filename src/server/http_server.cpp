#include <drogon/drogon.h>
#include <json/json.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <pyreth/common.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/protocol.hpp>
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
    uint16_t listen_port = static_cast<uint16_t>(config_.port);
    if (listen_port == 0) {
#ifdef _WIN32
      SOCKET probe_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (probe_sock != INVALID_SOCKET) {
        int opt = 1;
        ::setsockopt(probe_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt),
                     sizeof(opt));
        sockaddr_in probe_addr{};
        probe_addr.sin_family = AF_INET;
        probe_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        probe_addr.sin_port = 0;
        if (::bind(probe_sock, reinterpret_cast<sockaddr*>(&probe_addr), sizeof(probe_addr)) == 0) {
          int len = sizeof(probe_addr);
          if (::getsockname(probe_sock, reinterpret_cast<sockaddr*>(&probe_addr), &len) == 0) {
            listen_port = ntohs(probe_addr.sin_port);
          }
        }
        ::closesocket(probe_sock);
      }
#else
      int probe_sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (probe_sock >= 0) {
        int opt = 1;
        ::setsockopt(probe_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifdef SO_REUSEPORT
        ::setsockopt(probe_sock, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
        sockaddr_in probe_addr{};
        probe_addr.sin_family = AF_INET;
        probe_addr.sin_addr.s_addr = htonl(INADDR_ANY);
        probe_addr.sin_port = 0;
        if (::bind(probe_sock, reinterpret_cast<sockaddr*>(&probe_addr), sizeof(probe_addr)) == 0) {
          socklen_t len = sizeof(probe_addr);
          if (::getsockname(probe_sock, reinterpret_cast<sockaddr*>(&probe_addr), &len) == 0) {
            listen_port = ntohs(probe_addr.sin_port);
          }
        }
        ::close(probe_sock);
      }
#endif
    }

    drogon::app().addListener(config_.host, listen_port);
    drogon::app().setThreadNum(config_.io_threads);
    drogon::app().enableReusePort(true);
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

    // Register model discovery endpoints (Phase 2)
    drogon::app().registerHandler(
        "/v1/models",
        [this](const drogon::HttpRequestPtr& /*req*/,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          auto models = registry_.listModels();
          Json::Value root(Json::arrayValue);
          for (const auto& m : models) {
            Json::Value item;
            item["name"] = m.name;
            item["version"] = m.version;
            item["ready"] = (m.state == "ready");
            root.append(item);
          }
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, root));
          callback(resp);
        },
        {drogon::Get});

    drogon::app().registerHandlerViaRegex(
        R"(/v1/models/([^/]+))",
        [this](const drogon::HttpRequestPtr& /*req*/,
               std::function<void(const drogon::HttpResponsePtr&)>&& callback,
               std::string model_name) {
          auto entry_res = registry_.getModelEntry(model_name);
          if (!entry_res.isOk()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k404NotFound);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            Json::Value err;
            err["error"] = "Model '" + model_name + "' not found";
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            resp->setBody(Json::writeString(writer, err));
            callback(resp);
            return;
          }

          const auto& entry = entry_res.value();
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);

          Json::Value root;
          root["name"] = entry.name;
          Json::Value versions(Json::arrayValue);
          versions.append(entry.version);
          root["versions"] = versions;
          root["platform"] = "onnxruntime";
          root["inputs"] = Json::arrayValue;
          root["outputs"] = Json::arrayValue;

          if (entry.session) {
            const auto& info = entry.session->getModelInfo();
            for (const auto& inp : info.inputs) {
              Json::Value inp_val;
              inp_val["name"] = inp.name;
              inp_val["datatype"] = dataTypeToKServeString(inp.dtype);
              Json::Value shape_val(Json::arrayValue);
              for (int64_t d : inp.shape) {
                shape_val.append(static_cast<Json::Value::Int64>(d));
              }
              inp_val["shape"] = shape_val;
              root["inputs"].append(inp_val);
            }
            for (const auto& out : info.outputs) {
              Json::Value out_val;
              out_val["name"] = out.name;
              out_val["datatype"] = dataTypeToKServeString(out.dtype);
              Json::Value shape_val(Json::arrayValue);
              for (int64_t d : out.shape) {
                shape_val.append(static_cast<Json::Value::Int64>(d));
              }
              out_val["shape"] = shape_val;
              root["outputs"].append(out_val);
            }
          }

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

    drogon::app().registerBeginningAdvice([this, &port_promise, listen_port]() {
      auto listeners = drogon::app().getListeners();
      if (!listeners.empty() && listeners[0].toPort() != 0) {
        bound_port_ = listeners[0].toPort();
      } else {
        bound_port_ = listen_port;
      }
      port_promise.set_value(bound_port_);
    });

    server_thread_ = std::thread([]() { drogon::app().run(); });

    bound_port_ = port_future.get();
    running_.store(true);

    // Wait until listener socket is ready to accept connections (up to 2 seconds)
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(2)) {
#ifdef _WIN32
      SOCKET test_s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (test_s != INVALID_SOCKET) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(bound_port_);
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::connect(test_s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
          ::closesocket(test_s);
          break;
        }
        ::closesocket(test_s);
      }
#else
      int test_s = ::socket(AF_INET, SOCK_STREAM, 0);
      if (test_s >= 0) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(bound_port_);
        ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::connect(test_s, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
          ::close(test_s);
          break;
        }
        ::close(test_s);
      }
#endif
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

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
      drogon::app().quit();
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

  [[nodiscard]] Result<void> registerModel(std::string name,
                                           std::shared_ptr<InferenceSession> session,
                                           std::string version = "1") {
    return registry_.registerModel(std::move(name), std::move(session), std::move(version));
  }

  [[nodiscard]] Result<void> registerBatcher(std::string name,
                                             std::shared_ptr<DynamicBatcher> batcher) {
    return registry_.registerBatcher(std::move(name), std::move(batcher));
  }

  [[nodiscard]] Result<void> unregisterModel(std::string_view name) {
    return registry_.unregisterModel(name);
  }

  [[nodiscard]] bool hasModel(std::string_view name) const { return registry_.hasModel(name); }

  [[nodiscard]] ModelRegistry& modelRegistry() noexcept { return registry_; }
  [[nodiscard]] const ModelRegistry& modelRegistry() const noexcept { return registry_; }

  ServerConfig config_;
  ThreadPool pool_;
  ModelRegistry registry_;
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

Result<void> HttpServer::registerModel(std::string name, std::shared_ptr<InferenceSession> session,
                                       std::string version) {
  return impl_->registerModel(std::move(name), std::move(session), std::move(version));
}

Result<void> HttpServer::registerBatcher(std::string name,
                                         std::shared_ptr<DynamicBatcher> batcher) {
  return impl_->registerBatcher(std::move(name), std::move(batcher));
}

Result<void> HttpServer::unregisterModel(std::string_view name) {
  return impl_->unregisterModel(name);
}

bool HttpServer::hasModel(std::string_view name) const {
  return impl_->hasModel(name);
}

ModelRegistry& HttpServer::modelRegistry() noexcept {
  return impl_->modelRegistry();
}

const ModelRegistry& HttpServer::modelRegistry() const noexcept {
  return impl_->modelRegistry();
}

}  // namespace pyreth::server
