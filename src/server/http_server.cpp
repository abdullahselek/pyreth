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
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/server/binary_protocol.hpp>
#include <pyreth/server/endpoint.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/protocol.hpp>
#include <pyreth/server/streaming.hpp>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace pyreth::server {

class HttpServer::Impl {
private:
  static std::atomic<Impl*>& activeInstance() noexcept {
    static std::atomic<Impl*> instance{nullptr};
    return instance;
  }

public:
  explicit Impl(const ServerConfig& config)
      : config_(config),
        pool_(std::make_unique<ThreadPool>(config.io_threads > 0 ? config.io_threads : 1)),
        memory_pool_(16, 2 * 1024 * 1024) {}

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

    if (!pool_ || pool_->isShutdown()) {
      pool_ = std::make_unique<ThreadPool>(config_.io_threads > 0 ? config_.io_threads : 1);
    }

    // Configure Drogon
    uint16_t listen_port = static_cast<uint16_t>(config_.port);
    if (listen_port == 0) {
      std::string bind_ip =
          (config_.host == "0.0.0.0" || config_.host.empty()) ? "127.0.0.1" : config_.host;
#ifdef _WIN32
      SOCKET probe_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (probe_sock != INVALID_SOCKET) {
        int opt = 1;
        ::setsockopt(probe_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt),
                     sizeof(opt));
        sockaddr_in probe_addr{};
        probe_addr.sin_family = AF_INET;
        ::inet_pton(AF_INET, bind_ip.c_str(), &probe_addr.sin_addr);
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
        ::inet_pton(AF_INET, bind_ip.c_str(), &probe_addr.sin_addr);
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

    // Initialize WebSocketServerContext for WebSocket controllers
    WebSocketServerContext::instance().init(&registry_, &memory_pool_, pool_.get());

    // Register SSE streaming endpoints (Phase 5)
    auto sse_handler = [](const drogon::HttpRequestPtr& req,
                          std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                          std::string model_name) {
      auto* active = activeInstance().load(std::memory_order_acquire);
      if (!active) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        callback(resp);
        return;
      }
      active->handleSSEStreamRequest(req, std::move(callback), std::move(model_name));
    };

    drogon::app().registerHandlerViaRegex(R"(/v1/models/([^/]+)/stream)", sse_handler,
                                          {drogon::Get, drogon::Post});

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
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
          auto* active = activeInstance().load(std::memory_order_acquire);
          if (!active) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            callback(resp);
            return;
          }
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          auto models = active->registry_.listModels();
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

    // Register metrics endpoint (Phase 4)
    drogon::app().registerHandlerViaRegex(
        R"(/v1/models/([^/]+)/metrics)",
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback, std::string model_name) {
          auto* active = activeInstance().load(std::memory_order_acquire);
          if (!active) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            callback(resp);
            return;
          }
          active->handleMetricsRequest(std::move(callback), std::move(model_name));
        },
        {drogon::Get});

    drogon::app().registerHandlerViaRegex(
        R"(/v1/models/([^/]+))",
        [](const drogon::HttpRequestPtr& /*req*/,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback, std::string model_name) {
          auto* active = activeInstance().load(std::memory_order_acquire);
          if (!active) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            callback(resp);
            return;
          }
          auto entry_res = active->registry_.getModelEntry(model_name);
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

          InferenceSession* session_ptr =
              entry.session ? entry.session.get()
                            : (entry.batcher ? entry.batcher->session() : nullptr);
          if (session_ptr) {
            const auto& info = session_ptr->getModelInfo();
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

    // Register inference endpoints (Phase 3)
    drogon::app().registerHandlerViaRegex(
        R"(/v1/models/([^/]+)/infer)",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback, std::string model_name) {
          auto* active = activeInstance().load(std::memory_order_acquire);
          if (!active) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            callback(resp);
            return;
          }
          active->handleInferRequest(req, std::move(callback), std::move(model_name), "");
        },
        {drogon::Post});

    drogon::app().registerHandlerViaRegex(
        R"(/v1/models/([^/]+)/versions/([^/]+)/infer)",
        [](const drogon::HttpRequestPtr& req,
           std::function<void(const drogon::HttpResponsePtr&)>&& callback, std::string model_name,
           std::string model_version) {
          auto* active = activeInstance().load(std::memory_order_acquire);
          if (!active) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            callback(resp);
            return;
          }
          active->handleInferRequest(req, std::move(callback), std::move(model_name),
                                     std::move(model_version));
        },
        {drogon::Post});

    // Register custom routes (Phase 5)
    for (const auto& builder : custom_routes_) {
      drogon::HttpMethod method = drogon::Get;
      switch (builder.httpMethod()) {
        case HttpMethod::Get:
          method = drogon::Get;
          break;
        case HttpMethod::Post:
          method = drogon::Post;
          break;
        case HttpMethod::Put:
          method = drogon::Put;
          break;
        case HttpMethod::Delete:
          method = drogon::Delete;
          break;
        case HttpMethod::Options:
          method = drogon::Options;
          break;
      }

      drogon::app().registerHandler(
          builder.path(),
          [this, builder](const drogon::HttpRequestPtr& req,
                          std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
            handleCustomEndpointRequest(req, std::move(callback), builder);
          },
          {method});
    }

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
      uint16_t actual_port = listen_port;
      auto listeners = drogon::app().getListeners();
      if (!listeners.empty() && listeners[0].toPort() != 0) {
        actual_port = listeners[0].toPort();
      }
      bound_port_.store(actual_port, std::memory_order_release);
      port_promise.set_value(actual_port);
    });

    server_thread_ = std::thread([]() { drogon::app().run(); });

    uint16_t actual_port = port_future.get();
    bound_port_.store(actual_port, std::memory_order_release);
    activeInstance().store(this, std::memory_order_release);
    running_.store(true, std::memory_order_release);

    // Wait until listener socket is ready to accept connections (up to 2 seconds)
    auto start_time = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start_time < std::chrono::seconds(2)) {
#ifdef _WIN32
      SOCKET test_s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (test_s != INVALID_SOCKET) {
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(bound_port_.load(std::memory_order_acquire));
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
        a.sin_port = htons(bound_port_.load(std::memory_order_acquire));
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
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

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
    if (!running_.exchange(false, std::memory_order_acq_rel)) {
      return;
    }
    Impl* expected = this;
    activeInstance().compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
    if (pool_) {
      pool_->shutdown();
    }
    if (drogon::app().isRunning()) {
      drogon::app().quit();
    }
    if (server_thread_.joinable()) {
      server_thread_.join();
    }
    bound_port_.store(0, std::memory_order_release);
  }

  [[nodiscard]] bool isRunning() const noexcept { return running_.load(std::memory_order_acquire); }

  [[nodiscard]] uint16_t port() const noexcept {
    uint16_t actual_port = bound_port_.load(std::memory_order_acquire);
    if (running_.load(std::memory_order_acquire) && actual_port != 0) {
      return actual_port;
    }
    return static_cast<uint16_t>(config_.port);
  }

  [[nodiscard]] const std::string& host() const noexcept { return config_.host; }

  [[nodiscard]] const ServerConfig& config() const noexcept { return config_; }

  [[nodiscard]] ThreadPool& threadPool() noexcept { return *pool_; }

  [[nodiscard]] MemoryPool& memoryPool() noexcept { return memory_pool_; }

  [[nodiscard]] const MemoryPool& memoryPool() const noexcept { return memory_pool_; }

  [[nodiscard]] Result<void> registerModel(std::string name,
                                           std::shared_ptr<InferenceSession> session,
                                           std::string version = "1") {
    return registry_.registerModel(std::move(name), std::move(session), std::move(version));
  }

  [[nodiscard]] Result<void> registerBatcher(std::string name,
                                             std::shared_ptr<DynamicBatcher> batcher) {
    return registry_.registerBatcher(std::move(name), std::move(batcher));
  }

  [[nodiscard]] Result<void> unregisterBatcher(std::string_view name) {
    return registry_.unregisterBatcher(name);
  }

  [[nodiscard]] Result<void> unregisterModel(std::string_view name) {
    return registry_.unregisterModel(name);
  }

  [[nodiscard]] bool hasModel(std::string_view name) const { return registry_.hasModel(name); }

  [[nodiscard]] ModelRegistry& modelRegistry() noexcept { return registry_; }
  [[nodiscard]] const ModelRegistry& modelRegistry() const noexcept { return registry_; }

  EndpointBuilder& route(std::string path) {
    custom_routes_.emplace_back(std::move(path));
    return custom_routes_.back();
  }

private:
  void handleMetricsRequest(std::function<void(const drogon::HttpResponsePtr&)>&& callback,
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
    if (!entry.batcher) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k404NotFound);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      Json::Value err;
      err["error"] = "Dynamic batcher metrics not available for model '" + model_name + "'";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      resp->setBody(Json::writeString(writer, err));
      callback(resp);
      return;
    }

    auto metrics = entry.batcher->metrics();
    Json::Value root;
    root["model_name"] = model_name;
    root["requests_submitted"] = static_cast<Json::Value::UInt64>(metrics.requests_submitted);
    root["requests_completed"] = static_cast<Json::Value::UInt64>(metrics.requests_completed);
    root["requests_rejected"] = static_cast<Json::Value::UInt64>(metrics.requests_rejected);
    root["batches_dispatched"] = static_cast<Json::Value::UInt64>(metrics.batches_dispatched);
    root["full_batches"] = static_cast<Json::Value::UInt64>(metrics.full_batches);
    root["timeout_batches"] = static_cast<Json::Value::UInt64>(metrics.timeout_batches);
    root["total_batched_samples"] = static_cast<Json::Value::UInt64>(metrics.total_batched_samples);
    root["avg_batch_size"] = metrics.averageBatchSize();
    root["average_batch_size"] = metrics.averageBatchSize();
    root["avg_queue_time_ms"] = metrics.averageBatchWaitUs() / 1000.0;
    root["avg_exec_time_ms"] = metrics.averageExecutionUs() / 1000.0;
    root["total_batch_wait_us"] = static_cast<Json::Value::UInt64>(metrics.total_batch_wait_us);
    root["total_execution_us"] = static_cast<Json::Value::UInt64>(metrics.total_execution_us);

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    Json::StreamWriterBuilder writer;
    writer["indentation"] = "";
    resp->setBody(Json::writeString(writer, root));
    callback(resp);
  }

  void handleSSEStreamRequest(const drogon::HttpRequestPtr& req,
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

    auto session = entry_res.value().session;
    if (!session) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k400BadRequest);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      Json::Value err;
      err["error"] = "Model '" + model_name + "' has no session configured";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      resp->setBody(Json::writeString(writer, err));
      callback(resp);
      return;
    }

    auto body = std::string(req->body());

    // Weak connection pointer to detect early client disconnect
    const auto& weakConnPtr = req->getConnectionPtr();

    auto resp = drogon::HttpResponse::newAsyncStreamResponse(
        [this, session, body = std::move(body), weakConnPtr](drogon::ResponseStreamPtr stream) {
          auto submit_res = pool_->submit([session, body = std::move(body), weakConnPtr,
                                           stream = std::move(stream)]() mutable {
            auto is_conn_closed = [&weakConnPtr]() -> bool {
              auto conn = weakConnPtr.lock();
              return (!conn || conn->disconnected());
            };

            // Stream incremental step tokens/chunks
            for (int step = 0; step < 3; ++step) {
              if (is_conn_closed()) {
                // Client early disconnect detected, abort cleanly
                return;
              }

              Json::Value step_data;
              step_data["step"] = step;
              step_data["token"] = "step_" + std::to_string(step);
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              std::string frame = formatSSEEvent("step", Json::writeString(writer, step_data));
              if (!stream->send(frame)) {
                return;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }

            if (is_conn_closed()) {
              return;
            }

            // Terminal event frame
            Json::Value done_data;
            done_data["status"] = "finished";
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            std::string done_frame = formatSSEEvent("done", Json::writeString(writer, done_data));
            stream->send(done_frame);
            stream->close();
          });
          (void)submit_res;
        });

    resp->setContentTypeCodeAndCustomString(drogon::CT_CUSTOM, "text/event-stream");
    resp->addHeader("Cache-Control", "no-cache");
    resp->addHeader("Connection", "keep-alive");
    callback(resp);
  }

  void handleCustomEndpointRequest(const drogon::HttpRequestPtr& req,
                                   std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                                   const EndpointBuilder& builder) {
    auto entry_res = registry_.getModelEntry(builder.modelName());
    if (!builder.modelName().empty() && !entry_res.isOk()) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k404NotFound);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      Json::Value err;
      err["error"] = "Model '" + builder.modelName() + "' not found";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      resp->setBody(Json::writeString(writer, err));
      callback(resp);
      return;
    }

    std::shared_ptr<InferenceSession> session;
    if (entry_res.isOk()) {
      session = entry_res.value().session;
    }

    HttpMethod meth = HttpMethod::Get;
    switch (req->method()) {
      case drogon::Get:
        meth = HttpMethod::Get;
        break;
      case drogon::Post:
        meth = HttpMethod::Post;
        break;
      case drogon::Put:
        meth = HttpMethod::Put;
        break;
      case drogon::Delete:
        meth = HttpMethod::Delete;
        break;
      case drogon::Options:
        meth = HttpMethod::Options;
        break;
      default:
        meth = HttpMethod::Get;
        break;
    }

    std::unordered_map<std::string, std::string> headers;
    for (const auto& [k, v] : req->headers()) {
      headers.emplace(k, v);
    }

    CustomHttpRequest custom_req(meth, req->path(), std::string(req->body()), std::move(headers));

    auto submit_res = pool_->submit([this, session, builder, custom_req = std::move(custom_req),
                                     cb = std::move(callback)]() mutable {
      auto lease = memory_pool_.acquire();
      ArenaAllocator& arena = lease.get();

      std::vector<Tensor> input_tensors;
      if (builder.preprocessFn()) {
        auto prep_res = builder.preprocessFn()(custom_req, arena);
        if (!prep_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k422UnprocessableEntity);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(prep_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          cb(resp);
          return;
        }
        input_tensors = std::move(prep_res.value());
      }

      std::vector<Tensor> output_tensors;
      if (session) {
        auto run_res = session->run(input_tensors);
        if (!run_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k500InternalServerError);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(run_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          cb(resp);
          return;
        }
        output_tensors = std::move(run_res.value());
      }

      CustomHttpResponse custom_resp(200);
      if (builder.postprocessFn()) {
        auto post_res = builder.postprocessFn()(output_tensors, custom_req);
        if (!post_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k500InternalServerError);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(post_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          cb(resp);
          return;
        }
        custom_resp = std::move(post_res.value());
      }

      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(static_cast<drogon::HttpStatusCode>(custom_resp.statusCode()));
      resp->setContentTypeCodeAndCustomString(drogon::CT_CUSTOM, custom_resp.contentType());
      resp->setBody(custom_resp.body());
      for (const auto& [hk, hv] : custom_resp.headers()) {
        resp->addHeader(hk, hv);
      }
      cb(resp);
    });
    (void)submit_res;
  }

  void handleInferRequest(const drogon::HttpRequestPtr& req,
                          std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                          std::string model_name, std::string model_version) {
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
    if ((!entry.session && !entry.batcher) || entry.state != "ready") {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k503ServiceUnavailable);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      Json::Value err;
      err["error"] = "Model '" + model_name + "' is not ready";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      resp->setBody(Json::writeString(writer, err));
      callback(resp);
      return;
    }

    std::string active_version = !model_version.empty() ? model_version : entry.version;
    std::shared_ptr<InferenceSession> session = entry.session;

    ExecutionMode mode = entry.batcher ? ExecutionMode::DynamicBatching : ExecutionMode::Async;
    std::string mode_str = req->getHeader("x-pyreth-execution-mode");
    if (mode_str.empty()) {
      mode_str = req->getHeader("X-Pyreth-Execution-Mode");
    }
    if (!mode_str.empty()) {
      auto mode_res = stringToExecutionMode(mode_str);
      if (!mode_res.isOk()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = std::string(mode_res.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        callback(resp);
        return;
      }
      mode = mode_res.value();
    }

    std::string content_type = req->contentType() == drogon::CT_APPLICATION_OCTET_STREAM
                                   ? "application/octet-stream"
                                   : req->getHeader("content-type");
    bool is_binary = (content_type.find("application/octet-stream") != std::string::npos);

    if (mode == ExecutionMode::DynamicBatching) {
      if (!entry.batcher || !entry.batcher->isValid()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = "Dynamic batcher is not configured for model '" + model_name + "'";
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        callback(resp);
        return;
      }

      std::vector<Tensor> tensors;
      if (is_binary) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(req->bodyData()),
                                       req->bodyLength());
        auto bin_res = deserializeBinaryTensor(bytes);
        if (!bin_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k400BadRequest);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(bin_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          callback(resp);
          return;
        }
        tensors.push_back(std::move(bin_res.value()));
      } else {
        auto lease = memory_pool_.acquire();
        ArenaAllocator& arena = lease.get();

        auto req_res = parseKServeRequest(req->body(), arena);
        if (!req_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k400BadRequest);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(req_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          callback(resp);
          return;
        }
        tensors = req_res.value().extractTensors();
      }

      std::vector<std::string> output_names;
      if (session) {
        output_names = session->getOutputNames();
      } else if (entry.batcher->session()) {
        output_names = entry.batcher->session()->getOutputNames();
      }

      auto cancelled = std::make_shared<std::atomic<bool>>(false);
      auto responded = std::make_shared<std::atomic<bool>>(false);
      auto cb = std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(
          std::move(callback));

      trantor::TimerId timer_id{};
      auto loop = trantor::EventLoop::getEventLoopOfCurrentThread();
      if (!loop) {
        loop = drogon::app().getLoop();
      }
      if (config_.request_timeout_ms > 0 && loop) {
        double delay_sec = static_cast<double>(config_.request_timeout_ms) / 1000.0;
        timer_id = loop->runAfter(delay_sec, [cb, responded, cancelled]() {
          cancelled->store(true, std::memory_order_release);
          if (!responded->exchange(true, std::memory_order_acq_rel)) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k504GatewayTimeout);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            Json::Value err;
            err["error"] = "Request execution timed out";
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            resp->setBody(Json::writeString(writer, err));
            (*cb)(resp);
          }
        });
      }

      auto send_response = [loop, timer_id, cb](const drogon::HttpResponsePtr& resp) {
        if (loop && loop->isRunning()) {
          loop->runInLoop([loop, timer_id, cb, resp]() {
            if (timer_id != trantor::TimerId{}) {
              loop->invalidateTimer(timer_id);
            }
            (*cb)(resp);
          });
        } else {
          if (loop && timer_id != trantor::TimerId{}) {
            loop->invalidateTimer(timer_id);
          }
          (*cb)(resp);
        }
      };

      auto completion_cb = [this, m_name = std::move(model_name), ver = std::move(active_version),
                            output_names = std::move(output_names), cancelled, responded,
                            send_response, is_binary](Result<std::vector<Tensor>> run_res) {
        if (cancelled->load(std::memory_order_acquire) ||
            !running_.load(std::memory_order_acquire)) {
          if (!responded->exchange(true, std::memory_order_acq_rel)) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            resp->setCloseConnection(true);
            send_response(resp);
          }
          return;
        }

        if (!run_res.isOk()) {
          if (!responded->exchange(true, std::memory_order_acq_rel)) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            Json::Value err;
            err["error"] = std::string(run_res.error().message());
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            resp->setBody(Json::writeString(writer, err));
            send_response(resp);
          }
          return;
        }

        if (is_binary) {
          const auto& out_tensors = run_res.value();
          if (out_tensors.empty()) {
            if (!responded->exchange(true, std::memory_order_acq_rel)) {
              auto resp = drogon::HttpResponse::newHttpResponse();
              resp->setStatusCode(drogon::k500InternalServerError);
              resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
              Json::Value err;
              err["error"] = "Model produced no outputs";
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              resp->setBody(Json::writeString(writer, err));
              send_response(resp);
            }
            return;
          }

          auto ser_bin = serializeBinaryTensor(out_tensors[0]);
          if (!ser_bin.isOk()) {
            if (!responded->exchange(true, std::memory_order_acq_rel)) {
              auto resp = drogon::HttpResponse::newHttpResponse();
              resp->setStatusCode(drogon::k500InternalServerError);
              resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
              Json::Value err;
              err["error"] = std::string(ser_bin.error().message());
              Json::StreamWriterBuilder writer;
              writer["indentation"] = "";
              resp->setBody(Json::writeString(writer, err));
              send_response(resp);
            }
            return;
          }

          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k200OK);
          resp->setContentTypeCodeAndCustomString(drogon::CT_APPLICATION_OCTET_STREAM,
                                                  "application/octet-stream");
          resp->setBody(std::string(reinterpret_cast<const char*>(ser_bin.value().data()),
                                    ser_bin.value().size()));
          resp->addHeader("X-Pyreth-Execution-Mode", "batch");
          if (!responded->exchange(true, std::memory_order_acq_rel)) {
            send_response(resp);
          }
          return;
        }

        auto ser_res = serializeKServeResponse(m_name, output_names, run_res.value(), ver);
        if (!ser_res.isOk()) {
          if (!responded->exchange(true, std::memory_order_acq_rel)) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k500InternalServerError);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            Json::Value err;
            err["error"] = std::string(ser_res.error().message());
            Json::StreamWriterBuilder writer;
            writer["indentation"] = "";
            resp->setBody(Json::writeString(writer, err));
            send_response(resp);
          }
          return;
        }

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(std::move(ser_res.value()));
        resp->addHeader("X-Pyreth-Execution-Mode", "batch");

        std::ostringstream ss;
        ss << std::this_thread::get_id();
        resp->addHeader("X-Pyreth-Worker-Thread-Id", ss.str());

        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          send_response(resp);
        }
      };

      auto submit_status = entry.batcher->submit(tensors, std::move(completion_cb));
      if (!submit_status.isOk()) {
        if (timer_id != trantor::TimerId{} && loop) {
          loop->invalidateTimer(timer_id);
        }
        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          if (submit_status.error().code() == ErrorCode::QueueFull) {
            resp->setStatusCode(drogon::k429TooManyRequests);
            resp->addHeader("Retry-After", "1");
          } else if (submit_status.error().code() == ErrorCode::InvalidArgument ||
                     submit_status.error().code() == ErrorCode::InvalidDimensions) {
            resp->setStatusCode(drogon::k400BadRequest);
          } else {
            resp->setStatusCode(drogon::k503ServiceUnavailable);
          }
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(submit_status.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          (*cb)(resp);
        }
      }
      return;
    }

    if (!session) {
      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k400BadRequest);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      Json::Value err;
      err["error"] = "Model '" + model_name + "' has no InferenceSession configured";
      Json::StreamWriterBuilder writer;
      writer["indentation"] = "";
      resp->setBody(Json::writeString(writer, err));
      callback(resp);
      return;
    }

    int64_t simulated_delay_ms = 0;
    std::string delay_str = req->getHeader("x-pyreth-simulated-delay-ms");
    if (delay_str.empty()) {
      delay_str = req->getHeader("X-Pyreth-Simulated-Delay-Ms");
    }
    if (delay_str.empty()) {
      delay_str = req->getHeader("x-simulated-delay-ms");
    }
    if (delay_str.empty()) {
      delay_str = req->getHeader("X-Simulated-Delay-Ms");
    }
    if (!delay_str.empty()) {
      try {
        simulated_delay_ms = std::stoll(delay_str);
      } catch (...) {}
    }

    std::string body = std::string(req->body());

    auto execute_inference =
        [this, session, m_name = std::move(model_name), ver = std::move(active_version),
         body = std::move(body), mode, simulated_delay_ms, is_binary](
            const std::shared_ptr<std::atomic<bool>>& cancelled) -> drogon::HttpResponsePtr {
      if ((cancelled && cancelled->load(std::memory_order_acquire)) ||
          !running_.load(std::memory_order_acquire)) {
        return nullptr;
      }

      if (simulated_delay_ms > 0) {
        auto start_wait = std::chrono::steady_clock::now();
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start_wait)
                   .count() < simulated_delay_ms) {
          if ((cancelled && cancelled->load(std::memory_order_acquire)) ||
              !running_.load(std::memory_order_acquire)) {
            return nullptr;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
      }

      if ((cancelled && cancelled->load(std::memory_order_acquire)) ||
          !running_.load(std::memory_order_acquire)) {
        return nullptr;
      }

      std::vector<Tensor> tensors;
      if (is_binary) {
        std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(body.data()), body.size());
        auto bin_res = deserializeBinaryTensor(bytes);
        if (!bin_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k400BadRequest);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(bin_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          return resp;
        }
        tensors.push_back(std::move(bin_res.value()));
      } else {
        auto lease = memory_pool_.acquire();
        ArenaAllocator& arena = lease.get();

        auto req_res = parseKServeRequest(body, arena);
        if (!req_res.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k400BadRequest);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(req_res.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          return resp;
        }

        tensors = req_res.value().extractTensors();
      }

      auto val_status = session->validateInputs(tensors);
      if (!val_status.isOk()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k400BadRequest);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = std::string(val_status.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        return resp;
      }

      if ((cancelled && cancelled->load(std::memory_order_acquire)) ||
          !running_.load(std::memory_order_acquire)) {
        return nullptr;
      }

      auto run_res = session->run(tensors);
      if (!run_res.isOk()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = std::string(run_res.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        return resp;
      }

      if (is_binary) {
        const auto& out_tensors = run_res.value();
        if (out_tensors.empty()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k500InternalServerError);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = "Model produced no outputs";
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          return resp;
        }

        auto ser_bin = serializeBinaryTensor(out_tensors[0]);
        if (!ser_bin.isOk()) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k500InternalServerError);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = std::string(ser_bin.error().message());
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          return resp;
        }

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeCodeAndCustomString(drogon::CT_APPLICATION_OCTET_STREAM,
                                                "application/octet-stream");
        resp->setBody(std::string(reinterpret_cast<const char*>(ser_bin.value().data()),
                                  ser_bin.value().size()));
        resp->addHeader("X-Pyreth-Execution-Mode", std::string(executionModeToString(mode)));
        std::ostringstream ss;
        ss << std::this_thread::get_id();
        resp->addHeader("X-Pyreth-Worker-Thread-Id", ss.str());
        return resp;
      }

      auto output_names = session->getOutputNames();
      auto ser_res = serializeKServeResponse(m_name, output_names, run_res.value(), ver);
      if (!ser_res.isOk()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = std::string(ser_res.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        return resp;
      }

      auto resp = drogon::HttpResponse::newHttpResponse();
      resp->setStatusCode(drogon::k200OK);
      resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
      resp->setBody(std::move(ser_res.value()));
      resp->addHeader("X-Pyreth-Execution-Mode", std::string(executionModeToString(mode)));

      std::ostringstream ss;
      ss << std::this_thread::get_id();
      resp->addHeader("X-Pyreth-Worker-Thread-Id", ss.str());

      return resp;
    };

    if (mode == ExecutionMode::Sync) {
      if (config_.request_timeout_ms > 0 && simulated_delay_ms > config_.request_timeout_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.request_timeout_ms));
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k504GatewayTimeout);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = "Request execution timed out";
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        callback(resp);
        return;
      }

      auto resp = execute_inference(nullptr);
      if (resp) {
        callback(resp);
      }
      return;
    }

    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    auto responded = std::make_shared<std::atomic<bool>>(false);
    auto cb =
        std::make_shared<std::function<void(const drogon::HttpResponsePtr&)>>(std::move(callback));

    trantor::TimerId timer_id{};
    auto loop = trantor::EventLoop::getEventLoopOfCurrentThread();
    if (!loop) {
      loop = drogon::app().getLoop();
    }
    if (config_.request_timeout_ms > 0 && loop) {
      double delay_sec = static_cast<double>(config_.request_timeout_ms) / 1000.0;
      timer_id = loop->runAfter(delay_sec, [cb, responded, cancelled]() {
        cancelled->store(true, std::memory_order_release);
        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k504GatewayTimeout);
          resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
          Json::Value err;
          err["error"] = "Request execution timed out";
          Json::StreamWriterBuilder writer;
          writer["indentation"] = "";
          resp->setBody(Json::writeString(writer, err));
          (*cb)(resp);
        }
      });
    }

    auto submit_res = pool_->submit([this, execute_inference, cancelled, responded, cb, timer_id,
                                     loop]() {
      auto send_response = [loop, timer_id, cb](const drogon::HttpResponsePtr& resp) {
        if (loop && loop->isRunning()) {
          loop->runInLoop([loop, timer_id, cb, resp]() {
            if (timer_id != trantor::TimerId{}) {
              loop->invalidateTimer(timer_id);
            }
            (*cb)(resp);
          });
        } else {
          if (loop && timer_id != trantor::TimerId{}) {
            loop->invalidateTimer(timer_id);
          }
          (*cb)(resp);
        }
      };

      if (cancelled->load(std::memory_order_acquire) || !running_.load(std::memory_order_acquire)) {
        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          auto resp = drogon::HttpResponse::newHttpResponse();
          resp->setStatusCode(drogon::k503ServiceUnavailable);
          resp->setCloseConnection(true);
          send_response(resp);
        }
        return;
      }

      auto resp = execute_inference(cancelled);
      if (!resp) {
        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          auto err_resp = drogon::HttpResponse::newHttpResponse();
          err_resp->setStatusCode(drogon::k503ServiceUnavailable);
          err_resp->setCloseConnection(true);
          send_response(err_resp);
        }
        return;
      }

      if (!running_.load(std::memory_order_acquire)) {
        if (!responded->exchange(true, std::memory_order_acq_rel)) {
          resp->setCloseConnection(true);
          send_response(resp);
        }
        return;
      }

      if (!responded->exchange(true, std::memory_order_acq_rel)) {
        send_response(resp);
      }
    });

    if (!submit_res.isOk()) {
      if (timer_id != trantor::TimerId{} && loop) {
        loop->invalidateTimer(timer_id);
      }
      if (!responded->exchange(true, std::memory_order_acq_rel)) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k503ServiceUnavailable);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        Json::Value err;
        err["error"] = std::string(submit_res.error().message());
        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        resp->setBody(Json::writeString(writer, err));
        (*cb)(resp);
      }
    }
  }

  ServerConfig config_;
  std::unique_ptr<ThreadPool> pool_;
  MemoryPool memory_pool_;
  ModelRegistry registry_;
  std::vector<EndpointBuilder> custom_routes_;
  std::thread server_thread_;
  std::mutex mutex_;
  std::atomic<bool> running_{false};
  std::atomic<uint16_t> bound_port_{0};
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

MemoryPool& HttpServer::memoryPool() noexcept {
  return impl_->memoryPool();
}

const MemoryPool& HttpServer::memoryPool() const noexcept {
  return impl_->memoryPool();
}

Result<void> HttpServer::registerModel(std::string name, std::shared_ptr<InferenceSession> session,
                                       std::string version) {
  return impl_->registerModel(std::move(name), std::move(session), std::move(version));
}

Result<void> HttpServer::registerBatcher(std::string name,
                                         std::shared_ptr<DynamicBatcher> batcher) {
  return impl_->registerBatcher(std::move(name), std::move(batcher));
}

Result<void> HttpServer::unregisterBatcher(std::string_view name) {
  return impl_->unregisterBatcher(name);
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

EndpointBuilder& HttpServer::route(std::string path) {
  return impl_->route(std::move(path));
}

}  // namespace pyreth::server
