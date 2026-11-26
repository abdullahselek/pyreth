#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>
#include <json/json.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <pyreth/inference_session.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/model_registry.hpp>
#include <pyreth/server/server_config.hpp>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace pyreth;
using namespace pyreth::server;

namespace {

#ifdef _WIN32
using socket_t = SOCKET;
inline constexpr socket_t kInvalidSocket = INVALID_SOCKET;
inline int closeSocket(socket_t s) {
  return ::closesocket(s);
}
inline bool isValidSocket(socket_t s) {
  return s != kInvalidSocket;
}

class WinsockScope {
public:
  WinsockScope() {
    WSADATA wsa_data;
    ::WSAStartup(MAKEWORD(2, 2), &wsa_data);
  }
  ~WinsockScope() { ::WSACleanup(); }
};
static WinsockScope g_winsock_scope;
#else
using socket_t = int;
inline constexpr socket_t kInvalidSocket = -1;
inline int closeSocket(socket_t s) {
  return ::close(s);
}
inline bool isValidSocket(socket_t s) {
  return s >= 0;
}
#endif

// Minimal valid ONNX model protobuf (Identity: x -> y, float32[1, 2])
constexpr uint8_t kValidModelData[] = {
    0x08, 0x09, 0x12, 0x0b, 0x70, 0x79, 0x72, 0x65, 0x74, 0x68, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x3a,
    0x57, 0x0a, 0x1f, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x1a, 0x0d, 0x69, 0x64, 0x65, 0x6e, 0x74,
    0x69, 0x74, 0x79, 0x5f, 0x6e, 0x6f, 0x64, 0x65, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69,
    0x74, 0x79, 0x12, 0x0a, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5a, 0x13,
    0x0a, 0x01, 0x78, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a,
    0x02, 0x08, 0x02, 0x62, 0x13, 0x0a, 0x01, 0x79, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08,
    0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x11};

std::shared_ptr<InferenceSession> createValidSession() {
  auto res =
      InferenceSession::create(std::span<const uint8_t>(kValidModelData, sizeof(kValidModelData)));
  if (!res.isOk()) {
    return nullptr;
  }
  return std::make_shared<InferenceSession>(std::move(res.value()));
}

struct HttpResponse {
  int status_code{0};
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

std::string toLower(std::string_view s) {
  std::string res;
  res.reserve(s.size());
  for (char c : s) {
    res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }
  return res;
}

HttpResponse sendHttpRequest(
    const std::string& host, uint16_t port, const std::string& method, const std::string& path,
    const std::unordered_map<std::string, std::string>& custom_headers = {},
    const std::string& body = "", int timeout_sec = 20, int retries = 10) {
  for (int attempt = 0; attempt <= retries; ++attempt) {
    if (attempt > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50 * attempt));
    }

    std::string target_ip = (host == "0.0.0.0" || host.empty()) ? "127.0.0.1" : host;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    ::inet_pton(AF_INET, target_ip.c_str(), &addr.sin_addr);

    socket_t sock = kInvalidSocket;
    bool connected = false;
    for (int conn_try = 0; conn_try < 100; ++conn_try) {
      sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (!isValidSocket(sock)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
#ifdef _WIN32
      DWORD timeout_ms = static_cast<DWORD>(timeout_sec * 1000);
      ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                   sizeof(timeout_ms));
      ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                   sizeof(timeout_ms));
#else
      struct timeval tv{};
      tv.tv_sec = timeout_sec;
      tv.tv_usec = 0;
      ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
      if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
        connected = true;
        break;
      }
      closeSocket(sock);
      sock = kInvalidSocket;
      std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    if (!connected || !isValidSocket(sock)) {
      continue;
    }

    std::string req = method + " " + path + " HTTP/1.1\r\n";
    req += "Host: " + target_ip + ":" + std::to_string(port) + "\r\n";
    req += "Connection: close\r\n";
    if (!body.empty() || method == "POST" || method == "PUT") {
      req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    for (const auto& [k, v] : custom_headers) {
      req += k + ": " + v + "\r\n";
    }
    req += "\r\n";
    if (!body.empty()) {
      req += body;
    }

    size_t total_sent = 0;
    bool send_ok = true;
    while (total_sent < req.size()) {
#ifdef _WIN32
      int n = ::send(sock, req.data() + total_sent, static_cast<int>(req.size() - total_sent), 0);
#else
      ssize_t n = ::send(sock, req.data() + total_sent, req.size() - total_sent, 0);
#endif
      if (n <= 0) {
        if (n < 0 && errno == EINTR) {
          continue;
        }
        send_ok = false;
        break;
      }
      total_sent += static_cast<size_t>(n);
    }
    if (!send_ok || total_sent < req.size()) {
      closeSocket(sock);
      continue;
    }

    std::string response_str;
    char buf[4096];
    size_t expected_len = 0;
    bool has_content_length = false;

    while (true) {
#ifdef _WIN32
      int n = ::recv(sock, buf, static_cast<int>(sizeof(buf)), 0);
#else
      ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
      if (n <= 0) {
        if (n < 0 && errno == EINTR) {
          continue;
        }
        break;
      }
      response_str.append(buf, static_cast<size_t>(n));

      auto header_end = response_str.find("\r\n\r\n");
      if (header_end != std::string::npos) {
        if (!has_content_length) {
          auto lower_resp = toLower(response_str.substr(0, header_end));
          auto cl_pos = lower_resp.find("content-length:");
          if (cl_pos != std::string::npos) {
            auto val_start = cl_pos + 15;
            auto line_end = lower_resp.find("\r\n", val_start);
            if (line_end != std::string::npos) {
              std::string cl_str = lower_resp.substr(val_start, line_end - val_start);
              while (!cl_str.empty() && (cl_str.front() == ' ' || cl_str.front() == '\t')) {
                cl_str.erase(0, 1);
              }
              try {
                expected_len = std::stoull(cl_str);
                has_content_length = true;
              } catch (...) {}
            }
          }
        }
        if (has_content_length) {
          size_t body_len = response_str.size() - (header_end + 4);
          if (body_len >= expected_len) {
            break;
          }
        }
      }
    }
    closeSocket(sock);

    auto header_end = response_str.find("\r\n\r\n");
    if (header_end == std::string::npos) {
      if (attempt < retries) {
        continue;
      }
      return {};
    }

    size_t body_len = response_str.size() - (header_end + 4);
    if (has_content_length && body_len < expected_len) {
      if (attempt < retries) {
        continue;
      }
      return {};
    }

    HttpResponse resp;
    resp.body = response_str.substr(header_end + 4);
    std::string header_part = response_str.substr(0, header_end);

    auto first_line_end = header_part.find("\r\n");
    std::string status_line =
        (first_line_end != std::string::npos) ? header_part.substr(0, first_line_end) : header_part;
    auto first_space = status_line.find(' ');
    if (first_space != std::string::npos) {
      auto second_space = status_line.find(' ', first_space + 1);
      std::string code_str =
          (second_space != std::string::npos)
              ? status_line.substr(first_space + 1, second_space - first_space - 1)
              : status_line.substr(first_space + 1);
      try {
        resp.status_code = std::stoi(code_str);
      } catch (...) {
        resp.status_code = 0;
      }
    }

    size_t cur = (first_line_end != std::string::npos) ? first_line_end + 2 : std::string::npos;
    while (cur != std::string::npos && cur < header_part.size()) {
      auto next = header_part.find("\r\n", cur);
      std::string line = (next != std::string::npos) ? header_part.substr(cur, next - cur)
                                                     : header_part.substr(cur);
      cur = (next != std::string::npos) ? next + 2 : std::string::npos;
      auto colon = line.find(':');
      if (colon != std::string::npos) {
        std::string k = line.substr(0, colon);
        std::string v = line.substr(colon + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) {
          v.erase(0, 1);
        }
        resp.headers[toLower(k)] = v;
        resp.headers[k] = v;
      }
    }

    // Under high concurrency load, if server was temporarily busy (503) or dropped connection (0),
    // retry!
    if ((resp.status_code == 503 || resp.status_code == 0) && attempt < retries) {
      continue;
    }

    if (resp.status_code > 0) {
      return resp;
    }
  }

  return {};
}

std::string makeInferJson(float v0, float v1) {
  Json::Value root;
  Json::Value inputs(Json::arrayValue);
  Json::Value input0;
  input0["name"] = "x";
  Json::Value shape(Json::arrayValue);
  shape.append(1);
  shape.append(2);
  input0["shape"] = shape;
  input0["datatype"] = "FP32";
  Json::Value data(Json::arrayValue);
  data.append(v0);
  data.append(v1);
  input0["data"] = data;
  inputs.append(input0);
  root["inputs"] = inputs;

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return Json::writeString(writer, root);
}

Json::Value parseJson(const std::string& str) {
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  reader->parse(str.data(), str.data() + str.size(), &root, &errs);
  return root;
}

}  // namespace

// 1. Sync Inference Execution
TEST(ServerExecutionTest, SyncInferenceExecution) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  auto reg_res = server.registerModel("identity", session, "1");
  ASSERT_TRUE(reg_res.isOk());

  auto start_res = server.startAsync();
  ASSERT_TRUE(start_res.isOk());
  ASSERT_TRUE(server.isRunning());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  std::string body = makeInferJson(10.5f, 20.5f);
  auto resp = sendHttpRequest(
      "127.0.0.1", port, "POST", "/v1/models/identity/infer",
      {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "sync"}}, body);

  EXPECT_EQ(resp.status_code, 200);
  auto json = parseJson(resp.body);
  EXPECT_EQ(json["model_name"].asString(), "identity");
  ASSERT_TRUE(json.isMember("outputs"));
  ASSERT_EQ(json["outputs"].size(), 1u);
  EXPECT_EQ(json["outputs"][0]["name"].asString(), "y");
  EXPECT_FLOAT_EQ(json["outputs"][0]["data"][0].asFloat(), 10.5f);
  EXPECT_FLOAT_EQ(json["outputs"][0]["data"][1].asFloat(), 20.5f);

  auto it = resp.headers.find("x-pyreth-execution-mode");
  ASSERT_NE(it, resp.headers.end());
  EXPECT_EQ(it->second, "sync");

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 2. Async Inference Execution
TEST(ServerExecutionTest, AsyncInferenceExecution) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  std::string body = makeInferJson(42.0f, 99.0f);
  auto resp = sendHttpRequest(
      "127.0.0.1", port, "POST", "/v1/models/identity/infer",
      {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "async"}}, body);

  EXPECT_EQ(resp.status_code, 200);
  auto json = parseJson(resp.body);
  EXPECT_EQ(json["model_name"].asString(), "identity");
  ASSERT_TRUE(json.isMember("outputs"));
  ASSERT_EQ(json["outputs"].size(), 1u);
  EXPECT_FLOAT_EQ(json["outputs"][0]["data"][0].asFloat(), 42.0f);
  EXPECT_FLOAT_EQ(json["outputs"][0]["data"][1].asFloat(), 99.0f);

  auto it = resp.headers.find("x-pyreth-execution-mode");
  ASSERT_NE(it, resp.headers.end());
  EXPECT_EQ(it->second, "async");

  auto it_thread = resp.headers.find("x-pyreth-worker-thread-id");
  ASSERT_NE(it_thread, resp.headers.end());
  EXPECT_FALSE(it_thread->second.empty());

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 3. Non-Blocking Reactor Verification
TEST(ServerExecutionTest, NonBlockingReactorVerification) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  // Dispatch an async inference request with 800ms simulated delay in a separate client thread
  std::string body = makeInferJson(1.0f, 2.0f);
  auto slow_future = std::async(std::launch::async, [port, body]() {
    return sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/identity/infer",
                           {{"Content-Type", "application/json"},
                            {"X-Pyreth-Execution-Mode", "async"},
                            {"X-Pyreth-Simulated-Delay-Ms", "800"}},
                           body);
  });

  // Brief pause to ensure the slow inference request has reached the server and is executing
  std::this_thread::sleep_for(std::chrono::milliseconds(40));

  // Measure latency of GET /livez on the main thread while inference is delayed in ThreadPool
  auto live_start = std::chrono::steady_clock::now();
  auto live_resp = sendHttpRequest("127.0.0.1", port, "GET", "/livez");
  auto live_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - live_start)
                          .count();

  EXPECT_EQ(live_resp.status_code, 200);
  // If reactor loop was blocked by inference, livez would take >= 800ms. It must return much
  // faster!
  EXPECT_LT(live_elapsed, 400);

  auto slow_resp = slow_future.get();
  EXPECT_EQ(slow_resp.status_code, 200);

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 4. High Concurrency Stress Test (25 concurrent requests)
TEST(ServerExecutionTest, HighConcurrencyStress) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 4};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  constexpr int kTotalRequests = 25;
  std::vector<std::future<HttpResponse>> futures;
  futures.reserve(kTotalRequests);

  for (int i = 0; i < kTotalRequests; ++i) {
    futures.push_back(std::async(std::launch::async, [port, i]() {
      float v0 = static_cast<float>(i);
      float v1 = static_cast<float>(i * 2);
      std::string body = makeInferJson(v0, v1);
      return sendHttpRequest(
          "127.0.0.1", port, "POST", "/v1/models/identity/infer",
          {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "async"}}, body);
    }));
  }

  for (int i = 0; i < kTotalRequests; ++i) {
    auto resp = futures[static_cast<size_t>(i)].get();
    ASSERT_EQ(resp.status_code, 200);
    auto json = parseJson(resp.body);
    ASSERT_TRUE(json.isMember("outputs"));
    ASSERT_EQ(json["outputs"].size(), 1u);
    EXPECT_FLOAT_EQ(json["outputs"][0]["data"][0].asFloat(), static_cast<float>(i));
    EXPECT_FLOAT_EQ(json["outputs"][0]["data"][1].asFloat(), static_cast<float>(i * 2));
  }

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 5. Request Timeout Enforcement (504 Gateway Timeout)
TEST(ServerExecutionTest, RequestTimeoutEnforcement) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2, .request_timeout_ms = 400};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  // Injects an artificial compute delay of 1200ms exceeding request_timeout_ms (400ms)
  std::string body = makeInferJson(5.0f, 6.0f);
  auto resp = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/identity/infer",
                              {{"Content-Type", "application/json"},
                               {"X-Pyreth-Execution-Mode", "async"},
                               {"X-Pyreth-Simulated-Delay-Ms", "1200"}},
                              body);

  EXPECT_EQ(resp.status_code, 504);
  auto json = parseJson(resp.body);
  EXPECT_TRUE(json.isMember("error"));

  // Subsequent normal request without delay succeeds
  auto normal_resp = sendHttpRequest(
      "127.0.0.1", port, "POST", "/v1/models/identity/infer",
      {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "async"}}, body);
  EXPECT_EQ(normal_resp.status_code, 200);

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 6. Arena Per-Request Isolation
TEST(ServerExecutionTest, ArenaPerRequestIsolation) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 4};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  // Concurrently dispatch requests requiring distinct input values
  constexpr int kNumRequests = 12;
  std::vector<std::future<HttpResponse>> futures;
  futures.reserve(kNumRequests);

  for (int i = 0; i < kNumRequests; ++i) {
    futures.push_back(std::async(std::launch::async, [port, i]() {
      float v0 = static_cast<float>(100 + i * 3);
      float v1 = static_cast<float>(200 + i * 7);
      std::string body = makeInferJson(v0, v1);
      return sendHttpRequest(
          "127.0.0.1", port, "POST", "/v1/models/identity/infer",
          {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "async"}}, body);
    }));
  }

  for (int i = 0; i < kNumRequests; ++i) {
    auto resp = futures[static_cast<size_t>(i)].get();
    ASSERT_EQ(resp.status_code, 200);
    auto json = parseJson(resp.body);
    ASSERT_TRUE(json.isMember("outputs"));
    EXPECT_FLOAT_EQ(json["outputs"][0]["data"][0].asFloat(), static_cast<float>(100 + i * 3));
    EXPECT_FLOAT_EQ(json["outputs"][0]["data"][1].asFloat(), static_cast<float>(200 + i * 7));
  }

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

// 7. Graceful Shutdown With Pending Tasks
TEST(ServerExecutionTest, GracefulShutdownWithPendingTasks) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2};
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());
  uint16_t port = server.port();
  ASSERT_GT(port, 0);

  // Launch delayed requests in background threads
  std::string body = makeInferJson(7.0f, 8.0f);
  std::vector<std::thread> client_threads;
  for (int i = 0; i < 4; ++i) {
    client_threads.emplace_back([port, body]() {
      sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/identity/infer",
                      {{"Content-Type", "application/json"},
                       {"X-Pyreth-Execution-Mode", "async"},
                       {"X-Pyreth-Simulated-Delay-Ms", "500"}},
                      body, 5, 0);
    });
  }

  // Wait briefly so tasks are queued/running in ThreadPool
  std::this_thread::sleep_for(std::chrono::milliseconds(40));

  // Stop server while tasks are pending
  server.stop();
  EXPECT_FALSE(server.isRunning());

  for (auto& t : client_threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}
