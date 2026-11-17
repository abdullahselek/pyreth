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

#include <gtest/gtest.h>
#include <json/json.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <pyreth/inference_session.hpp>
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
inline int closeSocket(socket_t s) {
  return ::closesocket(s);
}
inline bool isValidSocket(socket_t s) {
  return s != INVALID_SOCKET;
}
#else
using socket_t = int;
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

HttpResponse sendHttpRequest(const std::string& host, uint16_t port, const std::string& method,
                             const std::string& path) {
  std::string target_ip = (host == "0.0.0.0" || host.empty()) ? "127.0.0.1" : host;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  ::inet_pton(AF_INET, target_ip.c_str(), &addr.sin_addr);

  socket_t sock;
  bool connected = false;
  for (int attempt = 0; attempt < 50; ++attempt) {
    sock = ::socket(AF_INET, SOCK_STREAM, 0);
    if (!isValidSocket(sock)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
#ifdef _WIN32
    DWORD timeout_ms = 5000;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    struct timeval tv{};
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    ::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ::setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
      connected = true;
      break;
    }
    closeSocket(sock);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (!connected) {
    return {};
  }

  std::string req = method + " " + path + " HTTP/1.1\r\n";
  req += "Host: " + target_ip + ":" + std::to_string(port) + "\r\n";
  req += "Connection: close\r\n\r\n";

  ::send(sock, req.data(), static_cast<int>(req.size()), 0);

  std::string response_str;
  char buf[4096];
  while (true) {
#ifdef _WIN32
    int n = ::recv(sock, buf, static_cast<int>(sizeof(buf)), 0);
#else
    ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
    if (n <= 0) {
      break;
    }
    response_str.append(buf, static_cast<size_t>(n));

    auto header_end = response_str.find("\r\n\r\n");
    if (header_end != std::string::npos) {
      auto cl_pos = response_str.find("content-length:");
      if (cl_pos == std::string::npos) {
        cl_pos = response_str.find("Content-Length:");
      }
      if (cl_pos != std::string::npos && cl_pos < header_end) {
        auto val_start = cl_pos + 15;
        auto line_end = response_str.find("\r\n", val_start);
        if (line_end != std::string::npos) {
          std::string cl_str = response_str.substr(val_start, line_end - val_start);
          while (!cl_str.empty() && (cl_str.front() == ' ' || cl_str.front() == '\t')) {
            cl_str.erase(0, 1);
          }
          try {
            size_t expected_len = std::stoull(cl_str);
            size_t body_len = response_str.size() - (header_end + 4);
            if (body_len >= expected_len) {
              break;
            }
          } catch (...) {}
        }
      }
    }
  }
  closeSocket(sock);

  HttpResponse resp;
  auto header_end = response_str.find("\r\n\r\n");
  if (header_end == std::string::npos) {
    return resp;
  }

  resp.body = response_str.substr(header_end + 4);
  std::string header_part = response_str.substr(0, header_end);

  auto first_line_end = header_part.find("\r\n");
  std::string status_line =
      (first_line_end != std::string::npos) ? header_part.substr(0, first_line_end) : header_part;
  auto first_space = status_line.find(' ');
  if (first_space != std::string::npos) {
    auto second_space = status_line.find(' ', first_space + 1);
    std::string code_str = (second_space != std::string::npos)
                               ? status_line.substr(first_space + 1, second_space - first_space - 1)
                               : status_line.substr(first_space + 1);
    try {
      resp.status_code = std::stoi(code_str);
    } catch (...) {
      resp.status_code = 0;
    }
  }

  return resp;
}

}  // namespace

TEST(ServerModelRegistryTest, RegisterAndLookup) {
  ModelRegistry registry;
  EXPECT_FALSE(registry.hasModel("classifier"));

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);

  auto res = registry.registerModel("classifier", session);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  EXPECT_TRUE(registry.hasModel("classifier"));
  EXPECT_EQ(registry.modelCount(), 1u);

  auto lookup_res = registry.getModel("classifier");
  ASSERT_TRUE(lookup_res.isOk());
  EXPECT_EQ(lookup_res.value(), session);
}

TEST(ServerModelRegistryTest, DuplicateRegistration) {
  ModelRegistry registry;
  auto session1 = createValidSession();
  auto session2 = createValidSession();

  ASSERT_TRUE(registry.registerModel("classifier", session1).isOk());

  auto dup_res = registry.registerModel("classifier", session2);
  EXPECT_FALSE(dup_res.isOk());
  EXPECT_EQ(dup_res.error().code(), ErrorCode::ModelAlreadyExists);
}

TEST(ServerModelRegistryTest, UnregisterModel) {
  ModelRegistry registry;
  auto session = createValidSession();
  ASSERT_TRUE(registry.registerModel("classifier", session).isOk());
  EXPECT_TRUE(registry.hasModel("classifier"));

  auto unreg_res = registry.unregisterModel("classifier");
  EXPECT_TRUE(unreg_res.isOk());
  EXPECT_FALSE(registry.hasModel("classifier"));

  auto lookup_res = registry.getModel("classifier");
  EXPECT_FALSE(lookup_res.isOk());
  EXPECT_EQ(lookup_res.error().code(), ErrorCode::ModelNotFound);

  // Unregister non-existent model
  auto unreg_again = registry.unregisterModel("classifier");
  EXPECT_FALSE(unreg_again.isOk());
  EXPECT_EQ(unreg_again.error().code(), ErrorCode::ModelNotFound);
}

TEST(ServerModelRegistryTest, ConcurrentAccess) {
  ModelRegistry registry;
  constexpr size_t kNumThreads = 16;
  constexpr size_t kIterations = 100;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::atomic<bool> start_signal{false};

  for (size_t t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      while (!start_signal.load()) {
        std::this_thread::yield();
      }

      std::string model_name = "model_" + std::to_string(t);
      for (size_t iter = 0; iter < kIterations; ++iter) {
        auto session = createValidSession();
        // Register
        (void)registry.registerModel(model_name, session);
        // Lookup
        if (registry.hasModel(model_name)) {
          auto res = registry.getModel(model_name);
          (void)res;
        }
        // List
        auto list = registry.listModels();
        (void)list;
        // Unregister
        (void)registry.unregisterModel(model_name);
      }
    });
  }

  start_signal.store(true);
  for (auto& th : threads) {
    th.join();
  }
}

TEST(ServerModelRegistryTest, ListModelsEndpoint) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);

  auto session1 = createValidSession();
  auto session2 = createValidSession();
  ASSERT_TRUE(server.registerModel("model_alpha", session1, "1").isOk());
  ASSERT_TRUE(server.registerModel("model_beta", session2, "2").isOk());

  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest(server.host(), server.port(), "GET", "/v1/models");
  EXPECT_EQ(resp.status_code, 200);

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool ok = reader->parse(resp.body.data(), resp.body.data() + resp.body.size(), &root, &errs);
  ASSERT_TRUE(ok) << errs;
  ASSERT_TRUE(root.isArray());
  EXPECT_EQ(root.size(), 2u);

  server.stop();
}

TEST(ServerModelRegistryTest, ModelMetadataEndpoint) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);

  auto session = createValidSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("classifier", session, "1").isOk());

  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest(server.host(), server.port(), "GET", "/v1/models/classifier");
  EXPECT_EQ(resp.status_code, 200);

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool ok = reader->parse(resp.body.data(), resp.body.data() + resp.body.size(), &root, &errs);
  ASSERT_TRUE(ok) << errs;

  EXPECT_EQ(root["name"].asString(), "classifier");
  ASSERT_TRUE(root["inputs"].isArray());
  ASSERT_EQ(root["inputs"].size(), 1u);
  EXPECT_EQ(root["inputs"][0]["name"].asString(), "x");
  EXPECT_EQ(root["inputs"][0]["datatype"].asString(), "FP32");
  EXPECT_EQ(root["inputs"][0]["shape"][0].asInt64(), 1);
  EXPECT_EQ(root["inputs"][0]["shape"][1].asInt64(), 2);

  ASSERT_TRUE(root["outputs"].isArray());
  ASSERT_EQ(root["outputs"].size(), 1u);
  EXPECT_EQ(root["outputs"][0]["name"].asString(), "y");
  EXPECT_EQ(root["outputs"][0]["datatype"].asString(), "FP32");
  EXPECT_EQ(root["outputs"][0]["shape"][0].asInt64(), 1);
  EXPECT_EQ(root["outputs"][0]["shape"][1].asInt64(), 2);

  server.stop();
}

TEST(ServerModelRegistryTest, ModelNotFoundEndpoint) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);

  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest(server.host(), server.port(), "GET", "/v1/models/non_existent");
  EXPECT_EQ(resp.status_code, 404);

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool ok = reader->parse(resp.body.data(), resp.body.data() + resp.body.size(), &root, &errs);
  ASSERT_TRUE(ok) << errs;

  EXPECT_TRUE(root.isMember("error"));
  EXPECT_NE(root["error"].asString().find("not found"), std::string::npos);

  server.stop();
}
