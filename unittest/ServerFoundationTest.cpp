#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <pyreth/common.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/server_config.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

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
inline int closeSocket(socket_t s) {
  return ::close(s);
}
inline bool isValidSocket(socket_t s) {
  return s >= 0;
}
#endif

struct HttpResponse {
  int status_code{0};
  std::unordered_map<std::string, std::string> headers;
  std::string body;
};

HttpResponse sendHttpRequest(
    const std::string& host, uint16_t port, const std::string& method, const std::string& path,
    const std::string& body = "",
    const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) {
  socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (!isValidSocket(sock)) {
    return {};
  }

  // Set timeout of 5 seconds
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

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  std::string target_ip = (host == "0.0.0.0" || host.empty()) ? "127.0.0.1" : host;
  ::inet_pton(AF_INET, target_ip.c_str(), &addr.sin_addr);

  if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    closeSocket(sock);
    return {};
  }

  std::string req = method + " " + path + " HTTP/1.1\r\n";
  req += "Host: " + target_ip + ":" + std::to_string(port) + "\r\n";
  req += "Connection: close\r\n";
  for (const auto& [k, v] : extra_headers) {
    req += k + ": " + v + "\r\n";
  }
  if (!body.empty() || method == "POST" || method == "PUT") {
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  }
  req += "\r\n";
  if (!body.empty()) {
    req += body;
  }

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

    // Check if headers have been fully received
    auto header_end = response_str.find("\r\n\r\n");
    if (header_end != std::string::npos) {
      // Check for 204 No Content
      if (response_str.find("204 No Content") != std::string::npos ||
          response_str.find("304 Not Modified") != std::string::npos) {
        break;
      }
      // Check for Content-Length
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

  // Parse status line
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

  // Parse headers
  size_t pos = (first_line_end != std::string::npos) ? first_line_end + 2 : std::string::npos;
  while (pos != std::string::npos && pos < header_part.size()) {
    auto next_pos = header_part.find("\r\n", pos);
    std::string line = (next_pos != std::string::npos) ? header_part.substr(pos, next_pos - pos)
                                                       : header_part.substr(pos);
    auto colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = line.substr(0, colon);
      std::string val = line.substr(colon + 1);
      while (!val.empty() && (val.front() == ' ' || val.front() == '\t'))
        val.erase(0, 1);
      while (!val.empty() && (val.back() == ' ' || val.back() == '\t'))
        val.pop_back();
      std::string lower_key = key;
      for (char& c : lower_key)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      resp.headers[lower_key] = val;
    }
    if (next_pos == std::string::npos)
      break;
    pos = next_pos + 2;
  }

  return resp;
}

}  // namespace

TEST(ServerFoundationTest, ServerConfig_DefaultValues) {
  ServerConfig config;
  EXPECT_EQ(config.host, "0.0.0.0");
  EXPECT_EQ(config.port, 8080);
  size_t expected_io_threads = std::max<size_t>(1, std::thread::hardware_concurrency() / 2);
  EXPECT_EQ(config.io_threads, expected_io_threads);
  EXPECT_EQ(config.request_timeout_ms, 30000);
  EXPECT_EQ(config.max_payload_size, 64 * 1024 * 1024);
  EXPECT_FALSE(config.enable_cors);
  EXPECT_TRUE(config.isValid());
  EXPECT_TRUE(config.validate().isOk());
}

TEST(ServerFoundationTest, ServerConfig_InvalidConfigurations) {
  // port > 65535
  {
    ServerConfig config;
    config.setPort(70000);
    EXPECT_FALSE(config.isValid());
    EXPECT_FALSE(config.validate().isOk());
    EXPECT_EQ(config.validate().error().code(), pyreth::StatusCode::INVALID_ARGUMENT);
  }
  // negative timeout
  {
    ServerConfig config;
    config.setRequestTimeoutMs(-1);
    EXPECT_FALSE(config.isValid());
    EXPECT_FALSE(config.validate().isOk());
    EXPECT_EQ(config.validate().error().code(), pyreth::StatusCode::INVALID_ARGUMENT);
  }
  // zero io_threads
  {
    ServerConfig config;
    config.setIoThreads(0);
    EXPECT_FALSE(config.isValid());
    EXPECT_FALSE(config.validate().isOk());
    EXPECT_EQ(config.validate().error().code(), pyreth::StatusCode::INVALID_ARGUMENT);
  }
  // zero max_payload_size
  {
    ServerConfig config;
    config.setMaxPayloadSize(0);
    EXPECT_FALSE(config.isValid());
    EXPECT_FALSE(config.validate().isOk());
    EXPECT_EQ(config.validate().error().code(), pyreth::StatusCode::INVALID_ARGUMENT);
  }
}

TEST(ServerFoundationTest, ServerLifecycle_StartStop) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);
  EXPECT_FALSE(server.isRunning());

  auto res = server.startAsync();
  ASSERT_TRUE(res.isOk()) << res.error().message();
  EXPECT_TRUE(server.isRunning());
  EXPECT_GT(server.port(), 0);

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

TEST(ServerFoundationTest, ServerLifecycle_DoubleStart) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);

  auto res1 = server.startAsync();
  ASSERT_TRUE(res1.isOk()) << res1.error().message();
  EXPECT_TRUE(server.isRunning());

  auto res2 = server.startAsync();
  EXPECT_FALSE(res2.isOk());
  EXPECT_TRUE(server.isRunning());

  server.stop();
  EXPECT_FALSE(server.isRunning());
}

TEST(ServerFoundationTest, ServerLifecycle_StopWithoutStart) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);
  EXPECT_FALSE(server.isRunning());

  EXPECT_NO_THROW(server.stop());
  EXPECT_FALSE(server.isRunning());
}

TEST(ServerFoundationTest, ServerLifecycle_RAIIDestruction) {
  uint16_t assigned_port = 0;
  {
    ServerConfig config;
    config.setPort(0);
    HttpServer server(config);
    auto res = server.startAsync();
    ASSERT_TRUE(res.isOk());
    EXPECT_TRUE(server.isRunning());
    assigned_port = server.port();
    EXPECT_GT(assigned_port, 0);
  }

  // Give OS a moment to finish teardown
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // After destruction, the port should be unbound; connect should fail
  socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_TRUE(isValidSocket(sock));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(assigned_port);
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  int conn_res = ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  closeSocket(sock);
  EXPECT_NE(conn_res, 0);
}

TEST(ServerFoundationTest, ServerHealthz_LivenessEndpoint) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);
  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest(server.host(), server.port(), "GET", "/livez");
  EXPECT_EQ(resp.status_code, 200);
  EXPECT_NE(resp.body.find("\"status\":\"alive\""), std::string::npos);

  server.stop();
}

TEST(ServerFoundationTest, ServerHealthz_HealthzEndpoint) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);
  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest(server.host(), server.port(), "GET", "/healthz");
  EXPECT_EQ(resp.status_code, 200);
  EXPECT_NE(resp.headers.find("content-type"), resp.headers.end());
  EXPECT_NE(resp.headers["content-type"].find("application/json"), std::string::npos);
  EXPECT_NE(resp.body.find("\"status\":\"ok\""), std::string::npos);
  EXPECT_NE(resp.body.find(pyreth::PYRETH_VERSION), std::string::npos);

  server.stop();
}

TEST(ServerFoundationTest, ServerHealthz_MethodNotAllowed) {
  ServerConfig config;
  config.setPort(0);
  HttpServer server(config);
  ASSERT_TRUE(server.startAsync().isOk());

  for (const std::string& method : {"POST", "PUT", "DELETE"}) {
    auto resp_healthz = sendHttpRequest(server.host(), server.port(), method, "/healthz");
    EXPECT_EQ(resp_healthz.status_code, 405)
        << "Method " << method << " on /healthz should return 405";

    auto resp_livez = sendHttpRequest(server.host(), server.port(), method, "/livez");
    EXPECT_EQ(resp_livez.status_code, 405) << "Method " << method << " on /livez should return 405";
  }

  server.stop();
}

TEST(ServerFoundationTest, ServerCors_CorsHeaders) {
  ServerConfig config;
  config.setPort(0);
  config.setEnableCors(true);
  HttpServer server(config);
  ASSERT_TRUE(server.startAsync().isOk());

  // Preflight OPTIONS
  auto resp_options =
      sendHttpRequest(server.host(), server.port(), "OPTIONS", "/healthz", "",
                      {{"Origin", "http://example.com"}, {"Access-Control-Request-Method", "GET"}});
  EXPECT_NE(resp_options.headers.find("access-control-allow-origin"), resp_options.headers.end());
  EXPECT_EQ(resp_options.headers["access-control-allow-origin"], "*");
  EXPECT_NE(resp_options.headers.find("access-control-allow-methods"), resp_options.headers.end());
  EXPECT_NE(resp_options.headers.find("access-control-allow-headers"), resp_options.headers.end());

  // Standard GET
  auto resp_get = sendHttpRequest(server.host(), server.port(), "GET", "/healthz");
  EXPECT_EQ(resp_get.status_code, 200);
  EXPECT_NE(resp_get.headers.find("access-control-allow-origin"), resp_get.headers.end());
  EXPECT_EQ(resp_get.headers["access-control-allow-origin"], "*");

  server.stop();
}
