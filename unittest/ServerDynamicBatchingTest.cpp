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
#include <latch>
#include <memory>
#include <numeric>
#include <pyreth/dynamic_batcher.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/model_registry.hpp>
#include <pyreth/server/server_config.hpp>
#include <span>
#include <string>
#include <string_view>
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

enum class OnnxDataType : uint32_t {
  FLOAT = 1,
  INT64 = 7,
};

struct DimSpec {
  int64_t value{-1};
  std::string param{};

  static DimSpec fixed(int64_t v) { return DimSpec{v, ""}; }
  static DimSpec dynamic(std::string sym = "") { return DimSpec{-1, std::move(sym)}; }
};

struct TensorSpec {
  std::string name;
  OnnxDataType elem_type;
  std::vector<DimSpec> dims;
};

class ProtoBuilder {
public:
  void writeVarint(uint64_t val) {
    while (val >= 0x80) {
      buf_.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
      val >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(val & 0x7F));
  }

  void writeTag(uint32_t field_number, uint8_t wire_type) {
    writeVarint((static_cast<uint64_t>(field_number) << 3) | wire_type);
  }

  void writeVarintField(uint32_t field_number, uint64_t val) {
    writeTag(field_number, 0);
    writeVarint(val);
  }

  void writeBytesField(uint32_t field_number, std::span<const uint8_t> bytes) {
    writeTag(field_number, 2);
    writeVarint(bytes.size());
    buf_.insert(buf_.end(), bytes.begin(), bytes.end());
  }

  void writeStringField(uint32_t field_number, std::string_view str) {
    writeTag(field_number, 2);
    writeVarint(str.size());
    buf_.insert(buf_.end(), reinterpret_cast<const uint8_t*>(str.data()),
                reinterpret_cast<const uint8_t*>(str.data()) + str.size());
  }

  void writeSubmessage(uint32_t field_number, const ProtoBuilder& sub) {
    writeBytesField(field_number, sub.bytes());
  }

  [[nodiscard]] const std::vector<uint8_t>& bytes() const noexcept { return buf_; }

private:
  std::vector<uint8_t> buf_;
};

ProtoBuilder encodeValueInfo(const TensorSpec& spec) {
  ProtoBuilder shape_proto;
  for (const auto& d : spec.dims) {
    ProtoBuilder dim_proto;
    if (!d.param.empty()) {
      dim_proto.writeStringField(2, d.param);
    } else if (d.value >= 0) {
      dim_proto.writeVarintField(1, static_cast<uint64_t>(d.value));
    }
    shape_proto.writeSubmessage(1, dim_proto);
  }

  ProtoBuilder tensor_type_proto;
  tensor_type_proto.writeVarintField(1, static_cast<uint64_t>(spec.elem_type));
  tensor_type_proto.writeSubmessage(2, shape_proto);

  ProtoBuilder type_proto;
  type_proto.writeSubmessage(1, tensor_type_proto);

  ProtoBuilder value_info;
  value_info.writeStringField(1, spec.name);
  value_info.writeSubmessage(2, type_proto);
  return value_info;
}

ProtoBuilder encodeNode(std::span<const std::string_view> inputs,
                        std::span<const std::string_view> outputs, std::string_view name,
                        std::string_view op_type) {
  ProtoBuilder node;
  for (auto in : inputs) {
    node.writeStringField(1, in);
  }
  for (auto out : outputs) {
    node.writeStringField(2, out);
  }
  node.writeStringField(3, name);
  node.writeStringField(4, op_type);
  return node;
}

std::vector<uint8_t> buildDynamicIdentityModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch"), DimSpec::fixed(4)}}};

  std::string_view id_in[] = {"x"};
  std::string_view id_out[] = {"y"};

  std::vector<ProtoBuilder> nodes = {
      encodeNode(id_in, id_out, "identity_node", "Identity"),
  };

  ProtoBuilder graph;
  for (const auto& n : nodes) {
    graph.writeSubmessage(1, n);
  }
  graph.writeStringField(2, "identity_graph");
  for (const auto& in : inputs) {
    graph.writeSubmessage(11, encodeValueInfo(in));
  }
  for (const auto& out : outputs) {
    graph.writeSubmessage(12, encodeValueInfo(out));
  }

  ProtoBuilder opset;
  opset.writeStringField(1, "");
  opset.writeVarintField(2, 17);

  ProtoBuilder model;
  model.writeVarintField(1, 9);
  model.writeStringField(2, "pyreth_dynamic_batching_server_test");
  model.writeSubmessage(7, graph);
  model.writeSubmessage(8, opset);
  return model.bytes();
}

std::shared_ptr<InferenceSession> createDynamicSession() {
  auto model_bytes = buildDynamicIdentityModel();
  auto res =
      InferenceSession::create(std::span<const uint8_t>(model_bytes.data(), model_bytes.size()));
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
    auto space1 = status_line.find(' ');
    if (space1 != std::string::npos) {
      auto space2 = status_line.find(' ', space1 + 1);
      std::string code_str = (space2 != std::string::npos)
                                 ? status_line.substr(space1 + 1, space2 - space1 - 1)
                                 : status_line.substr(space1 + 1);
      try {
        resp.status_code = std::stoi(code_str);
      } catch (...) {
        resp.status_code = 0;
      }
    }

    size_t line_start =
        (first_line_end != std::string::npos) ? first_line_end + 2 : std::string::npos;
    while (line_start != std::string::npos && line_start < header_part.size()) {
      auto line_end = header_part.find("\r\n", line_start);
      std::string line = (line_end != std::string::npos)
                             ? header_part.substr(line_start, line_end - line_start)
                             : header_part.substr(line_start);
      auto colon = line.find(':');
      if (colon != std::string::npos) {
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
          val.erase(0, 1);
        }
        resp.headers[toLower(key)] = val;
      }
      if (line_end == std::string::npos) {
        break;
      }
      line_start = line_end + 2;
    }

    return resp;
  }
  return {};
}

std::string buildKServeJsonRequest(const std::vector<float>& data,
                                   const std::vector<int64_t>& shape = {1, 4},
                                   const std::string& input_name = "x") {
  Json::Value root;
  Json::Value inputs(Json::arrayValue);
  Json::Value input;
  input["name"] = input_name;
  input["datatype"] = "FP32";
  Json::Value shape_arr(Json::arrayValue);
  for (int64_t d : shape) {
    shape_arr.append(static_cast<Json::Value::Int64>(d));
  }
  input["shape"] = shape_arr;

  Json::Value data_arr(Json::arrayValue);
  for (float v : data) {
    data_arr.append(v);
  }
  input["data"] = data_arr;
  inputs.append(input);
  root["inputs"] = inputs;

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return Json::writeString(writer, root);
}

bool parseKServeJsonResponse(const std::string& json_str, std::vector<float>& out_data,
                             std::vector<int64_t>& out_shape) {
  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::istringstream ss(json_str);
  if (!Json::parseFromStream(builder, ss, &root, &errs)) {
    return false;
  }
  if (!root.isMember("outputs") || !root["outputs"].isArray() || root["outputs"].empty()) {
    return false;
  }
  const auto& out = root["outputs"][0];
  out_shape.clear();
  if (out.isMember("shape") && out["shape"].isArray()) {
    for (const auto& d : out["shape"]) {
      out_shape.push_back(d.asInt64());
    }
  }
  out_data.clear();
  if (out.isMember("data") && out["data"].isArray()) {
    for (const auto& v : out["data"]) {
      out_data.push_back(v.asFloat());
    }
  }
  return true;
}

}  // namespace

class ServerDynamicBatchingTest : public ::testing::Test {
protected:
  void SetUp() override {
    session_ = createDynamicSession();
    ASSERT_NE(session_, nullptr);
  }

  void TearDown() override {
    if (server_ && server_->isRunning()) {
      server_->stop();
    }
  }

  std::shared_ptr<InferenceSession> session_;
  std::unique_ptr<HttpServer> server_;
};

TEST_F(ServerDynamicBatchingTest, ConcurrentBatchAggregation) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 4, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 16;
  batch_cfg.max_batch_delay_ms = 35;
  batch_cfg.max_queue_capacity = 1024;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("batched_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("batched_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());
  ASSERT_TRUE(server_->isRunning());

  const uint16_t port = server_->port();
  constexpr size_t kNumClients = 32;
  std::vector<HttpResponse> responses(kNumClients);
  std::vector<std::thread> client_threads;
  client_threads.reserve(kNumClients);

  std::latch start_latch(kNumClients);

  for (size_t i = 0; i < kNumClients; ++i) {
    client_threads.emplace_back([&, i]() {
      std::vector<float> input_data = {
          static_cast<float>(i),
          static_cast<float>(i) + 1.0f,
          static_cast<float>(i) + 2.0f,
          static_cast<float>(i) + 3.0f,
      };
      std::string body = buildKServeJsonRequest(input_data, {1, 4});

      start_latch.count_down();
      start_latch.wait();

      std::unordered_map<std::string, std::string> headers = {
          {"Content-Type", "application/json"},
          {"X-Pyreth-Execution-Mode", "batch"},
      };
      responses[i] = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/batched_model/infer",
                                     headers, body);
    });
  }

  for (auto& th : client_threads) {
    th.join();
  }

  for (size_t i = 0; i < kNumClients; ++i) {
    ASSERT_EQ(responses[i].status_code, 200) << "Client " << i << " failed: " << responses[i].body;
    auto mode_it = responses[i].headers.find("x-pyreth-execution-mode");
    ASSERT_NE(mode_it, responses[i].headers.end());
    EXPECT_EQ(mode_it->second, "batch");

    std::vector<float> out_data;
    std::vector<int64_t> out_shape;
    ASSERT_TRUE(parseKServeJsonResponse(responses[i].body, out_data, out_shape));
    ASSERT_EQ(out_shape, (std::vector<int64_t>{1, 4}));
    ASSERT_EQ(out_data.size(), 4u);
    EXPECT_FLOAT_EQ(out_data[0], static_cast<float>(i));
    EXPECT_FLOAT_EQ(out_data[1], static_cast<float>(i) + 1.0f);
    EXPECT_FLOAT_EQ(out_data[2], static_cast<float>(i) + 2.0f);
    EXPECT_FLOAT_EQ(out_data[3], static_cast<float>(i) + 3.0f);
  }

  auto metrics = batcher->metrics();
  EXPECT_EQ(metrics.requests_submitted, kNumClients);
  EXPECT_EQ(metrics.requests_completed, kNumClients);
  EXPECT_EQ(metrics.requests_rejected, 0u);
  EXPECT_LT(metrics.batches_dispatched, kNumClients);
  EXPECT_GT(metrics.averageBatchSize(), 1.0);
}

TEST_F(ServerDynamicBatchingTest, DelayTimeoutExpiration) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 16;
  batch_cfg.max_batch_delay_ms = 25;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("timeout_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("timeout_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());

  const uint16_t port = server_->port();
  std::vector<float> input_data = {10.0f, 20.0f, 30.0f, 40.0f};
  std::string body = buildKServeJsonRequest(input_data, {1, 4});

  auto start_time = std::chrono::steady_clock::now();
  auto resp = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/timeout_model/infer",
                              {{"Content-Type", "application/json"}}, body);
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time)
                        .count();

  EXPECT_EQ(resp.status_code, 200);
  EXPECT_GE(elapsed_ms, 20);

  std::vector<float> out_data;
  std::vector<int64_t> out_shape;
  ASSERT_TRUE(parseKServeJsonResponse(resp.body, out_data, out_shape));
  ASSERT_EQ(out_shape, (std::vector<int64_t>{1, 4}));
  EXPECT_FLOAT_EQ(out_data[0], 10.0f);
  EXPECT_FLOAT_EQ(out_data[3], 40.0f);

  auto metrics = batcher->metrics();
  EXPECT_EQ(metrics.requests_submitted, 1u);
  EXPECT_EQ(metrics.requests_completed, 1u);
  EXPECT_EQ(metrics.batches_dispatched, 1u);
  EXPECT_GE(metrics.timeout_batches, 1u);
}

TEST_F(ServerDynamicBatchingTest, DemuxSlicingAccuracy) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 4, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 16;
  batch_cfg.max_batch_delay_ms = 30;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("slice_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("slice_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());

  const uint16_t port = server_->port();
  constexpr size_t kNumClients = 8;
  std::vector<HttpResponse> responses(kNumClients);
  std::vector<std::thread> client_threads;
  client_threads.reserve(kNumClients);

  std::latch sync_latch(kNumClients);

  for (size_t i = 0; i < kNumClients; ++i) {
    client_threads.emplace_back([&, i]() {
      float base = static_cast<float>((i + 1) * 100);
      std::vector<float> input_data = {base, base + 1.0f, base + 2.0f, base + 3.0f};
      std::string body = buildKServeJsonRequest(input_data, {1, 4});

      sync_latch.count_down();
      sync_latch.wait();

      responses[i] = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/slice_model/infer",
                                     {{"Content-Type", "application/json"}}, body);
    });
  }

  for (auto& th : client_threads) {
    th.join();
  }

  for (size_t i = 0; i < kNumClients; ++i) {
    ASSERT_EQ(responses[i].status_code, 200);
    std::vector<float> out_data;
    std::vector<int64_t> out_shape;
    ASSERT_TRUE(parseKServeJsonResponse(responses[i].body, out_data, out_shape));
    ASSERT_EQ(out_shape, (std::vector<int64_t>{1, 4}));

    float base = static_cast<float>((i + 1) * 100);
    EXPECT_FLOAT_EQ(out_data[0], base);
    EXPECT_FLOAT_EQ(out_data[1], base + 1.0f);
    EXPECT_FLOAT_EQ(out_data[2], base + 2.0f);
    EXPECT_FLOAT_EQ(out_data[3], base + 3.0f);
  }
}

TEST_F(ServerDynamicBatchingTest, QueueCapacityOverload) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 4, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 64;
  batch_cfg.max_batch_delay_ms = 300;
  batch_cfg.max_queue_capacity = 2;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("overload_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("overload_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());

  const uint16_t port = server_->port();
  constexpr size_t kNumRequests = 20;
  std::vector<HttpResponse> responses(kNumRequests);
  std::vector<std::thread> threads;
  threads.reserve(kNumRequests);

  std::latch flood_latch(kNumRequests);

  for (size_t i = 0; i < kNumRequests; ++i) {
    threads.emplace_back([&, i]() {
      std::string body = buildKServeJsonRequest({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
      flood_latch.count_down();
      flood_latch.wait();

      responses[i] = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/overload_model/infer",
                                     {{"Content-Type", "application/json"}}, body);
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  size_t count_200 = 0;
  size_t count_429 = 0;

  for (size_t i = 0; i < kNumRequests; ++i) {
    if (responses[i].status_code == 200) {
      ++count_200;
    } else if (responses[i].status_code == 429) {
      ++count_429;
      auto retry_it = responses[i].headers.find("retry-after");
      EXPECT_NE(retry_it, responses[i].headers.end());
      EXPECT_FALSE(retry_it->second.empty());
      EXPECT_NE(responses[i].body.find("queue is full"), std::string::npos);
    }
  }

  EXPECT_GT(count_429, 0u);
  EXPECT_GT(count_200, 0u);
  EXPECT_EQ(count_200 + count_429, kNumRequests);

  auto metrics = batcher->metrics();
  EXPECT_EQ(metrics.requests_rejected, count_429);
}

TEST_F(ServerDynamicBatchingTest, MetricsTelemetryVerification) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 4;
  batch_cfg.max_batch_delay_ms = 15;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("telemetry_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("telemetry_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());

  const uint16_t port = server_->port();

  // Test metrics on nonexistent model -> 404
  auto not_found_resp = sendHttpRequest("127.0.0.1", port, "GET", "/v1/models/nonexistent/metrics");
  EXPECT_EQ(not_found_resp.status_code, 404);

  // Send 4 requests to process a batch
  for (int i = 0; i < 4; ++i) {
    std::string body = buildKServeJsonRequest({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    auto infer_resp = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/telemetry_model/infer",
                                      {{"Content-Type", "application/json"}}, body);
    ASSERT_EQ(infer_resp.status_code, 200);
  }

  auto metrics_resp =
      sendHttpRequest("127.0.0.1", port, "GET", "/v1/models/telemetry_model/metrics");
  ASSERT_EQ(metrics_resp.status_code, 200);
  EXPECT_NE(metrics_resp.headers["content-type"].find("application/json"), std::string::npos);

  Json::CharReaderBuilder reader_builder;
  Json::Value metrics_json;
  std::string errs;
  std::istringstream ss(metrics_resp.body);
  ASSERT_TRUE(Json::parseFromStream(reader_builder, ss, &metrics_json, &errs));

  EXPECT_EQ(metrics_json["model_name"].asString(), "telemetry_model");
  EXPECT_EQ(metrics_json["requests_submitted"].asUInt64(), 4u);
  EXPECT_EQ(metrics_json["requests_completed"].asUInt64(), 4u);
  EXPECT_EQ(metrics_json["requests_rejected"].asUInt64(), 0u);
  EXPECT_GE(metrics_json["batches_dispatched"].asUInt64(), 1u);
  EXPECT_GE(metrics_json["avg_batch_size"].asDouble(), 1.0);
  EXPECT_GE(metrics_json["avg_queue_time_ms"].asDouble(), 0.0);
  EXPECT_GE(metrics_json["avg_exec_time_ms"].asDouble(), 0.0);
}

TEST_F(ServerDynamicBatchingTest, DynamicBatcherLifecycle) {
  ServerConfig config{.host = "127.0.0.1", .port = 0, .io_threads = 2, .request_timeout_ms = 10000};
  server_ = std::make_unique<HttpServer>(config);

  DynamicBatcherConfig batch_cfg;
  batch_cfg.max_batch_size = 8;
  batch_cfg.max_batch_delay_ms = 20;

  auto batcher_res = DynamicBatcher::create(*session_, batch_cfg);
  ASSERT_TRUE(batcher_res.isOk());
  auto batcher = std::make_shared<DynamicBatcher>(std::move(batcher_res.value()));

  ASSERT_TRUE(server_->registerModel("lifecycle_model", session_).isOk());
  ASSERT_TRUE(server_->registerBatcher("lifecycle_model", batcher).isOk());
  ASSERT_TRUE(server_->startAsync().isOk());

  const uint16_t port = server_->port();

  // Send request using batcher
  std::string body = buildKServeJsonRequest({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
  auto resp = sendHttpRequest("127.0.0.1", port, "POST", "/v1/models/lifecycle_model/infer",
                              {{"Content-Type", "application/json"}}, body);
  EXPECT_EQ(resp.status_code, 200);

  // Unregister batcher
  ASSERT_TRUE(server_->unregisterBatcher("lifecycle_model").isOk());

  // Metrics endpoint should now return 404 since batcher is gone
  auto metrics_resp =
      sendHttpRequest("127.0.0.1", port, "GET", "/v1/models/lifecycle_model/metrics");
  EXPECT_EQ(metrics_resp.status_code, 404);

  // Requesting dynamic batching explicitly when batcher is unregistered should return 400
  auto unreg_batch_resp = sendHttpRequest(
      "127.0.0.1", port, "POST", "/v1/models/lifecycle_model/infer",
      {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "batch"}}, body);
  EXPECT_EQ(unreg_batch_resp.status_code, 400);

  // Unbatched async/sync requests still work on the model's session
  auto async_resp = sendHttpRequest(
      "127.0.0.1", port, "POST", "/v1/models/lifecycle_model/infer",
      {{"Content-Type", "application/json"}, {"X-Pyreth-Execution-Mode", "async"}}, body);
  EXPECT_EQ(async_resp.status_code, 200);
}
