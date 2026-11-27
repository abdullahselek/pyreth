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
#include <cstring>
#include <future>
#include <latch>
#include <memory>
#include <numeric>
#include <pyreth/inference_session.hpp>
#include <pyreth/server/binary_protocol.hpp>
#include <pyreth/server/endpoint.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/server/http_server.hpp>
#include <pyreth/server/model_registry.hpp>
#include <pyreth/server/server_config.hpp>
#include <pyreth/server/streaming.hpp>
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

std::vector<uint8_t> buildDynamicIdentityModel() {
  std::vector<TensorSpec> inputs = {
      {"input", OnnxDataType::FLOAT, {DimSpec::dynamic("batch"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"output", OnnxDataType::FLOAT, {DimSpec::dynamic("batch"), DimSpec::fixed(4)}}};

  ProtoBuilder node;
  node.writeStringField(1, "input");
  node.writeStringField(2, "output");
  node.writeStringField(3, "Identity");
  node.writeStringField(4, "Identity");

  std::vector<ProtoBuilder> nodes;
  nodes.push_back(node);

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
  model.writeStringField(2, "pyreth_streaming_server_test");
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
    const std::string& body = "", int timeout_sec = 10, int retries = 5) {
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
      } catch (...) {}
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
        std::string k = toLower(line.substr(0, colon));
        std::string v = line.substr(colon + 1);
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t'))
          v.erase(0, 1);
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t'))
          v.pop_back();
        resp.headers[k] = v;
      }
      line_start = (line_end != std::string::npos) ? line_end + 2 : std::string::npos;
    }

    return resp;
  }
  return {};
}

// WebSocket client frame helper (RFC 6455 masked client-to-server frames)
std::vector<uint8_t> makeWsClientFrame(std::span<const uint8_t> payload, uint8_t opcode = 0x01) {
  std::vector<uint8_t> frame;
  frame.push_back(0x80 | (opcode & 0x0F));  // FIN + Opcode (0x01 text, 0x02 binary)

  uint8_t mask_bit = 0x80;
  size_t len = payload.size();
  if (len <= 125) {
    frame.push_back(mask_bit | static_cast<uint8_t>(len));
  } else if (len <= 65535) {
    frame.push_back(mask_bit | 126);
    frame.push_back(static_cast<uint8_t>((len >> 8) & 0xFF));
    frame.push_back(static_cast<uint8_t>(len & 0xFF));
  } else {
    frame.push_back(mask_bit | 127);
    for (int i = 7; i >= 0; --i) {
      frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xFF));
    }
  }

  // Masking key (fixed 4 bytes for testing)
  uint8_t masking_key[4] = {0x12, 0x34, 0x56, 0x78};
  frame.insert(frame.end(), masking_key, masking_key + 4);

  // Masked payload
  for (size_t i = 0; i < len; ++i) {
    frame.push_back(payload[i] ^ masking_key[i % 4]);
  }

  return frame;
}

struct WsFrame {
  uint8_t opcode{0};
  std::vector<uint8_t> payload;
};

WsFrame readWsServerFrame(socket_t sock) {
  uint8_t header[2];
#ifdef _WIN32
  int n = ::recv(sock, reinterpret_cast<char*>(header), 2, MSG_WAITALL);
#else
  ssize_t n = ::recv(sock, header, 2, MSG_WAITALL);
#endif
  if (n < 2)
    return {};

  uint8_t opcode = header[0] & 0x0F;
  bool masked = (header[1] & 0x80) != 0;
  uint64_t payload_len = header[1] & 0x7F;

  if (payload_len == 126) {
    uint8_t ext[2];
#ifdef _WIN32
    ::recv(sock, reinterpret_cast<char*>(ext), 2, MSG_WAITALL);
#else
    ::recv(sock, ext, 2, MSG_WAITALL);
#endif
    payload_len = (static_cast<uint64_t>(ext[0]) << 8) | ext[1];
  } else if (payload_len == 127) {
    uint8_t ext[8];
#ifdef _WIN32
    ::recv(sock, reinterpret_cast<char*>(ext), 8, MSG_WAITALL);
#else
    ::recv(sock, ext, 8, MSG_WAITALL);
#endif
    payload_len = 0;
    for (int i = 0; i < 8; ++i) {
      payload_len = (payload_len << 8) | ext[i];
    }
  }

  uint8_t mask_key[4] = {0};
  if (masked) {
#ifdef _WIN32
    ::recv(sock, reinterpret_cast<char*>(mask_key), 4, MSG_WAITALL);
#else
    ::recv(sock, mask_key, 4, MSG_WAITALL);
#endif
  }

  std::vector<uint8_t> payload(payload_len);
  size_t total = 0;
  while (total < payload_len) {
#ifdef _WIN32
    int r = ::recv(sock, reinterpret_cast<char*>(payload.data() + total),
                   static_cast<int>(payload_len - total), 0);
#else
    ssize_t r = ::recv(sock, payload.data() + total, payload_len - total, 0);
#endif
    if (r <= 0)
      break;
    total += static_cast<size_t>(r);
  }

  if (masked) {
    for (size_t i = 0; i < payload_len; ++i) {
      payload[i] ^= mask_key[i % 4];
    }
  }

  return WsFrame{opcode, std::move(payload)};
}

}  // namespace

// =========================================================================
// 1. Binary Protocol Tests
// =========================================================================

TEST(BinaryProtocolTest, ValidHeaderAndPayload) {
  // Crafts a 64-byte binary header with magic 0x50595254, dtype=Float32, ndim=2, dims=[1, 4]
  BinaryTensorHeader header;
  header.magic = kBinaryTensorMagic;
  header.dtype = static_cast<uint8_t>(DataType::Float32);
  header.ndim = 2;
  header.dims[0] = 1;
  header.dims[1] = 4;

  std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
  std::vector<uint8_t> buffer(sizeof(header) + data.size() * sizeof(float));
  std::memcpy(buffer.data(), &header, sizeof(header));
  std::memcpy(buffer.data() + sizeof(header), data.data(), data.size() * sizeof(float));

  auto tensor_res = deserializeBinaryTensor(buffer);
  ASSERT_TRUE(tensor_res.isOk()) << tensor_res.error().message();

  const auto& tensor = tensor_res.value();
  EXPECT_EQ(tensor.ndim(), 2u);
  EXPECT_EQ(tensor.shape()[0], 1);
  EXPECT_EQ(tensor.shape()[1], 4);
  EXPECT_EQ(tensor.dtype(), DataType::Float32);

  // Assert zero-copy pointer equivalence
  EXPECT_EQ(tensor.data(), buffer.data() + kBinaryTensorHeaderSize);

  const float* ptr = static_cast<const float*>(tensor.data());
  EXPECT_FLOAT_EQ(ptr[0], 1.5f);
  EXPECT_FLOAT_EQ(ptr[1], 2.5f);
  EXPECT_FLOAT_EQ(ptr[2], 3.5f);
  EXPECT_FLOAT_EQ(ptr[3], 4.5f);
}

TEST(BinaryProtocolTest, CorruptedHeaderAndMagicValidation) {
  // Truncated header (< 64 bytes)
  std::vector<uint8_t> short_buf(32, 0);
  auto res1 = deserializeBinaryTensor(short_buf);
  EXPECT_FALSE(res1.isOk());

  // Bad magic
  BinaryTensorHeader bad_magic_header;
  bad_magic_header.magic = 0x12345678;
  bad_magic_header.dtype = 0;
  bad_magic_header.ndim = 1;
  bad_magic_header.dims[0] = 4;

  std::vector<uint8_t> bad_magic_buf(64 + 16, 0);
  std::memcpy(bad_magic_buf.data(), &bad_magic_header, sizeof(bad_magic_header));
  auto res2 = deserializeBinaryTensor(bad_magic_buf);
  EXPECT_FALSE(res2.isOk());

  // Unsupported dtype
  BinaryTensorHeader bad_dtype_header;
  bad_dtype_header.magic = kBinaryTensorMagic;
  bad_dtype_header.dtype = 99;
  bad_dtype_header.ndim = 1;
  bad_dtype_header.dims[0] = 4;

  std::vector<uint8_t> bad_dtype_buf(64 + 16, 0);
  std::memcpy(bad_dtype_buf.data(), &bad_dtype_header, sizeof(bad_dtype_header));
  auto res3 = deserializeBinaryTensor(bad_dtype_buf);
  EXPECT_FALSE(res3.isOk());

  // Size mismatch (header claims 4 floats = 16 bytes, but buffer only has 8 bytes)
  BinaryTensorHeader size_mismatch_header;
  size_mismatch_header.magic = kBinaryTensorMagic;
  size_mismatch_header.dtype = 0;
  size_mismatch_header.ndim = 1;
  size_mismatch_header.dims[0] = 4;

  std::vector<uint8_t> size_mismatch_buf(64 + 8, 0);
  std::memcpy(size_mismatch_buf.data(), &size_mismatch_header, sizeof(size_mismatch_header));
  auto res4 = deserializeBinaryTensor(size_mismatch_buf);
  EXPECT_FALSE(res4.isOk());
}

TEST(BinaryProtocolTest, EndToEndBinaryInference) {
  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 2;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());
  ASSERT_TRUE(server.startAsync().isOk());

  // Build binary tensor payload
  BinaryTensorHeader header;
  header.magic = kBinaryTensorMagic;
  header.dtype = static_cast<uint8_t>(DataType::Float32);
  header.ndim = 2;
  header.dims[0] = 1;
  header.dims[1] = 4;

  std::vector<float> input_data = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<uint8_t> body_bytes(sizeof(header) + input_data.size() * sizeof(float));
  std::memcpy(body_bytes.data(), &header, sizeof(header));
  std::memcpy(body_bytes.data() + sizeof(header), input_data.data(),
              input_data.size() * sizeof(float));

  std::string body_str(reinterpret_cast<const char*>(body_bytes.data()), body_bytes.size());
  std::unordered_map<std::string, std::string> headers = {
      {"Content-Type", "application/octet-stream"}, {"X-Pyreth-Execution-Mode", "sync"}};

  auto resp = sendHttpRequest("127.0.0.1", server.port(), "POST", "/v1/models/identity/infer",
                              headers, body_str);
  EXPECT_EQ(resp.status_code, 200);

  // Validate response binary format
  std::span<const uint8_t> resp_bytes(reinterpret_cast<const uint8_t*>(resp.body.data()),
                                      resp.body.size());
  auto out_tensor_res = deserializeBinaryTensor(resp_bytes);
  ASSERT_TRUE(out_tensor_res.isOk()) << out_tensor_res.error().message();

  const auto& out_tensor = out_tensor_res.value();
  EXPECT_EQ(out_tensor.ndim(), 2u);
  EXPECT_EQ(out_tensor.shape()[0], 1);
  EXPECT_EQ(out_tensor.shape()[1], 4);

  const float* out_ptr = static_cast<const float*>(out_tensor.data());
  for (size_t i = 0; i < input_data.size(); ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], input_data[i]);
  }

  server.stop();
}

// =========================================================================
// 2. Server-Sent Events (SSE) Streaming Tests
// =========================================================================

TEST(ServerStreamingTest, SSEEventFormatting) {
  std::string frame = formatSSEEvent("step", "{\"token\":\"world\"}");
  EXPECT_EQ(frame, "event: step\ndata: {\"token\":\"world\"}\n\n");

  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 2;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());
  ASSERT_TRUE(server.startAsync().isOk());

  auto resp = sendHttpRequest("127.0.0.1", server.port(), "GET", "/v1/models/identity/stream");
  EXPECT_EQ(resp.status_code, 200);
  EXPECT_NE(resp.headers.find("content-type"), resp.headers.end());
  EXPECT_NE(resp.headers["content-type"].find("text/event-stream"), std::string::npos);

  // Expect event frames: step 0, step 1, step 2, done
  EXPECT_NE(resp.body.find("event: step"), std::string::npos);
  EXPECT_NE(resp.body.find("event: done"), std::string::npos);
  EXPECT_NE(resp.body.find("\"status\":\"finished\""), std::string::npos);

  server.stop();
}

TEST(ServerStreamingTest, SSEClientEarlyDisconnect) {
  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 2;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());
  ASSERT_TRUE(server.startAsync().isOk());

  // Connect socket, send GET /v1/models/identity/stream, receive initial chunk, then close abruptly
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(server.port());
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

  socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_TRUE(isValidSocket(sock));
  ASSERT_EQ(::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  std::string req = "GET /v1/models/identity/stream HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
#ifdef _WIN32
  ::send(sock, req.data(), static_cast<int>(req.size()), 0);
#else
  ::send(sock, req.data(), req.size(), 0);
#endif

  // Read small chunk
  char buf[256];
#ifdef _WIN32
  int n = ::recv(sock, buf, sizeof(buf), 0);
#else
  ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
  EXPECT_GT(n, 0);

  // Close client socket immediately mid-stream
  closeSocket(sock);

  // Sleep slightly to let server thread observe socket closure without crashes or leaks
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  server.stop();
}

// =========================================================================
// 3. WebSocket Full-Duplex Tests
// =========================================================================

TEST(ServerWebSocketTest, HandshakeAndEcho) {
  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 2;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());
  ASSERT_TRUE(server.startAsync().isOk());

  // Connect socket and perform RFC 6455 WebSocket Handshake
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(server.port());
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

  socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_TRUE(isValidSocket(sock));
  ASSERT_EQ(::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);

  std::string handshake =
      "GET /v1/models/identity/ws HTTP/1.1\r\n"
      "Host: 127.0.0.1\r\n"
      "Upgrade: websocket\r\n"
      "Connection: Upgrade\r\n"
      "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
      "Sec-WebSocket-Version: 13\r\n\r\n";

#ifdef _WIN32
  ::send(sock, handshake.data(), static_cast<int>(handshake.size()), 0);
#else
  ::send(sock, handshake.data(), handshake.size(), 0);
#endif

  char buf[1024];
#ifdef _WIN32
  int n = ::recv(sock, buf, sizeof(buf), 0);
#else
  ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
  ASSERT_GT(n, 0);
  std::string upgrade_resp(buf, static_cast<size_t>(n));
  EXPECT_NE(upgrade_resp.find("101 Switching Protocols"), std::string::npos);

  // 1. Send text JSON inference frame
  std::string kserve_json =
      "{\"inputs\":[{\"name\":\"input\",\"shape\":[1,4],\"datatype\":\"FP32\",\"data\":[1.0,2.0,3."
      "0,4.0]}]}";
  auto text_frame = makeWsClientFrame(
      std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(kserve_json.data()),
                               kserve_json.size()),
      0x01);
#ifdef _WIN32
  ::send(sock, reinterpret_cast<const char*>(text_frame.data()),
         static_cast<int>(text_frame.size()), 0);
#else
  ::send(sock, text_frame.data(), text_frame.size(), 0);
#endif

  auto resp_frame = readWsServerFrame(sock);
  EXPECT_EQ(resp_frame.opcode, 0x01);  // Text frame
  std::string resp_text(reinterpret_cast<const char*>(resp_frame.payload.data()),
                        resp_frame.payload.size());
  EXPECT_NE(resp_text.find("\"model_name\":\"identity\""), std::string::npos);
  EXPECT_NE(resp_text.find("1.0"), std::string::npos);

  // 2. Send binary tensor frame
  BinaryTensorHeader bin_header;
  bin_header.magic = kBinaryTensorMagic;
  bin_header.dtype = static_cast<uint8_t>(DataType::Float32);
  bin_header.ndim = 2;
  bin_header.dims[0] = 1;
  bin_header.dims[1] = 4;

  std::vector<float> bin_data = {5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<uint8_t> bin_payload(sizeof(bin_header) + bin_data.size() * sizeof(float));
  std::memcpy(bin_payload.data(), &bin_header, sizeof(bin_header));
  std::memcpy(bin_payload.data() + sizeof(bin_header), bin_data.data(),
              bin_data.size() * sizeof(float));

  auto bin_frame = makeWsClientFrame(bin_payload, 0x02);  // Binary frame
#ifdef _WIN32
  ::send(sock, reinterpret_cast<const char*>(bin_frame.data()), static_cast<int>(bin_frame.size()),
         0);
#else
  ::send(sock, bin_frame.data(), bin_frame.size(), 0);
#endif

  auto resp_bin_frame = readWsServerFrame(sock);
  EXPECT_EQ(resp_bin_frame.opcode, 0x02);  // Binary frame
  auto out_res = deserializeBinaryTensor(resp_bin_frame.payload);
  ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
  const auto& out_t = out_res.value();
  const float* ptr = static_cast<const float*>(out_t.data());
  EXPECT_FLOAT_EQ(ptr[0], 5.0f);
  EXPECT_FLOAT_EQ(ptr[1], 6.0f);
  EXPECT_FLOAT_EQ(ptr[2], 7.0f);
  EXPECT_FLOAT_EQ(ptr[3], 8.0f);

  closeSocket(sock);
  server.stop();
}

TEST(ServerWebSocketTest, ConcurrentWebSocketSessions) {
  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 4;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());
  ASSERT_TRUE(server.startAsync().isOk());

  constexpr int kNumSessions = 10;
  std::latch start_latch(kNumSessions);
  std::vector<std::thread> threads;
  std::atomic<int> success_count{0};

  for (int i = 0; i < kNumSessions; ++i) {
    threads.emplace_back([&server, &start_latch, &success_count, i]() {
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(server.port());
      ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

      socket_t sock = ::socket(AF_INET, SOCK_STREAM, 0);
      if (!isValidSocket(sock))
        return;
      if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        closeSocket(sock);
        return;
      }

      std::string handshake =
          "GET /v1/models/identity/ws HTTP/1.1\r\n"
          "Host: 127.0.0.1\r\n"
          "Upgrade: websocket\r\n"
          "Connection: Upgrade\r\n"
          "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
          "Sec-WebSocket-Version: 13\r\n\r\n";

#ifdef _WIN32
      ::send(sock, handshake.data(), static_cast<int>(handshake.size()), 0);
#else
      ::send(sock, handshake.data(), handshake.size(), 0);
#endif

      char buf[1024];
#ifdef _WIN32
      int n = ::recv(sock, buf, sizeof(buf), 0);
#else
      ssize_t n = ::recv(sock, buf, sizeof(buf), 0);
#endif
      if (n <= 0) {
        closeSocket(sock);
        return;
      }

      start_latch.count_down();
      start_latch.wait();

      std::string req_json =
          "{\"inputs\":[{\"name\":\"input\",\"shape\":[1,4],\"datatype\":\"FP32\",\"data\":[" +
          std::to_string(i) + ".0,0.0,0.0,0.0]}]}";
      auto frame =
          makeWsClientFrame(std::span<const uint8_t>(
                                reinterpret_cast<const uint8_t*>(req_json.data()), req_json.size()),
                            0x01);

#ifdef _WIN32
      ::send(sock, reinterpret_cast<const char*>(frame.data()), static_cast<int>(frame.size()), 0);
#else
      ::send(sock, frame.data(), frame.size(), 0);
#endif

      auto resp_frame = readWsServerFrame(sock);
      if (resp_frame.opcode == 0x01) {
        std::string resp_str(reinterpret_cast<const char*>(resp_frame.payload.data()),
                             resp_frame.payload.size());
        if (resp_str.find(std::to_string(i) + ".0") != std::string::npos) {
          success_count.fetch_add(1, std::memory_order_relaxed);
        }
      }

      closeSocket(sock);
    });
  }

  for (auto& t : threads) {
    if (t.joinable())
      t.join();
  }

  EXPECT_EQ(success_count.load(), kNumSessions);
  server.stop();
}

// =========================================================================
// 4. Custom User-Defined Endpoints Tests
// =========================================================================

TEST(ServerCustomEndpointTest, PipelinePreprocessingAndPostprocessing) {
  ServerConfig config;
  config.host = "127.0.0.1";
  config.port = 0;
  config.io_threads = 2;

  HttpServer server(config);
  auto session = createDynamicSession();
  ASSERT_NE(session, nullptr);
  ASSERT_TRUE(server.registerModel("identity", session).isOk());

  // Define custom endpoint: POST /api/v1/sentiment
  server.route("/api/v1/sentiment")
      .method(HttpMethod::Post)
      .model("identity")
      .mode(ExecutionMode::Async)
      .preprocess(
          [](const CustomHttpRequest& req, ArenaAllocator& arena) -> Result<std::vector<Tensor>> {
            Json::Value root;
            Json::CharReaderBuilder reader;
            std::string errs;
            std::istringstream s(req.body());
            if (!Json::parseFromStream(reader, s, &root, &errs) || !root.isMember("score")) {
              return Result<std::vector<Tensor>>::failure(StatusCode::INVALID_ARGUMENT,
                                                          "Missing required 'score' field");
            }

            float score = root["score"].asFloat();
            void* mem = arena.allocate(4 * sizeof(float), alignof(float));
            float* fmem = static_cast<float*>(mem);
            fmem[0] = score;
            fmem[1] = score * 2.0f;
            fmem[2] = score * 3.0f;
            fmem[3] = score * 4.0f;

            Shape shape({1, 4});
            auto t_res = Tensor::fromBuffer(mem, shape, DataType::Float32);
            if (!t_res.isOk()) {
              return Result<std::vector<Tensor>>::failure(t_res.error());
            }

            std::vector<Tensor> tensors;
            tensors.push_back(std::move(t_res.value()));
            return Result<std::vector<Tensor>>::success(std::move(tensors));
          })
      .postprocess([](const std::vector<Tensor>& outputs,
                      const CustomHttpRequest& /*req*/) -> Result<CustomHttpResponse> {
        if (outputs.empty()) {
          return Result<CustomHttpResponse>::failure(StatusCode::INTERNAL_ERROR, "No outputs");
        }

        const float* ptr = static_cast<const float*>(outputs[0].data());
        Json::Value resp_json;
        resp_json["sentiment"] = ptr[0] > 0.5f ? "positive" : "negative";
        resp_json["score"] = ptr[0];

        Json::StreamWriterBuilder writer;
        writer["indentation"] = "";
        return Result<CustomHttpResponse>::success(
            CustomHttpResponse::json(Json::writeString(writer, resp_json), 200));
      });

  ASSERT_TRUE(server.startAsync().isOk());

  // 1. Success path
  std::string valid_body = "{\"score\":0.85}";
  auto resp1 = sendHttpRequest("127.0.0.1", server.port(), "POST", "/api/v1/sentiment",
                               {{"Content-Type", "application/json"}}, valid_body);
  EXPECT_EQ(resp1.status_code, 200);
  EXPECT_NE(resp1.body.find("\"sentiment\":\"positive\""), std::string::npos);

  // 2. Preprocessing error triggers HTTP 422 Unprocessable Entity
  std::string invalid_body = "{\"invalid_field\":123}";
  auto resp2 = sendHttpRequest("127.0.0.1", server.port(), "POST", "/api/v1/sentiment",
                               {{"Content-Type", "application/json"}}, invalid_body);
  EXPECT_EQ(resp2.status_code, 422);
  EXPECT_NE(resp2.body.find("Missing required 'score' field"), std::string::npos);

  server.stop();
}
