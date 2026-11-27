#pragma once

#include <functional>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <pyreth/server/execution_mode.hpp>
#include <pyreth/tensor.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pyreth::server {

enum class HttpMethod {
  Get,
  Post,
  Put,
  Delete,
  Options,
};

PYRETH_API std::string_view httpMethodToString(HttpMethod method) noexcept;
PYRETH_API Result<HttpMethod> stringToHttpMethod(std::string_view str) noexcept;

/**
 * @class CustomHttpRequest
 * @brief Lightweight wrapper around an incoming HTTP request for custom endpoint pipelines.
 */
class PYRETH_API CustomHttpRequest {
public:
  CustomHttpRequest() = default;
  CustomHttpRequest(HttpMethod method, std::string path, std::string body,
                    std::unordered_map<std::string, std::string> headers)
      : method_(method),
        path_(std::move(path)),
        body_(std::move(body)),
        headers_(std::move(headers)) {}

  [[nodiscard]] HttpMethod method() const noexcept { return method_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] const std::string& body() const noexcept { return body_; }
  [[nodiscard]] const std::unordered_map<std::string, std::string>& headers() const noexcept {
    return headers_;
  }

  [[nodiscard]] std::string getHeader(std::string_view name) const {
    for (const auto& [k, v] : headers_) {
      if (k.size() == name.size()) {
        bool match = true;
        for (size_t i = 0; i < k.size(); ++i) {
          if (std::tolower(static_cast<unsigned char>(k[i])) !=
              std::tolower(static_cast<unsigned char>(name[i]))) {
            match = false;
            break;
          }
        }
        if (match)
          return v;
      }
    }
    return "";
  }

private:
  HttpMethod method_{HttpMethod::Get};
  std::string path_;
  std::string body_;
  std::unordered_map<std::string, std::string> headers_;
};

/**
 * @class CustomHttpResponse
 * @brief Response container produced by custom endpoint postprocessing.
 */
class PYRETH_API CustomHttpResponse {
public:
  CustomHttpResponse() = default;
  explicit CustomHttpResponse(int status_code, std::string body = "",
                              std::string content_type = "application/json")
      : status_code_(status_code), body_(std::move(body)), content_type_(std::move(content_type)) {}

  [[nodiscard]] int statusCode() const noexcept { return status_code_; }
  void setStatusCode(int code) noexcept { status_code_ = code; }

  [[nodiscard]] const std::string& body() const noexcept { return body_; }
  void setBody(std::string body) { body_ = std::move(body); }

  [[nodiscard]] const std::string& contentType() const noexcept { return content_type_; }
  void setContentType(std::string content_type) { content_type_ = std::move(content_type); }

  [[nodiscard]] const std::unordered_map<std::string, std::string>& headers() const noexcept {
    return headers_;
  }
  void addHeader(std::string key, std::string value) {
    headers_.emplace(std::move(key), std::move(value));
  }

  static CustomHttpResponse json(std::string json_str, int status_code = 200) {
    return CustomHttpResponse(status_code, std::move(json_str), "application/json");
  }

  static CustomHttpResponse text(std::string text_str, int status_code = 200) {
    return CustomHttpResponse(status_code, std::move(text_str), "text/plain");
  }

  static CustomHttpResponse binary(std::string bytes, int status_code = 200) {
    return CustomHttpResponse(status_code, std::move(bytes), "application/octet-stream");
  }

  static CustomHttpResponse error(int status_code, std::string message) {
    std::string json_err = "{\"error\":\"" + std::move(message) + "\"}";
    return CustomHttpResponse(status_code, std::move(json_err), "application/json");
  }

private:
  int status_code_{200};
  std::string body_;
  std::string content_type_{"application/json"};
  std::unordered_map<std::string, std::string> headers_;
};

using PreprocessFn =
    std::function<Result<std::vector<Tensor>>(const CustomHttpRequest&, ArenaAllocator&)>;
using PostprocessFn =
    std::function<Result<CustomHttpResponse>(const std::vector<Tensor>&, const CustomHttpRequest&)>;

/**
 * @class EndpointBuilder
 * @brief Fluent builder for custom user-defined endpoints.
 */
class PYRETH_API EndpointBuilder {
public:
  explicit EndpointBuilder(std::string path) : path_(std::move(path)) {}

  EndpointBuilder& method(HttpMethod m) {
    method_ = m;
    return *this;
  }

  EndpointBuilder& model(std::string model_name) {
    model_name_ = std::move(model_name);
    return *this;
  }

  EndpointBuilder& mode(ExecutionMode exec_mode) {
    mode_ = exec_mode;
    return *this;
  }

  EndpointBuilder& preprocess(PreprocessFn fn) {
    preprocess_fn_ = std::move(fn);
    return *this;
  }

  EndpointBuilder& postprocess(PostprocessFn fn) {
    postprocess_fn_ = std::move(fn);
    return *this;
  }

  [[nodiscard]] const std::string& path() const noexcept { return path_; }
  [[nodiscard]] HttpMethod httpMethod() const noexcept { return method_; }
  [[nodiscard]] const std::string& modelName() const noexcept { return model_name_; }
  [[nodiscard]] ExecutionMode executionMode() const noexcept { return mode_; }
  [[nodiscard]] const PreprocessFn& preprocessFn() const noexcept { return preprocess_fn_; }
  [[nodiscard]] const PostprocessFn& postprocessFn() const noexcept { return postprocess_fn_; }

private:
  std::string path_;
  HttpMethod method_{HttpMethod::Post};
  std::string model_name_;
  ExecutionMode mode_{ExecutionMode::Async};
  PreprocessFn preprocess_fn_;
  PostprocessFn postprocess_fn_;
};

}  // namespace pyreth::server
