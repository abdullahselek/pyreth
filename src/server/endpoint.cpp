#include <algorithm>
#include <cctype>
#include <pyreth/server/endpoint.hpp>

namespace pyreth::server {

std::string_view httpMethodToString(HttpMethod method) noexcept {
  switch (method) {
    case HttpMethod::Get:
      return "GET";
    case HttpMethod::Post:
      return "POST";
    case HttpMethod::Put:
      return "PUT";
    case HttpMethod::Delete:
      return "DELETE";
    case HttpMethod::Options:
      return "OPTIONS";
  }
  return "UNKNOWN";
}

Result<HttpMethod> stringToHttpMethod(std::string_view str) noexcept {
  std::string upper;
  upper.reserve(str.size());
  for (char c : str) {
    upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }

  if (upper == "GET")
    return Result<HttpMethod>::success(HttpMethod::Get);
  if (upper == "POST")
    return Result<HttpMethod>::success(HttpMethod::Post);
  if (upper == "PUT")
    return Result<HttpMethod>::success(HttpMethod::Put);
  if (upper == "DELETE")
    return Result<HttpMethod>::success(HttpMethod::Delete);
  if (upper == "OPTIONS")
    return Result<HttpMethod>::success(HttpMethod::Options);

  return Result<HttpMethod>::failure(StatusCode::INVALID_ARGUMENT,
                                     "Invalid HTTP method: " + std::string(str));
}

}  // namespace pyreth::server
