#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <pyreth/common.hpp>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace pyreth {

/**
 * @enum StatusCode
 * @brief Discrete enumeration of engine status and error conditions.
 */
enum class StatusCode : uint8_t {
  OK = 0,
  INVALID_ARGUMENT,
  INVALID_DIMENSIONS,
  MODEL_NOT_FOUND,
  MODEL_LOAD_FAILED,
  BACKEND_ERROR,
  BACKEND_UNAVAILABLE,
  QUEUE_FULL,
  QUEUE_EMPTY,
  ALLOCATION_FAILED,
  TIMEOUT,
  INTERNAL_ERROR,
  UNKNOWN
};

/**
 * @brief Convert a StatusCode to its canonical human-readable string
 * representation.
 */
[[nodiscard]] constexpr std::string_view statusCodeToString(StatusCode code) noexcept {
  switch (code) {
    case StatusCode::OK:
      return "OK";
    case StatusCode::INVALID_ARGUMENT:
      return "Invalid argument";
    case StatusCode::INVALID_DIMENSIONS:
      return "Invalid dimensions";
    case StatusCode::MODEL_NOT_FOUND:
      return "Model not found";
    case StatusCode::MODEL_LOAD_FAILED:
      return "Model load failed";
    case StatusCode::BACKEND_ERROR:
      return "Backend error";
    case StatusCode::BACKEND_UNAVAILABLE:
      return "Backend unavailable";
    case StatusCode::QUEUE_FULL:
      return "Queue full";
    case StatusCode::QUEUE_EMPTY:
      return "Queue empty";
    case StatusCode::ALLOCATION_FAILED:
      return "Allocation failed";
    case StatusCode::TIMEOUT:
      return "Timeout";
    case StatusCode::INTERNAL_ERROR:
      return "Internal error";
    case StatusCode::UNKNOWN:
    default:
      return "Unknown error";
  }
}

/**
 * @class Error
 * @brief Zero-allocation error representation holding a StatusCode and
 * string_view description.
 */
class Error {
public:
  constexpr Error() noexcept : code_(StatusCode::OK), message_("") {}

  constexpr Error(StatusCode code, std::string_view message = "") noexcept
      : code_(code), message_(message.empty() ? statusCodeToString(code) : message) {}

  [[nodiscard]] constexpr StatusCode code() const noexcept { return code_; }
  [[nodiscard]] constexpr std::string_view message() const noexcept { return message_; }
  [[nodiscard]] constexpr bool isOk() const noexcept { return code_ == StatusCode::OK; }

  // Factory methods for convenient construction
  static constexpr Error ok() noexcept { return Error(StatusCode::OK); }
  static constexpr Error invalidArgument(std::string_view msg = "") noexcept {
    return Error(StatusCode::INVALID_ARGUMENT, msg);
  }
  static constexpr Error invalidDimensions(std::string_view msg = "") noexcept {
    return Error(StatusCode::INVALID_DIMENSIONS, msg);
  }
  static constexpr Error modelNotFound(std::string_view msg = "") noexcept {
    return Error(StatusCode::MODEL_NOT_FOUND, msg);
  }
  static constexpr Error modelLoadFailed(std::string_view msg = "") noexcept {
    return Error(StatusCode::MODEL_LOAD_FAILED, msg);
  }
  static constexpr Error backendError(std::string_view msg = "") noexcept {
    return Error(StatusCode::BACKEND_ERROR, msg);
  }
  static constexpr Error backendUnavailable(std::string_view msg = "") noexcept {
    return Error(StatusCode::BACKEND_UNAVAILABLE, msg);
  }
  static constexpr Error queueFull(std::string_view msg = "") noexcept {
    return Error(StatusCode::QUEUE_FULL, msg);
  }
  static constexpr Error queueEmpty(std::string_view msg = "") noexcept {
    return Error(StatusCode::QUEUE_EMPTY, msg);
  }
  static constexpr Error allocationFailed(std::string_view msg = "") noexcept {
    return Error(StatusCode::ALLOCATION_FAILED, msg);
  }
  static constexpr Error timeout(std::string_view msg = "") noexcept {
    return Error(StatusCode::TIMEOUT, msg);
  }
  static constexpr Error internalError(std::string_view msg = "") noexcept {
    return Error(StatusCode::INTERNAL_ERROR, msg);
  }
  static constexpr Error unknown(std::string_view msg = "") noexcept {
    return Error(StatusCode::UNKNOWN, msg);
  }

  constexpr bool operator==(const Error& other) const noexcept {
    return code_ == other.code_ && message_ == other.message_;
  }
  constexpr bool operator!=(const Error& other) const noexcept { return !(*this == other); }
  constexpr bool operator==(StatusCode sc) const noexcept { return code_ == sc; }
  constexpr bool operator!=(StatusCode sc) const noexcept { return code_ != sc; }

private:
  StatusCode code_{StatusCode::OK};
  std::string_view message_{""};
};

// Forward declaration of primary Result template
template <typename T, typename E = Error>
class [[nodiscard]] Result;

/**
 * @class Result<void, E>
 * @brief Specialization of Result for operations that return no value on
 * success.
 */
template <typename E>
class [[nodiscard]] Result<void, E> {
public:
  using ValueType = void;
  using ErrorType = E;

  constexpr Result() noexcept : error_(std::nullopt) {}

  constexpr Result(const E& err) : error_(err) {}
  constexpr Result(E&& err) noexcept(std::is_nothrow_move_constructible_v<E>)
      : error_(std::move(err)) {}

  template <typename S>
    requires std::is_same_v<std::decay_t<S>, StatusCode> && std::is_constructible_v<E, S>
  constexpr Result(S code) : error_(E(code)) {}

  static constexpr Result success() noexcept { return Result(); }

  static constexpr Result ok() noexcept { return success(); }

  template <typename... Args>
  static constexpr Result failure(Args&&... args) {
    return Result(E(std::forward<Args>(args)...));
  }

  [[nodiscard]] constexpr bool isOk() const noexcept { return !error_.has_value(); }

  [[nodiscard]] constexpr bool isError() const noexcept { return error_.has_value(); }

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return isOk(); }

  [[nodiscard]] constexpr bool hasValue() const noexcept { return isOk(); }

  constexpr void value() const noexcept {}

  [[nodiscard]] constexpr const E& error() const& noexcept { return *error_; }

  [[nodiscard]] constexpr E& error() & noexcept { return *error_; }

  [[nodiscard]] constexpr E&& error() && noexcept { return *std::move(error_); }

  template <typename Fn>
  constexpr auto andThen(Fn&& fn) const& {
    using ReturnType = std::invoke_result_t<Fn>;
    if (isOk()) {
      return std::invoke(std::forward<Fn>(fn));
    }
    return ReturnType(error());
  }

  template <typename Fn>
  constexpr auto andThen(Fn&& fn) && {
    using ReturnType = std::invoke_result_t<Fn>;
    if (isOk()) {
      return std::invoke(std::forward<Fn>(fn));
    }
    return ReturnType(std::move(*this).error());
  }

  template <typename Fn>
  constexpr auto transform(Fn&& fn) const& {
    using ReturnType = std::invoke_result_t<Fn>;
    if constexpr (std::is_void_v<ReturnType>) {
      if (isOk()) {
        std::invoke(std::forward<Fn>(fn));
        return Result<void, E>::success();
      }
      return Result<void, E>(error());
    } else {
      using ResultType = Result<ReturnType, E>;
      if (isOk()) {
        return ResultType(std::invoke(std::forward<Fn>(fn)));
      }
      return ResultType(error());
    }
  }

  template <typename Fn>
  constexpr auto transform(Fn&& fn) && {
    using ReturnType = std::invoke_result_t<Fn>;
    if constexpr (std::is_void_v<ReturnType>) {
      if (isOk()) {
        std::invoke(std::forward<Fn>(fn));
        return Result<void, E>::success();
      }
      return Result<void, E>(std::move(*this).error());
    } else {
      using ResultType = Result<ReturnType, E>;
      if (isOk()) {
        return ResultType(std::invoke(std::forward<Fn>(fn)));
      }
      return ResultType(std::move(*this).error());
    }
  }

  template <typename Fn>
  constexpr auto orElse(Fn&& fn) const& {
    if (isError()) {
      return std::invoke(std::forward<Fn>(fn), error());
    }
    return *this;
  }

  template <typename Fn>
  constexpr auto orElse(Fn&& fn) && {
    if (isError()) {
      return std::invoke(std::forward<Fn>(fn), std::move(*this).error());
    }
    return std::move(*this);
  }

private:
  std::optional<E> error_{std::nullopt};
};

/**
 * @class Result<T, E>
 * @brief Monadic return type representing either a successful value T or an
 * error E.
 *
 * Implements zero-allocation, exception-free value-or-error semantics with
 * C++20 monadic chaining (andThen, transform, orElse).
 */
template <typename T, typename E>
class [[nodiscard]] Result {
public:
  using ValueType = T;
  using ErrorType = E;

  constexpr Result(const T& val) : storage_(std::in_place_index<0>, val) {}
  constexpr Result(T&& val) noexcept(std::is_nothrow_move_constructible_v<T>)
      : storage_(std::in_place_index<0>, std::move(val)) {}

  constexpr Result(const E& err) : storage_(std::in_place_index<1>, err) {}
  constexpr Result(E&& err) noexcept(std::is_nothrow_move_constructible_v<E>)
      : storage_(std::in_place_index<1>, std::move(err)) {}

  template <typename S>
    requires std::is_same_v<std::decay_t<S>, StatusCode> && std::is_constructible_v<E, S>
  constexpr Result(S code) : storage_(std::in_place_index<1>, E(code)) {}

  template <typename... Args>
  static constexpr Result success(Args&&... args) {
    return Result(T(std::forward<Args>(args)...));
  }

  template <typename... Args>
  static constexpr Result failure(Args&&... args) {
    return Result(E(std::forward<Args>(args)...));
  }

  [[nodiscard]] constexpr bool isOk() const noexcept { return storage_.index() == 0; }

  [[nodiscard]] constexpr bool isError() const noexcept { return storage_.index() == 1; }

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return isOk(); }

  [[nodiscard]] constexpr bool hasValue() const noexcept { return isOk(); }

  [[nodiscard]] constexpr const T& value() const& { return std::get<0>(storage_); }

  [[nodiscard]] constexpr T& value() & { return std::get<0>(storage_); }

  [[nodiscard]] constexpr T&& value() && { return std::get<0>(std::move(storage_)); }

  [[nodiscard]] constexpr const E& error() const& noexcept { return std::get<1>(storage_); }

  [[nodiscard]] constexpr E& error() & noexcept { return std::get<1>(storage_); }

  [[nodiscard]] constexpr E&& error() && noexcept { return std::get<1>(std::move(storage_)); }

  template <typename U>
  [[nodiscard]] constexpr T valueOr(U&& default_value) const& {
    if (isOk()) {
      return value();
    }
    return static_cast<T>(std::forward<U>(default_value));
  }

  template <typename U>
  [[nodiscard]] constexpr T valueOr(U&& default_value) && {
    if (isOk()) {
      return std::move(*this).value();
    }
    return static_cast<T>(std::forward<U>(default_value));
  }

  template <typename Fn>
  constexpr auto andThen(Fn&& fn) const& {
    using ReturnType = std::invoke_result_t<Fn, const T&>;
    if (isOk()) {
      return std::invoke(std::forward<Fn>(fn), value());
    }
    return ReturnType(error());
  }

  template <typename Fn>
  constexpr auto andThen(Fn&& fn) && {
    using ReturnType = std::invoke_result_t<Fn, T&&>;
    if (isOk()) {
      return std::invoke(std::forward<Fn>(fn), std::move(*this).value());
    }
    return ReturnType(std::move(*this).error());
  }

  template <typename Fn>
  constexpr auto transform(Fn&& fn) const& {
    using ReturnType = std::invoke_result_t<Fn, const T&>;
    if constexpr (std::is_void_v<ReturnType>) {
      if (isOk()) {
        std::invoke(std::forward<Fn>(fn), value());
        return Result<void, E>::success();
      }
      return Result<void, E>(error());
    } else {
      using ResultType = Result<ReturnType, E>;
      if (isOk()) {
        return ResultType(std::invoke(std::forward<Fn>(fn), value()));
      }
      return ResultType(error());
    }
  }

  template <typename Fn>
  constexpr auto transform(Fn&& fn) && {
    using ReturnType = std::invoke_result_t<Fn, T&&>;
    if constexpr (std::is_void_v<ReturnType>) {
      if (isOk()) {
        std::invoke(std::forward<Fn>(fn), std::move(*this).value());
        return Result<void, E>::success();
      }
      return Result<void, E>(std::move(*this).error());
    } else {
      using ResultType = Result<ReturnType, E>;
      if (isOk()) {
        return ResultType(std::invoke(std::forward<Fn>(fn), std::move(*this).value()));
      }
      return ResultType(std::move(*this).error());
    }
  }

  template <typename Fn>
  constexpr auto orElse(Fn&& fn) const& {
    if (isError()) {
      return std::invoke(std::forward<Fn>(fn), error());
    }
    return *this;
  }

  template <typename Fn>
  constexpr auto orElse(Fn&& fn) && {
    if (isError()) {
      return std::invoke(std::forward<Fn>(fn), std::move(*this).error());
    }
    return std::move(*this);
  }

private:
  std::variant<T, E> storage_;
};

/**
 * @brief Type alias representing operation status with no payload on success.
 */
using Status = Result<void, Error>;

}  // namespace pyreth
