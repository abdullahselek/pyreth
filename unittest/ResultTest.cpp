#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <type_traits>

#include <pyreth/result.hpp>

namespace {

using namespace pyreth;

// =============================================================================
// StatusCode & Error Tests
// =============================================================================

TEST(ResultTest, StatusCodeToStringConversions) {
  EXPECT_EQ(statusCodeToString(StatusCode::OK), "OK");
  EXPECT_EQ(statusCodeToString(StatusCode::INVALID_ARGUMENT), "Invalid argument");
  EXPECT_EQ(statusCodeToString(StatusCode::INVALID_DIMENSIONS), "Invalid dimensions");
  EXPECT_EQ(statusCodeToString(StatusCode::MODEL_NOT_FOUND), "Model not found");
  EXPECT_EQ(statusCodeToString(StatusCode::MODEL_LOAD_FAILED), "Model load failed");
  EXPECT_EQ(statusCodeToString(StatusCode::BACKEND_ERROR), "Backend error");
  EXPECT_EQ(statusCodeToString(StatusCode::BACKEND_UNAVAILABLE), "Backend unavailable");
  EXPECT_EQ(statusCodeToString(StatusCode::QUEUE_FULL), "Queue full");
  EXPECT_EQ(statusCodeToString(StatusCode::QUEUE_EMPTY), "Queue empty");
  EXPECT_EQ(statusCodeToString(StatusCode::ALLOCATION_FAILED), "Allocation failed");
  EXPECT_EQ(statusCodeToString(StatusCode::TIMEOUT), "Timeout");
  EXPECT_EQ(statusCodeToString(StatusCode::INTERNAL_ERROR), "Internal error");
  EXPECT_EQ(statusCodeToString(StatusCode::UNKNOWN), "Unknown error");
}

TEST(ResultTest, ErrorDefaultConstruction) {
  constexpr Error err;
  EXPECT_TRUE(err.isOk());
  EXPECT_EQ(err.code(), StatusCode::OK);
  EXPECT_EQ(err.message(), "");
}

TEST(ResultTest, ErrorCustomMessageAndFallback) {
  const Error errWithCustom(StatusCode::INVALID_ARGUMENT, "Custom parameter error");
  EXPECT_FALSE(errWithCustom.isOk());
  EXPECT_EQ(errWithCustom.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(errWithCustom.message(), "Custom parameter error");

  const Error errWithFallback(StatusCode::MODEL_NOT_FOUND);
  EXPECT_FALSE(errWithFallback.isOk());
  EXPECT_EQ(errWithFallback.code(), StatusCode::MODEL_NOT_FOUND);
  EXPECT_EQ(errWithFallback.message(), "Model not found");
}

TEST(ResultTest, ErrorFactoriesAndComparisons) {
  const Error okErr = Error::ok();
  EXPECT_TRUE(okErr.isOk());
  EXPECT_EQ(okErr, StatusCode::OK);

  const Error modelNotFound = Error::modelNotFound("model.onnx missing");
  EXPECT_FALSE(modelNotFound.isOk());
  EXPECT_EQ(modelNotFound.code(), StatusCode::MODEL_NOT_FOUND);
  EXPECT_EQ(modelNotFound, StatusCode::MODEL_NOT_FOUND);
  EXPECT_NE(modelNotFound, StatusCode::OK);

  const Error queueFull = Error::queueFull();
  EXPECT_EQ(queueFull, StatusCode::QUEUE_FULL);

  const Error backendErr = Error::backendError("Metal initialization failed");
  EXPECT_EQ(backendErr, StatusCode::BACKEND_ERROR);
  EXPECT_EQ(backendErr.message(), "Metal initialization failed");

  const Error invalidDims = Error::invalidDimensions();
  EXPECT_EQ(invalidDims, StatusCode::INVALID_DIMENSIONS);
}

// =============================================================================
// Status (Result<void, Error>) Tests
// =============================================================================

TEST(ResultTest, StatusSuccessAndError) {
  const Status okStatus = Status::ok();
  EXPECT_TRUE(okStatus.isOk());
  EXPECT_FALSE(okStatus.isError());
  EXPECT_TRUE(static_cast<bool>(okStatus));
  EXPECT_TRUE(okStatus.hasValue());

  const Status failStatus = Error::modelNotFound("Failed to find model");
  EXPECT_FALSE(failStatus.isOk());
  EXPECT_TRUE(failStatus.isError());
  EXPECT_FALSE(static_cast<bool>(failStatus));
  EXPECT_EQ(failStatus.error().code(), StatusCode::MODEL_NOT_FOUND);
  EXPECT_EQ(failStatus.error().message(), "Failed to find model");

  const Status fromCode = StatusCode::QUEUE_FULL;
  EXPECT_TRUE(fromCode.isError());
  EXPECT_EQ(fromCode.error().code(), StatusCode::QUEUE_FULL);
}

TEST(ResultTest, StatusMonadicAndThen) {
  bool called = false;
  auto step1 = [&called]() -> Status {
    called = true;
    return Status::ok();
  };

  Status s1 = Status::ok();
  Status res1 = s1.andThen(step1);
  EXPECT_TRUE(res1.isOk());
  EXPECT_TRUE(called);

  called = false;
  Status s2 = StatusCode::INVALID_ARGUMENT;
  Status res2 = s2.andThen(step1);
  EXPECT_TRUE(res2.isError());
  EXPECT_FALSE(called);
  EXPECT_EQ(res2.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(ResultTest, StatusMonadicTransform) {
  Status okStatus = Status::ok();
  auto transformed = okStatus.transform([]() -> int { return 42; });
  EXPECT_TRUE(transformed.isOk());
  EXPECT_EQ(transformed.value(), 42);

  Status errStatus = StatusCode::TIMEOUT;
  auto transformedErr = errStatus.transform([]() -> int { return 42; });
  EXPECT_TRUE(transformedErr.isError());
  EXPECT_EQ(transformedErr.error().code(), StatusCode::TIMEOUT);
}

TEST(ResultTest, StatusMonadicOrElse) {
  bool handled = false;
  Status errStatus = StatusCode::BACKEND_UNAVAILABLE;
  Status recovered = errStatus.orElse([&handled](const Error& err) -> Status {
    EXPECT_EQ(err.code(), StatusCode::BACKEND_UNAVAILABLE);
    handled = true;
    return Status::ok();
  });

  EXPECT_TRUE(handled);
  EXPECT_TRUE(recovered.isOk());

  handled = false;
  Status okStatus = Status::ok();
  Status untouched = okStatus.orElse([&handled](const Error&) -> Status {
    handled = true;
    return Status::ok();
  });

  EXPECT_FALSE(handled);
  EXPECT_TRUE(untouched.isOk());
}

// =============================================================================
// Result<T, Error> Tests
// =============================================================================

TEST(ResultTest, ResultWithValueSuccess) {
  Result<int> res(100);
  EXPECT_TRUE(res.isOk());
  EXPECT_FALSE(res.isError());
  EXPECT_TRUE(static_cast<bool>(res));
  EXPECT_TRUE(res.hasValue());
  EXPECT_EQ(res.value(), 100);
  EXPECT_EQ(res.valueOr(0), 100);

  // Factory creation
  auto resFactory = Result<std::string>::success("pyreth");
  EXPECT_TRUE(resFactory.isOk());
  EXPECT_EQ(resFactory.value(), "pyreth");
}

TEST(ResultTest, ResultWithErrorFailure) {
  Result<int> res(Error::allocationFailed("Out of arena memory"));
  EXPECT_FALSE(res.isOk());
  EXPECT_TRUE(res.isError());
  EXPECT_FALSE(static_cast<bool>(res));
  EXPECT_FALSE(res.hasValue());
  EXPECT_EQ(res.error().code(), StatusCode::ALLOCATION_FAILED);
  EXPECT_EQ(res.error().message(), "Out of arena memory");
  EXPECT_EQ(res.valueOr(999), 999);

  // Construction from StatusCode
  Result<int> resFromCode = StatusCode::QUEUE_FULL;
  EXPECT_TRUE(resFromCode.isError());
  EXPECT_EQ(resFromCode.error().code(), StatusCode::QUEUE_FULL);

  // Factory failure
  auto resFactory = Result<int>::failure(StatusCode::TIMEOUT, "Timed out waiting for task");
  EXPECT_TRUE(resFactory.isError());
  EXPECT_EQ(resFactory.error().code(), StatusCode::TIMEOUT);
}

TEST(ResultTest, ResultMoveOnlyType) {
  auto ptr = std::make_unique<int>(1234);
  Result<std::unique_ptr<int>> res(std::move(ptr));
  EXPECT_TRUE(res.isOk());
  ASSERT_NE(res.value(), nullptr);
  EXPECT_EQ(*res.value(), 1234);

  // Move value out
  std::unique_ptr<int> moved = std::move(res).value();
  ASSERT_NE(moved, nullptr);
  EXPECT_EQ(*moved, 1234);
}

TEST(ResultTest, ResultMonadicAndThen) {
  auto parsePositive = [](int x) -> Result<std::string> {
    if (x > 0) {
      return Result<std::string>::success("value:" + std::to_string(x));
    }
    return Result<std::string>::failure(StatusCode::INVALID_ARGUMENT, "Number must be positive");
  };

  // Success path
  Result<int> inputOk(42);
  Result<std::string> chainedOk = inputOk.andThen(parsePositive);
  EXPECT_TRUE(chainedOk.isOk());
  EXPECT_EQ(chainedOk.value(), "value:42");

  // Logic error path in chained function
  Result<int> inputNegative(-5);
  Result<std::string> chainedFail = inputNegative.andThen(parsePositive);
  EXPECT_TRUE(chainedFail.isError());
  EXPECT_EQ(chainedFail.error().code(), StatusCode::INVALID_ARGUMENT);

  // Short-circuit initial error path
  Result<int> initialErr(StatusCode::MODEL_LOAD_FAILED);
  Result<std::string> shortCircuited = initialErr.andThen(parsePositive);
  EXPECT_TRUE(shortCircuited.isError());
  EXPECT_EQ(shortCircuited.error().code(), StatusCode::MODEL_LOAD_FAILED);
}

TEST(ResultTest, ResultMonadicTransform) {
  Result<int> res(5);

  // Transform value: int -> double
  auto doubleRes = res.transform([](int x) { return static_cast<double>(x) * 2.5; });
  EXPECT_TRUE(doubleRes.isOk());
  EXPECT_DOUBLE_EQ(doubleRes.value(), 12.5);

  // Transform value to void
  bool executed = false;
  auto voidRes = res.transform([&executed](int x) {
    EXPECT_EQ(x, 5);
    executed = true;
  });
  EXPECT_TRUE(executed);
  EXPECT_TRUE(voidRes.isOk());

  // Error bypassed through transform
  Result<int> errRes(StatusCode::INTERNAL_ERROR);
  auto bypassed = errRes.transform([](int x) { return x * 10; });
  EXPECT_TRUE(bypassed.isError());
  EXPECT_EQ(bypassed.error().code(), StatusCode::INTERNAL_ERROR);
}

TEST(ResultTest, ResultMonadicOrElse) {
  bool recovered = false;
  Result<int> errRes(StatusCode::QUEUE_EMPTY);

  Result<int> fixed = errRes.orElse([&recovered](const Error& err) -> Result<int> {
    EXPECT_EQ(err.code(), StatusCode::QUEUE_EMPTY);
    recovered = true;
    return Result<int>::success(0);
  });

  EXPECT_TRUE(recovered);
  EXPECT_TRUE(fixed.isOk());
  EXPECT_EQ(fixed.value(), 0);

  // Success case should not invoke orElse
  recovered = false;
  Result<int> okRes(10);
  Result<int> untouched = okRes.orElse([&recovered](const Error&) -> Result<int> {
    recovered = true;
    return Result<int>::success(0);
  });

  EXPECT_FALSE(recovered);
  EXPECT_TRUE(untouched.isOk());
  EXPECT_EQ(untouched.value(), 10);
}

// =============================================================================
// Memory & Zero-Allocation Layout Verification
// =============================================================================

TEST(ResultTest, ZeroHeapAllocationLayout) {
  // Error should be compact (code + string_view = 1 + padding + 16 = 24 bytes on 64-bit)
  EXPECT_LE(sizeof(Error), 24u);

  // Status is optional<Error>, should fit within 32 bytes without dynamic allocation
  EXPECT_LE(sizeof(Status), 32u);

  // Result<int> should be variant<int, Error>, stack-allocated with zero heap overhead
  EXPECT_LE(sizeof(Result<int>), 32u);

  // Check noexcept moves
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<Error>);
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<Status>);
  EXPECT_TRUE(std::is_nothrow_move_constructible_v<Result<int>>);
}

} // namespace
