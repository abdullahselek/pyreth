#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <pyreth/common.hpp>
#include <pyreth/pyreth.hpp>

namespace {

// Sanity check that the GoogleTest environment itself is wired up correctly.
TEST(Foundation, GTestEnvironmentIsWired) {
  SUCCEED() << "GoogleTest is configured and running under CTest.";
}

// The engine publishes a non-empty, semver-shaped version string.
TEST(Foundation, VersionStringIsWellFormed) {
  const char* version = pyreth::PYRETH_VERSION;

  ASSERT_NE(version, nullptr);
  const std::size_t length = std::strlen(version);
  ASSERT_NE(length, 0u);
  // "MAJOR.MINOR.PATCH" -> exactly two dots.
  EXPECT_EQ(std::count(version, version + length, '.'), 2);
}

// The core engine object can be constructed and destroyed cleanly.
TEST(Foundation, EngineLifecycle) {
  pyreth::Pyreth engine;
  SUCCEED();
}

// The convenience run() helper executes cleanly.
TEST(Foundation, RunHelper) {
  pyreth::run();
  SUCCEED();
}

}  // namespace
