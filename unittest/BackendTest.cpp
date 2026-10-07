#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <pyreth/backend.hpp>

namespace {

using namespace pyreth;

TEST(BackendTest, RuntimeVersionIsNonEmptyAndValid) {
  const std::string version = Backend::getRuntimeVersion();
  ASSERT_FALSE(version.empty());
  // Version string should start with a digit (e.g. "1.19.2")
  EXPECT_TRUE(std::isdigit(static_cast<unsigned char>(version.front())));
  EXPECT_NE(version.find('.'), std::string::npos);
}

TEST(BackendTest, AvailableProvidersIncludesCPU) {
  const std::vector<std::string> providers = Backend::getAvailableProviders();
  ASSERT_FALSE(providers.empty());

  const bool hasCpu = std::any_of(providers.begin(), providers.end(), [](const std::string& p) {
    return p == "CPUExecutionProvider";
  });
  EXPECT_TRUE(hasCpu);
  EXPECT_TRUE(Backend::isProviderAvailable("CPUExecutionProvider"));
  EXPECT_TRUE(Backend::isProviderAvailable(ExecutionProvider::CPU));
}

TEST(BackendTest, NonExistentProviderReturnsFalse) {
  EXPECT_FALSE(Backend::isProviderAvailable("NonExistentProvider"));
  EXPECT_FALSE(Backend::isProviderAvailable(ExecutionProvider::UNKNOWN));
}

TEST(BackendTest, EnumParsingAndStringConversions) {
  EXPECT_EQ(parseExecutionProvider("CPUExecutionProvider"), ExecutionProvider::CPU);
  EXPECT_EQ(parseExecutionProvider("CoreMLExecutionProvider"), ExecutionProvider::COREML);
  EXPECT_EQ(parseExecutionProvider("CUDAExecutionProvider"), ExecutionProvider::CUDA);
  EXPECT_EQ(parseExecutionProvider("TensorrtExecutionProvider"), ExecutionProvider::TENSORRT);
  EXPECT_EQ(parseExecutionProvider("DmlExecutionProvider"), ExecutionProvider::DIRECTML);
  EXPECT_EQ(parseExecutionProvider("ROCMExecutionProvider"), ExecutionProvider::ROCM);
  EXPECT_EQ(parseExecutionProvider("InvalidProvider"), ExecutionProvider::UNKNOWN);

  EXPECT_EQ(executionProviderToString(ExecutionProvider::CPU), "CPUExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::COREML), "CoreMLExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::CUDA), "CUDAExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::TENSORRT), "TensorrtExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::DIRECTML), "DmlExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::ROCM), "ROCMExecutionProvider");
  EXPECT_EQ(executionProviderToString(ExecutionProvider::UNKNOWN), "Unknown");
}

TEST(BackendTest, TypedExecutionProvidersList) {
  const std::vector<ExecutionProvider> typed = Backend::getAvailableExecutionProviders();
  ASSERT_FALSE(typed.empty());

  const bool hasCpu = std::any_of(typed.begin(), typed.end(), [](ExecutionProvider ep) {
    return ep == ExecutionProvider::CPU;
  });
  EXPECT_TRUE(hasCpu);
}

TEST(BackendTest, EnvironmentInitializationAndHardwareReporting) {
  // Verify clean ORT environment construction and destruction
  EXPECT_NO_THROW({
    Ort::Env test_env(ORT_LOGGING_LEVEL_WARNING, "PyrethBackendTestEnv");
  });

  const ExecutionProvider best = Backend::getBestAvailableProvider();
  EXPECT_NE(best, ExecutionProvider::UNKNOWN);

  const bool hasHw = Backend::hasHardwareAcceleration();
  if (hasHw) {
    EXPECT_NE(best, ExecutionProvider::CPU);
  } else {
    EXPECT_EQ(best, ExecutionProvider::CPU);
  }

#if defined(__APPLE__) && defined(__aarch64__)
  // On Apple Silicon with official prebuilt ORT, CoreML is supported out of the box
  EXPECT_TRUE(Backend::isProviderAvailable(ExecutionProvider::COREML));
  EXPECT_EQ(best, ExecutionProvider::COREML);
  EXPECT_TRUE(hasHw);
#endif
}

} // namespace
