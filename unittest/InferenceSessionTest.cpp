#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <pyreth/backend.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/runtime.hpp>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace pyreth;

// Minimal valid ONNX model protobuf (Identity node: x -> y, float32[1, 2], opset 17, IR v9)
constexpr uint8_t kValidModelData[] = {
    0x08, 0x09, 0x12, 0x0b, 0x70, 0x79, 0x72, 0x65, 0x74, 0x68, 0x5f, 0x74, 0x65, 0x73, 0x74, 0x3a,
    0x57, 0x0a, 0x1f, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x1a, 0x0d, 0x69, 0x64, 0x65, 0x6e, 0x74,
    0x69, 0x74, 0x79, 0x5f, 0x6e, 0x6f, 0x64, 0x65, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69,
    0x74, 0x79, 0x12, 0x0a, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x5a, 0x13,
    0x0a, 0x01, 0x78, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08, 0x0a, 0x02, 0x08, 0x01, 0x0a,
    0x02, 0x08, 0x02, 0x62, 0x13, 0x0a, 0x01, 0x79, 0x12, 0x0e, 0x0a, 0x0c, 0x08, 0x01, 0x12, 0x08,
    0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08, 0x02, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x11};

std::filesystem::path writeTempModelFile(std::span<const uint8_t> data,
                                         const std::string& filename = "test_model.onnx") {
  const auto temp_path = std::filesystem::temp_directory_path() / filename;
  std::ofstream out(temp_path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return temp_path;
}

// ---------------------------------------------------------------------------
// Runtime Singleton Tests
// ---------------------------------------------------------------------------

TEST(RuntimeTest, SingletonInstanceAndInitialization) {
  auto& rt = Runtime::instance();
  EXPECT_TRUE(rt.isInitialized());

  const auto status = rt.init(LogLevel::Info, "TestRuntime");
  EXPECT_TRUE(status.isOk());
  EXPECT_EQ(rt.getLogLevel(), LogLevel::Info);
  EXPECT_EQ(rt.getLogId(), "TestRuntime");

  EXPECT_NO_THROW({
    auto& env = rt.getEnv();
    (void)env;
  });
}

TEST(RuntimeTest, ShutdownAndLazyRestoration) {
  auto& rt = Runtime::instance();
  rt.shutdown();
  EXPECT_FALSE(rt.isInitialized());

  // Calling getEnv() lazily re-instantiates the environment
  EXPECT_NO_THROW({
    auto& env = rt.getEnv();
    (void)env;
  });
  EXPECT_TRUE(rt.isInitialized());
}

TEST(RuntimeTest, ConcurrentAccess) {
  auto& rt = Runtime::instance();
  constexpr size_t num_threads = 8;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&rt]() {
      for (int j = 0; j < 50; ++j) {
        auto& env = rt.getEnv();
        (void)env;
        EXPECT_TRUE(rt.isInitialized());
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }
}

TEST(RuntimeTest, LogLevelStringConversions) {
  EXPECT_EQ(logLevelToString(LogLevel::Verbose), "Verbose");
  EXPECT_EQ(logLevelToString(LogLevel::Info), "Info");
  EXPECT_EQ(logLevelToString(LogLevel::Warning), "Warning");
  EXPECT_EQ(logLevelToString(LogLevel::Error), "Error");
  EXPECT_EQ(logLevelToString(LogLevel::Fatal), "Fatal");
  EXPECT_EQ(logLevelToString(static_cast<LogLevel>(99)), "Unknown");
}

// ---------------------------------------------------------------------------
// InferenceSession Lifecycle & Error Handling Tests
// ---------------------------------------------------------------------------

TEST(InferenceSessionTest, DefaultConstructedIsInvalid) {
  InferenceSession session;
  EXPECT_FALSE(session.isValid());
  EXPECT_EQ(session.getInputCount(), 0);
  EXPECT_EQ(session.getOutputCount(), 0);
  EXPECT_TRUE(session.getInputNames().empty());
  EXPECT_TRUE(session.getOutputNames().empty());
  EXPECT_TRUE(session.getModelPath().empty());
  EXPECT_THROW((void)session.getOrtSession(), std::runtime_error);
}

TEST(InferenceSessionTest, NonExistentModelPathFails) {
  const auto res = InferenceSession::create("non_existent_file_pyreth_12345.onnx");
  EXPECT_FALSE(res.isOk());
  EXPECT_EQ(res.error().code(), StatusCode::MODEL_NOT_FOUND);
}

TEST(InferenceSessionTest, EmptyModelBytesBufferFails) {
  const auto res = InferenceSession::create(std::span<const uint8_t>{});
  EXPECT_FALSE(res.isOk());
  EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceSessionTest, CorruptedModelBytesFail) {
  const uint8_t garbage_bytes[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
  const auto res = InferenceSession::create(garbage_bytes);
  EXPECT_FALSE(res.isOk());
  EXPECT_EQ(res.error().code(), StatusCode::MODEL_LOAD_FAILED);
}

TEST(InferenceSessionTest, CorruptedModelFileFails) {
  const uint8_t garbage_bytes[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77};
  const auto temp_file = writeTempModelFile(garbage_bytes, "corrupt_model.onnx");

  const auto res = InferenceSession::create(temp_file.string());
  EXPECT_FALSE(res.isOk());
  EXPECT_EQ(res.error().code(), StatusCode::MODEL_LOAD_FAILED);

  std::filesystem::remove(temp_file);
}

// ---------------------------------------------------------------------------
// Valid Model Loading & Inspection Tests
// ---------------------------------------------------------------------------

TEST(InferenceSessionTest, ValidModelFromMemorySuccess) {
  const auto res = InferenceSession::create(kValidModelData);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  auto& session = res.value();
  EXPECT_TRUE(session.isValid());
  EXPECT_EQ(session.getInputCount(), 1);
  EXPECT_EQ(session.getOutputCount(), 1);

  const auto in_names = session.getInputNames();
  ASSERT_EQ(in_names.size(), 1);
  EXPECT_EQ(in_names[0], "x");

  const auto out_names = session.getOutputNames();
  ASSERT_EQ(out_names.size(), 1);
  EXPECT_EQ(out_names[0], "y");

  EXPECT_NO_THROW({
    auto& ort_sess = session.getOrtSession();
    EXPECT_EQ(ort_sess.GetInputCount(), 1);
    EXPECT_EQ(ort_sess.GetOutputCount(), 1);
  });
}

TEST(InferenceSessionTest, ValidModelFromFileSuccess) {
  const auto temp_file = writeTempModelFile(kValidModelData, "valid_model.onnx");

  const auto res = InferenceSession::create(temp_file.string());
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& session = res.value();
  EXPECT_TRUE(session.isValid());
  EXPECT_EQ(session.getModelPath(), temp_file.string());
  EXPECT_EQ(session.getInputCount(), 1);
  EXPECT_EQ(session.getOutputCount(), 1);

  std::filesystem::remove(temp_file);
}

// ---------------------------------------------------------------------------
// Session Options & Configuration Verification Tests
// ---------------------------------------------------------------------------

TEST(InferenceSessionTest, CustomSessionOptionsConfiguration) {
  SessionConfig config;
  config.setIntraOpNumThreads(2)
      .setInterOpNumThreads(2)
      .setGraphOptimizationLevel(OptimizationLevel::Basic)
      .setExecutionMode(SessionExecutionMode::Sequential)
      .setEnableCpuMemArena(false)
      .setEnableMemPattern(false)
      .setLogSeverityLevel(LogLevel::Error)
      .setLogId("CustomSessionLogId");

  const auto res = InferenceSession::create(kValidModelData, config);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& session = res.value();
  EXPECT_TRUE(session.isValid());

  const auto& applied_cfg = session.getConfig();
  EXPECT_EQ(applied_cfg.intra_op_num_threads, 2);
  EXPECT_EQ(applied_cfg.inter_op_num_threads, 2);
  EXPECT_EQ(applied_cfg.graph_optimization_level, OptimizationLevel::Basic);
  EXPECT_EQ(applied_cfg.execution_mode, SessionExecutionMode::Sequential);
  EXPECT_FALSE(applied_cfg.enable_cpu_mem_arena);
  EXPECT_FALSE(applied_cfg.enable_mem_pattern);
  EXPECT_EQ(applied_cfg.log_severity_level, LogLevel::Error);
  EXPECT_EQ(applied_cfg.log_id, "CustomSessionLogId");
}

TEST(InferenceSessionTest, EnumStringConversions) {
  EXPECT_EQ(optimizationLevelToString(OptimizationLevel::Disable), "Disable");
  EXPECT_EQ(optimizationLevelToString(OptimizationLevel::Basic), "Basic");
  EXPECT_EQ(optimizationLevelToString(OptimizationLevel::Extended), "Extended");
  EXPECT_EQ(optimizationLevelToString(OptimizationLevel::All), "All");
  EXPECT_EQ(optimizationLevelToString(static_cast<OptimizationLevel>(99)), "Unknown");

  EXPECT_EQ(sessionExecutionModeToString(SessionExecutionMode::Sequential), "Sequential");
  EXPECT_EQ(sessionExecutionModeToString(SessionExecutionMode::Parallel), "Parallel");
  EXPECT_EQ(sessionExecutionModeToString(static_cast<SessionExecutionMode>(99)), "Unknown");
}

// ---------------------------------------------------------------------------
// Execution Provider Registration & Fallback Tests
// ---------------------------------------------------------------------------

TEST(InferenceSessionTest, CpuProviderActiveByDefault) {
  SessionConfig config;
  config.setProviders({ExecutionProvider::CPU});

  const auto res = InferenceSession::create(kValidModelData, config);
  ASSERT_TRUE(res.isOk());

  const auto& providers = res.value().getActiveProviders();
  EXPECT_FALSE(providers.empty());
  EXPECT_EQ(providers[0], ExecutionProvider::CPU);
}

TEST(InferenceSessionTest, UnavailableProviderWithFallbackEnabledFallsBackToCpu) {
  SessionConfig config;
  // Request an unavailable execution provider (e.g. UNKNOWN)
  config.setProviders({ExecutionProvider::UNKNOWN}).setAllowProviderFallback(true);

  const auto res = InferenceSession::create(kValidModelData, config);
  ASSERT_TRUE(res.isOk());

  const auto& active = res.value().getActiveProviders();
  ASSERT_FALSE(active.empty());
  EXPECT_EQ(active[0], ExecutionProvider::CPU);
}

TEST(InferenceSessionTest, UnavailableProviderWithFallbackDisabledReturnsError) {
  SessionConfig config;
  config.setProviders({ExecutionProvider::UNKNOWN}).setAllowProviderFallback(false);

  const auto res = InferenceSession::create(kValidModelData, config);
  EXPECT_FALSE(res.isOk());
  EXPECT_EQ(res.error().code(), StatusCode::BACKEND_UNAVAILABLE);
}

#if defined(__APPLE__) && defined(__aarch64__)
TEST(InferenceSessionTest, CoreMLProviderRegistrationOnAppleSilicon) {
  if (Backend::isProviderAvailable(ExecutionProvider::COREML)) {
    SessionConfig config;
    config.setProviders({ExecutionProvider::COREML, ExecutionProvider::CPU});

    const auto res = InferenceSession::create(kValidModelData, config);
    ASSERT_TRUE(res.isOk()) << res.error().message();

    const auto& active = res.value().getActiveProviders();
    const bool has_coreml =
        std::find(active.begin(), active.end(), ExecutionProvider::COREML) != active.end();
    EXPECT_TRUE(has_coreml);
  }
}
#endif

// ---------------------------------------------------------------------------
// Move Semantics Tests
// ---------------------------------------------------------------------------

TEST(InferenceSessionTest, MoveSemanticsTransferOwnership) {
  auto res = InferenceSession::create(kValidModelData);
  ASSERT_TRUE(res.isOk());

  InferenceSession s1 = std::move(res.value());
  EXPECT_TRUE(s1.isValid());
  EXPECT_EQ(s1.getInputCount(), 1);

  // Move constructor
  InferenceSession s2 = std::move(s1);
  EXPECT_TRUE(s2.isValid());
  EXPECT_FALSE(s1.isValid());
  EXPECT_EQ(s2.getInputCount(), 1);

  // Move assignment
  InferenceSession s3;
  EXPECT_FALSE(s3.isValid());
  s3 = std::move(s2);
  EXPECT_TRUE(s3.isValid());
  EXPECT_FALSE(s2.isValid());
  EXPECT_EQ(s3.getInputCount(), 1);
}

}  // namespace
