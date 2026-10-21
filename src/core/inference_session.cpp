#include <onnxruntime_cxx_api.h>

#if defined(__APPLE__) && __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#endif

#include <algorithm>
#include <filesystem>
#include <memory>
#include <pyreth/backend.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/runtime.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pyreth {

static OrtLoggingLevel toOrtLoggingLevel(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::Verbose:
      return ORT_LOGGING_LEVEL_VERBOSE;
    case LogLevel::Info:
      return ORT_LOGGING_LEVEL_INFO;
    case LogLevel::Warning:
      return ORT_LOGGING_LEVEL_WARNING;
    case LogLevel::Error:
      return ORT_LOGGING_LEVEL_ERROR;
    case LogLevel::Fatal:
      return ORT_LOGGING_LEVEL_FATAL;
  }
  return ORT_LOGGING_LEVEL_WARNING;
}

static Result<Ort::SessionOptions> configureSessionOptions(
    const SessionConfig& config, std::vector<ExecutionProvider>& active_providers) {
  Ort::SessionOptions opts;

  opts.SetIntraOpNumThreads(config.intra_op_num_threads);
  opts.SetInterOpNumThreads(config.inter_op_num_threads);

  switch (config.graph_optimization_level) {
    case OptimizationLevel::Disable:
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
      break;
    case OptimizationLevel::Basic:
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
      break;
    case OptimizationLevel::Extended:
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      break;
    case OptimizationLevel::All:
      opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      break;
  }

  switch (config.execution_mode) {
    case SessionExecutionMode::Sequential:
      opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      break;
    case SessionExecutionMode::Parallel:
      opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
      break;
  }

  if (config.enable_cpu_mem_arena) {
    opts.EnableCpuMemArena();
  } else {
    opts.DisableCpuMemArena();
  }

  if (config.enable_mem_pattern) {
    opts.EnableMemPattern();
  } else {
    opts.DisableMemPattern();
  }

  opts.SetLogSeverityLevel(static_cast<int>(toOrtLoggingLevel(config.log_severity_level)));
  if (!config.log_id.empty()) {
    opts.SetLogId(config.log_id.c_str());
  }

  if (config.enable_profiling) {
#if defined(_WIN32)
    std::wstring wprefix(config.profile_file_prefix.begin(), config.profile_file_prefix.end());
    opts.EnableProfiling(wprefix.c_str());
#else
    opts.EnableProfiling(config.profile_file_prefix.c_str());
#endif
  }

  active_providers.clear();
  for (const auto provider : config.providers) {
    if (provider == ExecutionProvider::CPU) {
      if (std::find(active_providers.begin(), active_providers.end(), ExecutionProvider::CPU) ==
          active_providers.end()) {
        active_providers.push_back(ExecutionProvider::CPU);
      }
      continue;
    }

    if (!Backend::isProviderAvailable(provider)) {
      if (!config.allow_provider_fallback) {
        return Result<Ort::SessionOptions>::failure(
            StatusCode::BACKEND_UNAVAILABLE, "Requested execution provider is not available: " +
                                                 std::string(executionProviderToString(provider)));
      }
      continue;
    }

    bool registered = false;
    std::string err_msg;

    switch (provider) {
      case ExecutionProvider::COREML: {
#if defined(__APPLE__) && __has_include(<coreml_provider_factory.h>)
        uint32_t flags = 0;
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CoreML(
            static_cast<OrtSessionOptions*>(opts), flags);
        if (status == nullptr) {
          registered = true;
        } else {
          err_msg = Ort::GetApi().GetErrorMessage(status);
          Ort::GetApi().ReleaseStatus(status);
        }
#endif
        break;
      }
      case ExecutionProvider::CUDA: {
        try {
          OrtCUDAProviderOptions cuda_options;
          opts.AppendExecutionProvider_CUDA(cuda_options);
          registered = true;
        } catch (const std::exception& e) {
          err_msg = e.what();
        }
        break;
      }
      case ExecutionProvider::TENSORRT: {
        try {
          OrtTensorRTProviderOptions trt_options;
          opts.AppendExecutionProvider_TensorRT(trt_options);
          registered = true;
        } catch (const std::exception& e) {
          err_msg = e.what();
        }
        break;
      }
      case ExecutionProvider::ROCM: {
        try {
          OrtROCMProviderOptions rocm_options;
          opts.AppendExecutionProvider_ROCM(rocm_options);
          registered = true;
        } catch (const std::exception& e) {
          err_msg = e.what();
        }
        break;
      }
      default:
        break;
    }

    if (registered) {
      active_providers.push_back(provider);
    } else {
      if (!config.allow_provider_fallback) {
        return Result<Ort::SessionOptions>::failure(
            StatusCode::BACKEND_ERROR, "Failed to append execution provider " +
                                           std::string(executionProviderToString(provider)) +
                                           (!err_msg.empty() ? ": " + err_msg : ""));
      }
    }
  }

  if (active_providers.empty()) {
    active_providers.push_back(ExecutionProvider::CPU);
  }

  return Result<Ort::SessionOptions>::success(std::move(opts));
}

struct InferenceSession::Impl {
  std::unique_ptr<Ort::Session> session;
  SessionConfig config;
  std::vector<ExecutionProvider> active_providers;
  std::string model_path;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  void populateIoNames() {
    if (!session) {
      return;
    }
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t in_count = session->GetInputCount();
    input_names.clear();
    input_names.reserve(in_count);
    for (size_t i = 0; i < in_count; ++i) {
      auto name = session->GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
    }

    const size_t out_count = session->GetOutputCount();
    output_names.clear();
    output_names.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
    }
  }
};

InferenceSession::InferenceSession() noexcept : impl_(nullptr) {}

InferenceSession::InferenceSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

InferenceSession::InferenceSession(InferenceSession&& other) noexcept = default;

InferenceSession& InferenceSession::operator=(InferenceSession&& other) noexcept = default;

InferenceSession::~InferenceSession() = default;

Result<InferenceSession> InferenceSession::create(std::string_view model_path,
                                                  const SessionConfig& config) {
  std::filesystem::path p(model_path);
  std::error_code ec;
  if (!std::filesystem::exists(p, ec) || ec) {
    return Result<InferenceSession>::failure(StatusCode::MODEL_NOT_FOUND,
                                             "Model file not found: " + std::string(model_path));
  }

  std::vector<ExecutionProvider> active_providers;
  auto opts_res = configureSessionOptions(config, active_providers);
  if (!opts_res.isOk()) {
    return Result<InferenceSession>::failure(opts_res.error());
  }

  try {
    auto session =
        std::make_unique<Ort::Session>(Runtime::instance().getEnv(), p.c_str(), opts_res.value());

    auto impl = std::make_unique<Impl>();
    impl->session = std::move(session);
    impl->config = config;
    impl->active_providers = std::move(active_providers);
    impl->model_path = std::string(model_path);
    impl->populateIoNames();

    return Result<InferenceSession>::success(InferenceSession(std::move(impl)));
  } catch (const Ort::Exception& e) {
    return Result<InferenceSession>::failure(StatusCode::MODEL_LOAD_FAILED, e.what());
  } catch (const std::exception& e) {
    return Result<InferenceSession>::failure(StatusCode::MODEL_LOAD_FAILED, e.what());
  }
}

Result<InferenceSession> InferenceSession::create(std::span<const uint8_t> model_bytes,
                                                  const SessionConfig& config) {
  if (model_bytes.empty()) {
    return Result<InferenceSession>::failure(StatusCode::INVALID_ARGUMENT,
                                             "Model data buffer cannot be empty");
  }

  std::vector<ExecutionProvider> active_providers;
  auto opts_res = configureSessionOptions(config, active_providers);
  if (!opts_res.isOk()) {
    return Result<InferenceSession>::failure(opts_res.error());
  }

  try {
    auto session = std::make_unique<Ort::Session>(Runtime::instance().getEnv(), model_bytes.data(),
                                                  model_bytes.size(), opts_res.value());

    auto impl = std::make_unique<Impl>();
    impl->session = std::move(session);
    impl->config = config;
    impl->active_providers = std::move(active_providers);
    impl->populateIoNames();

    return Result<InferenceSession>::success(InferenceSession(std::move(impl)));
  } catch (const Ort::Exception& e) {
    return Result<InferenceSession>::failure(StatusCode::MODEL_LOAD_FAILED, e.what());
  } catch (const std::exception& e) {
    return Result<InferenceSession>::failure(StatusCode::MODEL_LOAD_FAILED, e.what());
  }
}

bool InferenceSession::isValid() const noexcept {
  return impl_ != nullptr && impl_->session != nullptr;
}

const SessionConfig& InferenceSession::getConfig() const noexcept {
  static const SessionConfig empty_config{};
  return impl_ ? impl_->config : empty_config;
}

const std::vector<ExecutionProvider>& InferenceSession::getActiveProviders() const noexcept {
  static const std::vector<ExecutionProvider> empty_providers{};
  return impl_ ? impl_->active_providers : empty_providers;
}

std::string_view InferenceSession::getModelPath() const noexcept {
  return impl_ ? std::string_view(impl_->model_path) : std::string_view{};
}

size_t InferenceSession::getInputCount() const {
  return impl_ ? impl_->input_names.size() : 0;
}

size_t InferenceSession::getOutputCount() const {
  return impl_ ? impl_->output_names.size() : 0;
}

std::vector<std::string> InferenceSession::getInputNames() const {
  return impl_ ? impl_->input_names : std::vector<std::string>{};
}

std::vector<std::string> InferenceSession::getOutputNames() const {
  return impl_ ? impl_->output_names : std::vector<std::string>{};
}

Ort::Session& InferenceSession::getOrtSession() {
  if (!isValid()) {
    throw std::runtime_error(
        "Attempted to access Ort::Session on an invalid or uninitialized InferenceSession");
  }
  return *(impl_->session);
}

const Ort::Session& InferenceSession::getOrtSession() const {
  if (!isValid()) {
    throw std::runtime_error(
        "Attempted to access Ort::Session on an invalid or uninitialized InferenceSession");
  }
  return *(impl_->session);
}

}  // namespace pyreth
