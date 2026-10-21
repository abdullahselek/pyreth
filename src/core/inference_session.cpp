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

static DataType fromOrtDataType(ONNXTensorElementDataType ort_type) noexcept {
  switch (ort_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DataType::Float32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DataType::Float16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return DataType::BFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return DataType::Float64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return DataType::Int8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DataType::UInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return DataType::Int16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return DataType::UInt16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DataType::Int32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return DataType::UInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DataType::Int64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return DataType::UInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DataType::Bool;
    default:
      return DataType::Float32;
  }
}

static TensorInfo extractTensorInfo(const char* name, const Ort::TypeInfo& type_info) {
  TensorInfo info;
  info.name = name ? name : "";
  if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    info.dtype = fromOrtDataType(tensor_info.GetElementType());
    info.shape = tensor_info.GetShape();
    const size_t dim_count = tensor_info.GetDimensionsCount();
    info.symbolic_dims.resize(dim_count);
    if (dim_count > 0) {
      std::vector<const char*> sym_ptrs(dim_count, nullptr);
      tensor_info.GetSymbolicDimensions(sym_ptrs.data(), dim_count);
      for (size_t d = 0; d < dim_count; ++d) {
        if (sym_ptrs[d] != nullptr) {
          info.symbolic_dims[d] = sym_ptrs[d];
        }
      }
    }
  }
  return info;
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
  ModelInfo model_info;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  void populateModelInfo() {
    if (!session) {
      return;
    }
    Ort::AllocatorWithDefaultOptions allocator;

    // Extract model-level metadata
    try {
      Ort::ModelMetadata meta = session->GetModelMetadata();
      if (auto prod = meta.GetProducerNameAllocated(allocator)) {
        model_info.producer_name = prod.get();
      }
      if (auto gname = meta.GetGraphNameAllocated(allocator)) {
        model_info.graph_name = gname.get();
      }
      if (auto dom = meta.GetDomainAllocated(allocator)) {
        model_info.domain = dom.get();
      }
      if (auto desc = meta.GetDescriptionAllocated(allocator)) {
        model_info.description = desc.get();
      }
      if (auto gdesc = meta.GetGraphDescriptionAllocated(allocator)) {
        model_info.graph_description = gdesc.get();
      }
      model_info.version = meta.GetVersion();

      auto keys = meta.GetCustomMetadataMapKeysAllocated(allocator);
      for (const auto& key_ptr : keys) {
        if (key_ptr) {
          auto val_ptr = meta.LookupCustomMetadataMapAllocated(key_ptr.get(), allocator);
          model_info.custom_metadata[key_ptr.get()] = val_ptr ? val_ptr.get() : "";
        }
      }
    } catch (...) {
      // Metadata fields are optional in ONNX models
    }

    // Extract input tensor metadata
    const size_t in_count = session->GetInputCount();
    input_names.clear();
    input_names.reserve(in_count);
    model_info.inputs.clear();
    model_info.inputs.reserve(in_count);
    for (size_t i = 0; i < in_count; ++i) {
      auto name = session->GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
      model_info.inputs.push_back(extractTensorInfo(name.get(), session->GetInputTypeInfo(i)));
    }

    // Extract output tensor metadata
    const size_t out_count = session->GetOutputCount();
    output_names.clear();
    output_names.reserve(out_count);
    model_info.outputs.clear();
    model_info.outputs.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
      model_info.outputs.push_back(extractTensorInfo(name.get(), session->GetOutputTypeInfo(i)));
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
    impl->model_info.model_path = std::string(model_path);
    impl->populateModelInfo();

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
    impl->populateModelInfo();

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
  return impl_ ? std::string_view(impl_->model_info.model_path) : std::string_view{};
}

size_t InferenceSession::getInputCount() const noexcept {
  return impl_ ? impl_->model_info.inputs.size() : 0;
}

size_t InferenceSession::getOutputCount() const noexcept {
  return impl_ ? impl_->model_info.outputs.size() : 0;
}

std::vector<std::string> InferenceSession::getInputNames() const {
  return impl_ ? impl_->input_names : std::vector<std::string>{};
}

std::vector<std::string> InferenceSession::getOutputNames() const {
  return impl_ ? impl_->output_names : std::vector<std::string>{};
}

const ModelInfo& InferenceSession::getModelInfo() const noexcept {
  static const ModelInfo empty_info{};
  return impl_ ? impl_->model_info : empty_info;
}

const std::vector<TensorInfo>& InferenceSession::getInputs() const noexcept {
  static const std::vector<TensorInfo> empty_inputs{};
  return impl_ ? impl_->model_info.inputs : empty_inputs;
}

const std::vector<TensorInfo>& InferenceSession::getOutputs() const noexcept {
  static const std::vector<TensorInfo> empty_outputs{};
  return impl_ ? impl_->model_info.outputs : empty_outputs;
}

Result<TensorInfo> InferenceSession::getInputInfo(size_t index) const noexcept {
  if (!impl_ || index >= impl_->model_info.inputs.size()) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT, "Input index out of range");
  }
  return Result<TensorInfo>::success(impl_->model_info.inputs[index]);
}

Result<TensorInfo> InferenceSession::getInputInfo(std::string_view name) const noexcept {
  if (!impl_) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT,
                                       "Session is uninitialized or invalid");
  }
  const auto* info = impl_->model_info.findInput(name);
  if (!info) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT, "Unknown model input name");
  }
  return Result<TensorInfo>::success(*info);
}

Result<TensorInfo> InferenceSession::getOutputInfo(size_t index) const noexcept {
  if (!impl_ || index >= impl_->model_info.outputs.size()) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT, "Output index out of range");
  }
  return Result<TensorInfo>::success(impl_->model_info.outputs[index]);
}

Result<TensorInfo> InferenceSession::getOutputInfo(std::string_view name) const noexcept {
  if (!impl_) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT,
                                       "Session is uninitialized or invalid");
  }
  const auto* info = impl_->model_info.findOutput(name);
  if (!info) {
    return Result<TensorInfo>::failure(StatusCode::INVALID_ARGUMENT, "Unknown model output name");
  }
  return Result<TensorInfo>::success(*info);
}

Status InferenceSession::validateInputs(std::span<const Tensor> inputs) const noexcept {
  if (!isValid()) {
    return Status::failure(StatusCode::INVALID_ARGUMENT, "Session is uninitialized or invalid");
  }
  return impl_->model_info.validateInputs(inputs);
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
