#include <onnxruntime_cxx_api.h>

#if defined(__APPLE__) && __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <pyreth/backend.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/ort_bridge.hpp>
#include <pyreth/runtime.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pyreth {

Result<Ort::Value> tensorToOrtValue(const Tensor& tensor, const Ort::MemoryInfo& memory_info) {
  if (!tensor.isValid()) {
    return Result<Ort::Value>::failure(StatusCode::INVALID_ARGUMENT,
                                       "Cannot wrap invalid or uninitialized Tensor in Ort::Value");
  }
  if (!tensor.isContiguous()) {
    return Result<Ort::Value>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Zero-copy Ort::Value bridge requires a contiguous Tensor; call toContiguous() first");
  }

  const ONNXTensorElementDataType ort_dtype = toOrtDataType(tensor.dtype());
  if (ort_dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
    return Result<Ort::Value>::failure(StatusCode::INVALID_ARGUMENT,
                                       "Unsupported Tensor DataType for Ort::Value bridge");
  }

  try {
    auto ort_val =
        Ort::Value::CreateTensor(memory_info, const_cast<void*>(tensor.data()), tensor.totalBytes(),
                                 tensor.shape().data(), tensor.shape().rank(), ort_dtype);
    return Result<Ort::Value>::success(std::move(ort_val));
  } catch (const Ort::Exception& e) {
    return Result<Ort::Value>::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Result<Ort::Value>::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

Result<Ort::Value> tensorToOrtValue(const Tensor& tensor) {
  static const Ort::MemoryInfo cpu_memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  return tensorToOrtValue(tensor, cpu_memory_info);
}

Result<Tensor> tensorFromOrtValue(Ort::Value&& ort_value) {
  if (!ort_value.HasValue() || !ort_value.IsTensor()) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Ort::Value is null or does not contain a tensor");
  }

  try {
    auto type_shape = ort_value.GetTensorTypeAndShapeInfo();
    const DataType dtype = fromOrtDataType(type_shape.GetElementType());
    Shape shape(type_shape.GetShape());
    void* raw_data = ort_value.GetTensorMutableRawData();

    auto owner = std::make_shared<Ort::Value>(std::move(ort_value));
    return Tensor::fromBuffer(raw_data, shape, dtype, {}, std::move(owner));
  } catch (const Ort::Exception& e) {
    return Result<Tensor>::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Result<Tensor>::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

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
  std::vector<const char*> input_name_ptrs;
  std::vector<const char*> output_name_ptrs;
  Ort::MemoryInfo cpu_memory_info{
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)};

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
    input_name_ptrs.clear();
    input_name_ptrs.reserve(in_count);
    model_info.inputs.clear();
    model_info.inputs.reserve(in_count);
    for (size_t i = 0; i < in_count; ++i) {
      auto name = session->GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
      input_name_ptrs.push_back(input_names.back().c_str());
      model_info.inputs.push_back(extractTensorInfo(name.get(), session->GetInputTypeInfo(i)));
    }

    // Extract output tensor metadata
    const size_t out_count = session->GetOutputCount();
    output_names.clear();
    output_names.reserve(out_count);
    output_name_ptrs.clear();
    output_name_ptrs.reserve(out_count);
    model_info.outputs.clear();
    model_info.outputs.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      auto name = session->GetOutputNameAllocated(i, allocator);
      output_names.emplace_back(name.get());
      output_name_ptrs.push_back(output_names.back().c_str());
      model_info.outputs.push_back(extractTensorInfo(name.get(), session->GetOutputTypeInfo(i)));
    }
  }
};

static bool tryResolveOutputShape(const TensorInfo& out_info,
                                  const std::vector<TensorInfo>& in_infos,
                                  std::span<const Tensor> inputs, Shape& resolved_shape) {
  std::vector<int64_t> dims(out_info.shape.size(), 0);
  for (size_t d = 0; d < out_info.shape.size(); ++d) {
    if (out_info.shape[d] >= 0) {
      dims[d] = out_info.shape[d];
      continue;
    }

    // Dynamic dimension: attempt to resolve via symbolic dimension name from inputs
    const std::string& sym = (d < out_info.symbolic_dims.size()) ? out_info.symbolic_dims[d] : "";
    bool found = false;
    if (!sym.empty()) {
      for (size_t in_idx = 0; in_idx < in_infos.size() && !found; ++in_idx) {
        const auto& in_syms = in_infos[in_idx].symbolic_dims;
        for (size_t in_d = 0; in_d < in_syms.size(); ++in_d) {
          if (in_syms[in_d] == sym) {
            dims[d] = inputs[in_idx].shape()[in_d];
            found = true;
            break;
          }
        }
      }
    }
    if (!found) {
      return false;
    }
  }
  resolved_shape = Shape(std::move(dims));
  return true;
}

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

Result<std::vector<Tensor>> InferenceSession::run(std::span<const Tensor> inputs) {
  if (!isValid()) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INVALID_ARGUMENT,
                                                "Session is uninitialized or invalid");
  }

  auto val_status = impl_->model_info.validateInputs(inputs);
  if (!val_status.isOk()) {
    return Result<std::vector<Tensor>>::failure(val_status.error());
  }

  // Pin input TensorStorage references across the entire ORT execution scope.
  // Declared before `ort_inputs` so non-owning Ort::Value wrappers are destroyed first.
  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(pinned_inputs.size());
  for (const auto& tensor : pinned_inputs) {
    auto ort_val_res = tensorToOrtValue(tensor, impl_->cpu_memory_info);
    if (!ort_val_res.isOk()) {
      return Result<std::vector<Tensor>>::failure(ort_val_res.error());
    }
    ort_inputs.push_back(std::move(ort_val_res).value());
  }

  try {
    Ort::RunOptions run_opts{nullptr};
    auto ort_outputs = impl_->session->Run(
        run_opts, impl_->input_name_ptrs.data(), ort_inputs.data(), ort_inputs.size(),
        impl_->output_name_ptrs.data(), impl_->output_name_ptrs.size());

    std::vector<Tensor> outputs;
    outputs.reserve(ort_outputs.size());
    for (auto& ort_out : ort_outputs) {
      auto tensor_res = tensorFromOrtValue(std::move(ort_out));
      if (!tensor_res.isOk()) {
        return Result<std::vector<Tensor>>::failure(tensor_res.error());
      }
      outputs.push_back(std::move(tensor_res).value());
    }
    return Result<std::vector<Tensor>>::success(std::move(outputs));
  } catch (const Ort::Exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

Result<std::vector<Tensor>> InferenceSession::run(std::span<const Tensor> inputs,
                                                  ArenaAllocator& output_arena) {
  if (!isValid()) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INVALID_ARGUMENT,
                                                "Session is uninitialized or invalid");
  }

  auto val_status = impl_->model_info.validateInputs(inputs);
  if (!val_status.isOk()) {
    return Result<std::vector<Tensor>>::failure(val_status.error());
  }

  // Pin input and pre-allocated output TensorStorage handles before Ort::Value wrappers.
  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());
  const size_t out_count = impl_->model_info.outputs.size();
  std::vector<Tensor> outputs(out_count);
  std::vector<bool> preallocated(out_count, false);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(pinned_inputs.size());
  for (const auto& tensor : pinned_inputs) {
    auto ort_val_res = tensorToOrtValue(tensor, impl_->cpu_memory_info);
    if (!ort_val_res.isOk()) {
      return Result<std::vector<Tensor>>::failure(ort_val_res.error());
    }
    ort_inputs.push_back(std::move(ort_val_res).value());
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs.reserve(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    Shape resolved_shape;
    if (tryResolveOutputShape(impl_->model_info.outputs[i], impl_->model_info.inputs, inputs,
                              resolved_shape)) {
      auto out_res =
          Tensor::create(resolved_shape, impl_->model_info.outputs[i].dtype, output_arena);
      if (!out_res.isOk()) {
        return Result<std::vector<Tensor>>::failure(out_res.error());
      }
      outputs[i] = std::move(out_res).value();
      auto ort_out_res = tensorToOrtValue(outputs[i], impl_->cpu_memory_info);
      if (!ort_out_res.isOk()) {
        return Result<std::vector<Tensor>>::failure(ort_out_res.error());
      }
      ort_outputs.push_back(std::move(ort_out_res).value());
      preallocated[i] = true;
    } else {
      ort_outputs.emplace_back(nullptr);
    }
  }

  try {
    Ort::RunOptions run_opts{nullptr};
    impl_->session->Run(run_opts, impl_->input_name_ptrs.data(), ort_inputs.data(),
                        ort_inputs.size(), impl_->output_name_ptrs.data(), ort_outputs.data(),
                        ort_outputs.size());

    for (size_t i = 0; i < out_count; ++i) {
      if (!preallocated[i]) {
        auto type_shape = ort_outputs[i].GetTensorTypeAndShapeInfo();
        Shape actual_shape(type_shape.GetShape());
        DataType dtype = fromOrtDataType(type_shape.GetElementType());
        auto out_res = Tensor::create(actual_shape, dtype, output_arena);
        if (!out_res.isOk()) {
          return Result<std::vector<Tensor>>::failure(out_res.error());
        }
        outputs[i] = std::move(out_res).value();
        if (outputs[i].totalBytes() > 0) {
          std::memcpy(outputs[i].data(), ort_outputs[i].GetTensorRawData(),
                      outputs[i].totalBytes());
        }
      }
    }
    return Result<std::vector<Tensor>>::success(std::move(outputs));
  } catch (const Ort::Exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

Result<std::vector<Tensor>> InferenceSession::run(std::span<const Tensor> inputs,
                                                  MemoryPool& output_pool) {
  if (!isValid()) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INVALID_ARGUMENT,
                                                "Session is uninitialized or invalid");
  }

  auto val_status = impl_->model_info.validateInputs(inputs);
  if (!val_status.isOk()) {
    return Result<std::vector<Tensor>>::failure(val_status.error());
  }

  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());
  const size_t out_count = impl_->model_info.outputs.size();
  std::vector<Tensor> outputs(out_count);
  std::vector<bool> preallocated(out_count, false);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(pinned_inputs.size());
  for (const auto& tensor : pinned_inputs) {
    auto ort_val_res = tensorToOrtValue(tensor, impl_->cpu_memory_info);
    if (!ort_val_res.isOk()) {
      return Result<std::vector<Tensor>>::failure(ort_val_res.error());
    }
    ort_inputs.push_back(std::move(ort_val_res).value());
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs.reserve(out_count);
  for (size_t i = 0; i < out_count; ++i) {
    Shape resolved_shape;
    if (tryResolveOutputShape(impl_->model_info.outputs[i], impl_->model_info.inputs, inputs,
                              resolved_shape)) {
      auto out_res =
          Tensor::create(resolved_shape, impl_->model_info.outputs[i].dtype, output_pool);
      if (!out_res.isOk()) {
        return Result<std::vector<Tensor>>::failure(out_res.error());
      }
      outputs[i] = std::move(out_res).value();
      auto ort_out_res = tensorToOrtValue(outputs[i], impl_->cpu_memory_info);
      if (!ort_out_res.isOk()) {
        return Result<std::vector<Tensor>>::failure(ort_out_res.error());
      }
      ort_outputs.push_back(std::move(ort_out_res).value());
      preallocated[i] = true;
    } else {
      ort_outputs.emplace_back(nullptr);
    }
  }

  try {
    Ort::RunOptions run_opts{nullptr};
    impl_->session->Run(run_opts, impl_->input_name_ptrs.data(), ort_inputs.data(),
                        ort_inputs.size(), impl_->output_name_ptrs.data(), ort_outputs.data(),
                        ort_outputs.size());

    for (size_t i = 0; i < out_count; ++i) {
      if (!preallocated[i]) {
        auto type_shape = ort_outputs[i].GetTensorTypeAndShapeInfo();
        Shape actual_shape(type_shape.GetShape());
        DataType dtype = fromOrtDataType(type_shape.GetElementType());
        auto out_res = Tensor::create(actual_shape, dtype, output_pool);
        if (!out_res.isOk()) {
          return Result<std::vector<Tensor>>::failure(out_res.error());
        }
        outputs[i] = std::move(out_res).value();
        if (outputs[i].totalBytes() > 0) {
          std::memcpy(outputs[i].data(), ort_outputs[i].GetTensorRawData(),
                      outputs[i].totalBytes());
        }
      }
    }
    return Result<std::vector<Tensor>>::success(std::move(outputs));
  } catch (const Ort::Exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

Status InferenceSession::run(std::span<const Tensor> inputs, std::span<Tensor> outputs) {
  if (!isValid()) {
    return Status::failure(StatusCode::INVALID_ARGUMENT, "Session is uninitialized or invalid");
  }

  auto val_status = impl_->model_info.validateInputs(inputs);
  if (!val_status.isOk()) {
    return val_status;
  }

  if (outputs.size() != impl_->model_info.outputs.size()) {
    return Status::failure(StatusCode::INVALID_ARGUMENT,
                           "Number of pre-allocated output tensors does not match model outputs");
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    auto out_status = impl_->model_info.outputs[i].validateTensor(outputs[i]);
    if (!out_status.isOk()) {
      return out_status;
    }
  }

  // Pin both input and output TensorStorage handles before creating Ort::Value wrappers.
  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());
  std::vector<Tensor> pinned_outputs(outputs.begin(), outputs.end());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.reserve(pinned_inputs.size());
  for (const auto& tensor : pinned_inputs) {
    auto ort_val_res = tensorToOrtValue(tensor, impl_->cpu_memory_info);
    if (!ort_val_res.isOk()) {
      return Status::failure(ort_val_res.error());
    }
    ort_inputs.push_back(std::move(ort_val_res).value());
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs.reserve(pinned_outputs.size());
  for (const auto& tensor : pinned_outputs) {
    auto ort_val_res = tensorToOrtValue(tensor, impl_->cpu_memory_info);
    if (!ort_val_res.isOk()) {
      return Status::failure(ort_val_res.error());
    }
    ort_outputs.push_back(std::move(ort_val_res).value());
  }

  try {
    Ort::RunOptions run_opts{nullptr};
    impl_->session->Run(run_opts, impl_->input_name_ptrs.data(), ort_inputs.data(),
                        ort_inputs.size(), impl_->output_name_ptrs.data(), ort_outputs.data(),
                        ort_outputs.size());
    return Status::ok();
  } catch (const Ort::Exception& e) {
    return Status::failure(StatusCode::BACKEND_ERROR, e.what());
  } catch (const std::exception& e) {
    return Status::failure(StatusCode::INTERNAL_ERROR, e.what());
  }
}

Result<std::vector<Tensor>> InferenceSession::run(
    const std::unordered_map<std::string, Tensor>& named_inputs) {
  if (!isValid()) {
    return Result<std::vector<Tensor>>::failure(StatusCode::INVALID_ARGUMENT,
                                                "Session is uninitialized or invalid");
  }
  if (named_inputs.size() != impl_->model_info.inputs.size()) {
    return Result<std::vector<Tensor>>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Number of named input tensors does not match model input count");
  }

  std::vector<Tensor> ordered_inputs(impl_->model_info.inputs.size());
  for (const auto& [name, tensor] : named_inputs) {
    auto idx_res = impl_->model_info.getInputIndex(name);
    if (!idx_res.isOk()) {
      return Result<std::vector<Tensor>>::failure(idx_res.error());
    }
    ordered_inputs[idx_res.value()] = tensor;
  }

  return run(std::span<const Tensor>(ordered_inputs));
}

Result<TaskHandle<Result<std::vector<Tensor>>>> InferenceSession::runAsync(
    ThreadPool& pool, std::span<const Tensor> inputs) {
  if (!isValid()) {
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_ARGUMENT, "Session is uninitialized or invalid");
  }

  // Capture shallow Tensor copies immediately on the calling thread so underlying
  // ArenaStorage / PooledArenaStorage stays pinned until the worker finishes.
  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());

  return pool.submit(
      [this, captured_inputs = std::move(pinned_inputs)]() -> Result<std::vector<Tensor>> {
        return this->run(std::span<const Tensor>(captured_inputs));
      });
}

Result<TaskHandle<Result<std::vector<Tensor>>>> InferenceSession::runAsync(
    ThreadPool& pool, std::span<const Tensor> inputs, MemoryPool& output_pool) {
  if (!isValid()) {
    return Result<TaskHandle<Result<std::vector<Tensor>>>>::failure(
        StatusCode::INVALID_ARGUMENT, "Session is uninitialized or invalid");
  }

  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());

  return pool.submit([this, &output_pool,
                      captured_inputs = std::move(pinned_inputs)]() -> Result<std::vector<Tensor>> {
    return this->run(std::span<const Tensor>(captured_inputs), output_pool);
  });
}

Result<TaskHandle<Status>> InferenceSession::runAsync(ThreadPool& pool,
                                                      std::span<const Tensor> inputs,
                                                      std::span<Tensor> outputs) {
  if (!isValid()) {
    return Result<TaskHandle<Status>>::failure(StatusCode::INVALID_ARGUMENT,
                                               "Session is uninitialized or invalid");
  }

  std::vector<Tensor> pinned_inputs(inputs.begin(), inputs.end());
  std::vector<Tensor> pinned_outputs(outputs.begin(), outputs.end());

  return pool.submit([this, captured_inputs = std::move(pinned_inputs),
                      captured_outputs = std::move(pinned_outputs)]() mutable -> Status {
    return this->run(std::span<const Tensor>(captured_inputs), std::span<Tensor>(captured_outputs));
  });
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
