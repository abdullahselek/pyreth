#include <onnxruntime_cxx_api.h>

#include <mutex>
#include <pyreth/runtime.hpp>

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

struct Runtime::Impl {
  mutable std::mutex mutex;
  std::unique_ptr<Ort::Env> env;
  LogLevel log_level{LogLevel::Warning};
  std::string log_id{"PyrethRuntime"};

  Impl() {
    try {
      env = std::make_unique<Ort::Env>(toOrtLoggingLevel(log_level), log_id.c_str());
    } catch (...) {}
  }

  void ensureInitializedLocked() {
    if (!env) {
      env = std::make_unique<Ort::Env>(toOrtLoggingLevel(log_level), log_id.c_str());
    }
  }
};

Runtime& Runtime::instance() {
  static Runtime rt;
  return rt;
}

Runtime::Runtime() : impl_(std::make_unique<Impl>()) {}

Runtime::~Runtime() = default;

Status Runtime::init(LogLevel log_level, std::string_view log_id) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  try {
    impl_->log_level = log_level;
    impl_->log_id = std::string(log_id);
    impl_->env = std::make_unique<Ort::Env>(toOrtLoggingLevel(log_level), impl_->log_id.c_str());
    return Status::ok();
  } catch (const Ort::Exception&) {
    return Status::failure(StatusCode::BACKEND_ERROR,
                           "Failed to initialize ONNX Runtime environment");
  } catch (const std::exception&) {
    return Status::failure(StatusCode::INTERNAL_ERROR,
                           "Internal error initializing ONNX Runtime environment");
  }
}

bool Runtime::isInitialized() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->env != nullptr;
}

LogLevel Runtime::getLogLevel() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->log_level;
}

std::string Runtime::getLogId() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->log_id;
}

Ort::Env& Runtime::getEnv() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->ensureInitializedLocked();
  return *(impl_->env);
}

const Ort::Env& Runtime::getEnv() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->ensureInitializedLocked();
  return *(impl_->env);
}

void Runtime::shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->env.reset();
}

}  // namespace pyreth
