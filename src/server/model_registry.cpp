#include <mutex>
#include <pyreth/server/model_registry.hpp>
#include <shared_mutex>
#include <utility>

namespace pyreth::server {

ModelRegistry::ModelRegistry(ModelRegistry&& other) noexcept {
  std::unique_lock<std::shared_mutex> lock(other.mutex_);
  models_ = std::move(other.models_);
}

ModelRegistry& ModelRegistry::operator=(ModelRegistry&& other) noexcept {
  if (this != &other) {
    std::unique_lock<std::shared_mutex> lock_this(mutex_, std::defer_lock);
    std::unique_lock<std::shared_mutex> lock_other(other.mutex_, std::defer_lock);
    std::lock(lock_this, lock_other);
    models_ = std::move(other.models_);
  }
  return *this;
}

Result<void> ModelRegistry::registerModel(std::string name,
                                          std::shared_ptr<InferenceSession> session,
                                          std::string version) {
  if (name.empty()) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT, "Model name cannot be empty");
  }
  if (!session || !session->isValid()) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                 "Cannot register null or invalid InferenceSession");
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(name);
  if (it != models_.end() && it->second.session != nullptr) {
    return Result<void>::failure(StatusCode::MODEL_ALREADY_EXISTS, "Model already exists");
  }

  if (it != models_.end()) {
    it->second.session = std::move(session);
    if (!version.empty()) {
      it->second.version = std::move(version);
    }
    it->second.state = "ready";
  } else {
    ModelEntry entry;
    entry.name = name;
    entry.version = version.empty() ? "1" : std::move(version);
    entry.state = "ready";
    entry.session = std::move(session);
    models_.emplace(std::move(name), std::move(entry));
  }

  return Result<void>::success();
}

Result<void> ModelRegistry::registerBatcher(std::string name,
                                            std::shared_ptr<DynamicBatcher> batcher) {
  if (name.empty()) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT, "Model name cannot be empty");
  }
  if (!batcher) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                 "Cannot register null DynamicBatcher");
  }

  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(name);
  if (it != models_.end()) {
    it->second.batcher = std::move(batcher);
  } else {
    ModelEntry entry;
    entry.name = name;
    entry.version = "1";
    entry.state = "ready";
    entry.batcher = std::move(batcher);
    models_.emplace(std::move(name), std::move(entry));
  }

  return Result<void>::success();
}

Result<void> ModelRegistry::unregisterBatcher(std::string_view name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(std::string(name));
  if (it == models_.end() || !it->second.batcher) {
    return Result<void>::failure(StatusCode::MODEL_NOT_FOUND, "Batcher not found");
  }
  it->second.batcher = nullptr;
  return Result<void>::success();
}

Result<void> ModelRegistry::unregisterModel(std::string_view name) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(std::string(name));
  if (it == models_.end()) {
    return Result<void>::failure(StatusCode::MODEL_NOT_FOUND, "Model not found");
  }
  models_.erase(it);
  return Result<void>::success();
}

bool ModelRegistry::hasModel(std::string_view name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return models_.find(std::string(name)) != models_.end();
}

Result<std::shared_ptr<InferenceSession>> ModelRegistry::getModel(std::string_view name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(std::string(name));
  if (it == models_.end() || !it->second.session) {
    return Result<std::shared_ptr<InferenceSession>>::failure(StatusCode::MODEL_NOT_FOUND,
                                                              "Model not found");
  }
  return Result<std::shared_ptr<InferenceSession>>::success(it->second.session);
}

Result<std::shared_ptr<DynamicBatcher>> ModelRegistry::getBatcher(std::string_view name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(std::string(name));
  if (it == models_.end() || !it->second.batcher) {
    return Result<std::shared_ptr<DynamicBatcher>>::failure(StatusCode::MODEL_NOT_FOUND,
                                                            "Batcher not found");
  }
  return Result<std::shared_ptr<DynamicBatcher>>::success(it->second.batcher);
}

Result<ModelEntry> ModelRegistry::getModelEntry(std::string_view name) const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  auto it = models_.find(std::string(name));
  if (it == models_.end()) {
    return Result<ModelEntry>::failure(StatusCode::MODEL_NOT_FOUND, "Model not found");
  }
  return Result<ModelEntry>::success(it->second);
}

std::vector<ModelEntry> ModelRegistry::listModels() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  std::vector<ModelEntry> result;
  result.reserve(models_.size());
  for (const auto& [_, entry] : models_) {
    result.push_back(entry);
  }
  return result;
}

size_t ModelRegistry::modelCount() const {
  std::shared_lock<std::shared_mutex> lock(mutex_);
  return models_.size();
}

void ModelRegistry::clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  models_.clear();
}

}  // namespace pyreth::server
