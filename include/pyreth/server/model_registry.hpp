#pragma once

#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/dynamic_batcher.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/result.hpp>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pyreth::server {

/**
 * @struct ModelEntry
 * @brief Metadata and execution handles for a registered model.
 */
struct PYRETH_API ModelEntry {
  std::string name;
  std::string version{"1"};
  std::string state{"ready"};
  std::shared_ptr<InferenceSession> session;
  std::shared_ptr<DynamicBatcher> batcher;
};

/**
 * @class ModelRegistry
 * @brief Thread-safe repository managing active models, InferenceSessions, and DynamicBatchers.
 */
class PYRETH_API ModelRegistry {
public:
  ModelRegistry() = default;
  ~ModelRegistry() = default;

  ModelRegistry(const ModelRegistry&) = delete;
  ModelRegistry& operator=(const ModelRegistry&) = delete;

  ModelRegistry(ModelRegistry&& other) noexcept;
  ModelRegistry& operator=(ModelRegistry&& other) noexcept;

  [[nodiscard]] Result<void> registerModel(std::string name,
                                           std::shared_ptr<InferenceSession> session,
                                           std::string version = "1");

  [[nodiscard]] Result<void> registerBatcher(std::string name,
                                             std::shared_ptr<DynamicBatcher> batcher);

  [[nodiscard]] Result<void> registerBatcher(std::string name, DynamicBatcher batcher) {
    return registerBatcher(std::move(name), std::make_shared<DynamicBatcher>(std::move(batcher)));
  }

  [[nodiscard]] Result<void> unregisterBatcher(std::string_view name);

  [[nodiscard]] Result<void> unregisterModel(std::string_view name);

  [[nodiscard]] bool hasModel(std::string_view name) const;

  [[nodiscard]] Result<std::shared_ptr<InferenceSession>> getModel(std::string_view name) const;

  [[nodiscard]] Result<std::shared_ptr<DynamicBatcher>> getBatcher(std::string_view name) const;

  [[nodiscard]] Result<ModelEntry> getModelEntry(std::string_view name) const;

  [[nodiscard]] std::vector<ModelEntry> listModels() const;

  [[nodiscard]] size_t modelCount() const;

  void clear();

private:
  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, ModelEntry> models_;
};

}  // namespace pyreth::server
