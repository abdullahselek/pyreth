#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <pyreth/backend.hpp>

namespace pyreth {

ExecutionProvider parseExecutionProvider(std::string_view name) noexcept {
  if (name == "CPUExecutionProvider") {
    return ExecutionProvider::CPU;
  }
  if (name == "CoreMLExecutionProvider") {
    return ExecutionProvider::COREML;
  }
  if (name == "CUDAExecutionProvider") {
    return ExecutionProvider::CUDA;
  }
  if (name == "TensorrtExecutionProvider") {
    return ExecutionProvider::TENSORRT;
  }
  if (name == "DmlExecutionProvider") {
    return ExecutionProvider::DIRECTML;
  }
  if (name == "ROCMExecutionProvider") {
    return ExecutionProvider::ROCM;
  }
  return ExecutionProvider::UNKNOWN;
}

std::string Backend::getRuntimeVersion() {
  return Ort::GetVersionString();
}

std::vector<std::string> Backend::getAvailableProviders() {
  return Ort::GetAvailableProviders();
}

std::vector<ExecutionProvider> Backend::getAvailableExecutionProviders() {
  const auto providers = getAvailableProviders();
  std::vector<ExecutionProvider> result;
  result.reserve(providers.size());
  for (const auto& provider : providers) {
    result.push_back(parseExecutionProvider(provider));
  }
  return result;
}

bool Backend::isProviderAvailable(std::string_view provider_name) {
  const auto providers = getAvailableProviders();
  return std::any_of(providers.begin(), providers.end(),
                     [provider_name](const std::string& p) { return p == provider_name; });
}

bool Backend::isProviderAvailable(ExecutionProvider provider) {
  return isProviderAvailable(executionProviderToString(provider));
}

ExecutionProvider Backend::getBestAvailableProvider() {
  // Ordered by preference
  constexpr ExecutionProvider priority[] = {
      ExecutionProvider::TENSORRT, ExecutionProvider::CUDA, ExecutionProvider::COREML,
      ExecutionProvider::DIRECTML, ExecutionProvider::ROCM,
  };

  for (const auto ep : priority) {
    if (isProviderAvailable(ep)) {
      return ep;
    }
  }

  return ExecutionProvider::CPU;
}

bool Backend::hasHardwareAcceleration() {
  const auto best = getBestAvailableProvider();
  return best != ExecutionProvider::CPU && best != ExecutionProvider::UNKNOWN;
}

}  // namespace pyreth
