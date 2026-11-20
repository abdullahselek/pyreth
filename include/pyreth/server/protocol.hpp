#pragma once

#include <cstdint>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pyreth::server {

/**
 * @brief Maps a KServe v2 datatype string (e.g. "FP32", "INT64") to Pyreth DataType.
 */
PYRETH_API Result<DataType> kserveStringToDataType(std::string_view str);

/**
 * @brief Maps Pyreth DataType to canonical KServe v2 datatype string.
 */
PYRETH_API std::string dataTypeToKServeString(DataType dtype);

/**
 * @struct KServeTensorDesc
 * @brief Descriptor and storage for an input/output tensor in KServe protocol.
 */
struct PYRETH_API KServeTensorDesc {
  std::string name;
  Shape shape;
  DataType datatype{DataType::Float32};
  Tensor tensor;
};

/**
 * @struct KServeRequest
 * @brief Parsed KServe v2 JSON inference request payload.
 */
struct PYRETH_API KServeRequest {
  std::vector<KServeTensorDesc> inputs;

  [[nodiscard]] std::vector<Tensor> extractTensors() const;
  [[nodiscard]] std::vector<std::string> extractNames() const;
};

/**
 * @struct KServeResponse
 * @brief Structured KServe v2 JSON inference response payload.
 */
struct PYRETH_API KServeResponse {
  std::string model_name;
  std::string model_version;
  std::vector<KServeTensorDesc> outputs;
};

/**
 * @brief Parses a KServe v2 JSON request using memory leased from a request-scoped ArenaAllocator.
 */
PYRETH_API Result<KServeRequest> parseKServeRequest(std::string_view json_str,
                                                    ArenaAllocator& arena);

/**
 * @brief Parses a KServe v2 JSON request using standard heap tensor allocations.
 */
PYRETH_API Result<KServeRequest> parseKServeRequest(std::string_view json_str);

/**
 * @brief Convenience overload parsing JSON directly into a list of Tensors using ArenaAllocator.
 */
PYRETH_API Result<std::vector<Tensor>> parseKServeJson(
    std::string_view json_str, ArenaAllocator& arena,
    std::vector<std::string>* input_names = nullptr);

/**
 * @brief Convenience overload parsing JSON directly into a list of Tensors using heap allocations.
 */
PYRETH_API Result<std::vector<Tensor>> parseKServeJson(
    std::string_view json_str, std::vector<std::string>* input_names = nullptr);

/**
 * @brief Serializes output tensors into a standard KServe v2 JSON response.
 */
PYRETH_API Result<std::string> serializeKServeResponse(std::string_view model_name,
                                                       std::span<const std::string> output_names,
                                                       std::span<const Tensor> outputs,
                                                       std::string_view model_version = "");

/**
 * @brief Serializes a structured KServeResponse into a JSON string.
 */
PYRETH_API Result<std::string> serializeKServeResponse(const KServeResponse& response);

/**
 * @brief Convenience alias for serializeKServeResponse.
 */
PYRETH_API Result<std::string> serializeKServeJson(std::string_view model_name,
                                                   std::span<const std::string> output_names,
                                                   std::span<const Tensor> outputs,
                                                   std::string_view model_version = "");

/**
 * @brief Validates a parsed KServeRequest against model input schema in ModelInfo.
 */
PYRETH_API Result<void> validateKServeRequest(const KServeRequest& req, const ModelInfo& info);

/**
 * @brief Validates tensors and optional input names against model input schema in ModelInfo.
 */
PYRETH_API Result<void> validateKServeRequest(std::span<const Tensor> tensors,
                                              std::span<const std::string> names,
                                              const ModelInfo& info);

}  // namespace pyreth::server
