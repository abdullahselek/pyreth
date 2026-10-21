#pragma once

#include <onnxruntime_cxx_api.h>

#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>

namespace pyreth {

/**
 * @brief Converts a Pyreth DataType to the corresponding ONNX Runtime element data type.
 */
[[nodiscard]] constexpr ONNXTensorElementDataType toOrtDataType(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::Float32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DataType::Float16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DataType::BFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    case DataType::Float64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DataType::Int8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DataType::UInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DataType::Int16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DataType::UInt16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DataType::Int32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DataType::UInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DataType::Int64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DataType::UInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case DataType::Bool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
}

/**
 * @brief Converts an ONNX Runtime element data type to the corresponding Pyreth DataType.
 */
[[nodiscard]] constexpr DataType fromOrtDataType(ONNXTensorElementDataType ort_type) noexcept {
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

/**
 * @brief Creates a zero-copy Ort::Value wrapping the underlying memory of a contiguous Tensor.
 *
 * Uses `Ort::Value::CreateTensor` (`CreateTensorWithDataAsOrtValue`) so no buffer allocation
 * or memory copy occurs.
 *
 * @warning The returned `Ort::Value` does NOT own the underlying memory buffer. The caller
 * must ensure `tensor` (or a shallow copy sharing its `TensorStorage`) remains alive and
 * pinned for the entire lifetime of the returned `Ort::Value`.
 */
[[nodiscard]] PYRETH_API Result<Ort::Value> tensorToOrtValue(const Tensor& tensor,
                                                             const Ort::MemoryInfo& memory_info);

/**
 * @brief Creates a zero-copy Ort::Value wrapping the underlying CPU memory of a contiguous Tensor.
 */
[[nodiscard]] PYRETH_API Result<Ort::Value> tensorToOrtValue(const Tensor& tensor);

/**
 * @brief Creates a zero-copy Pyreth Tensor taking ownership of an rvalue Ort::Value.
 *
 * Wraps the raw tensor pointer of `ort_value` inside an `ExternalStorage` whose keep-alive
 * `owner` holds the moved `Ort::Value`. No memory is copied, and the `Ort::Value` is released
 * automatically when the last `Tensor` view referencing it is destroyed.
 */
[[nodiscard]] PYRETH_API Result<Tensor> tensorFromOrtValue(Ort::Value&& ort_value);

}  // namespace pyreth
