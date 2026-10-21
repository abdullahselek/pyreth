#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pyreth {

/**
 * @struct TensorInfo
 * @brief Cached metadata describing a single model input or output tensor node.
 *
 * Captures the node's graph name, element data type, static/dynamic dimension values
 * (where -1 denotes a dynamic dimension), and optional symbolic dimension names
 * (such as "batch_size" or "sequence_length").
 */
struct PYRETH_API TensorInfo {
  std::string name;
  DataType dtype{DataType::Float32};
  std::vector<int64_t> shape;
  std::vector<std::string> symbolic_dims;

  /**
   * @brief Returns the tensor rank (number of dimensions).
   */
  [[nodiscard]] size_t rank() const noexcept { return shape.size(); }

  /**
   * @brief Returns true if this tensor is a rank-0 scalar.
   */
  [[nodiscard]] bool isScalar() const noexcept { return shape.empty(); }

  /**
   * @brief Returns true if the given dimension axis is dynamic (value < 0).
   */
  [[nodiscard]] bool isDynamicDim(size_t axis) const noexcept {
    return axis < shape.size() && shape[axis] < 0;
  }

  /**
   * @brief Returns true if any dimension of this tensor is dynamic (value < 0).
   */
  [[nodiscard]] bool hasDynamicShape() const noexcept {
    for (int64_t d : shape) {
      if (d < 0) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Returns true if the leading dimension (axis 0) is dynamic.
   */
  [[nodiscard]] bool hasDynamicBatch() const noexcept { return !shape.empty() && shape[0] < 0; }

  /**
   * @brief Returns the product of all fixed (positive) dimensions.
   *
   * Useful for computing the static per-sample element count when axis 0 is a dynamic batch.
   */
  [[nodiscard]] int64_t fixedElementCount() const noexcept {
    if (shape.empty()) {
      return 1;
    }
    int64_t product = 1;
    for (int64_t d : shape) {
      if (d == 0) {
        return 0;
      }
      if (d > 0) {
        product *= d;
      }
    }
    return product;
  }

  /**
   * @brief Zero-overhead validation of a concrete runtime Shape against this node's specification.
   *
   * Fixed dimensions (>= 0) must match exactly. Dynamic dimensions (< 0) accept any positive
   * runtime dimension (> 0).
   */
  [[nodiscard]] Status validateShape(const Shape& runtime_shape) const noexcept {
    if (runtime_shape.rank() != shape.size()) {
      return Status::failure(StatusCode::INVALID_DIMENSIONS,
                             "Tensor rank does not match model specification");
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      const int64_t expected = shape[i];
      const int64_t actual = runtime_shape[i];
      if (expected >= 0) {
        if (actual != expected) {
          return Status::failure(StatusCode::INVALID_DIMENSIONS,
                                 "Fixed tensor dimension does not match model specification");
        }
      } else {
        if (actual <= 0) {
          return Status::failure(StatusCode::INVALID_DIMENSIONS,
                                 "Dynamic tensor dimension must be strictly positive at runtime");
        }
      }
    }
    return Status::ok();
  }

  /**
   * @brief Zero-overhead validation of a concrete runtime Tensor (validity, DataType, and Shape).
   */
  [[nodiscard]] Status validateTensor(const Tensor& tensor) const noexcept {
    if (!tensor.storage()) {
      return Status::failure(StatusCode::INVALID_ARGUMENT, "Tensor is uninitialized or invalid");
    }
    if (tensor.dtype() != dtype) {
      return Status::failure(StatusCode::INVALID_ARGUMENT,
                             "Tensor DataType does not match model specification");
    }
    auto shape_status = validateShape(tensor.shape());
    if (!shape_status.isOk()) {
      return shape_status;
    }
    if (!tensor.isValid()) {
      return Status::failure(StatusCode::INVALID_ARGUMENT, "Tensor data buffer is null or invalid");
    }
    return Status::ok();
  }

  /**
   * @brief Formats a human-readable summary of this tensor's metadata.
   */
  [[nodiscard]] std::string toString() const {
    std::ostringstream ss;
    ss << "TensorInfo(name=\"" << name << "\", dtype=" << dataTypeToString(dtype) << ", shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      if (i < symbolic_dims.size() && !symbolic_dims[i].empty()) {
        ss << symbolic_dims[i] << "(" << shape[i] << ")";
      } else {
        ss << shape[i];
      }
    }
    ss << "])";
    return ss.str();
  }

  [[nodiscard]] bool operator==(const TensorInfo& other) const noexcept {
    return name == other.name && dtype == other.dtype && shape == other.shape &&
           symbolic_dims == other.symbolic_dims;
  }

  [[nodiscard]] bool operator!=(const TensorInfo& other) const noexcept {
    return !(*this == other);
  }
};

using IOTensorInfo = TensorInfo;

inline std::ostream& operator<<(std::ostream& os, const TensorInfo& info) {
  return os << info.toString();
}

/**
 * @struct ModelInfo
 * @brief Cached model metadata and IO schema for zero-overhead runtime inspection and validation.
 *
 * Populated once when an InferenceSession is created, avoiding repeated ONNX Runtime C-API calls
 * and heap allocations on the inference hot path.
 */
struct PYRETH_API ModelInfo {
  std::string producer_name;
  std::string graph_name;
  std::string domain;
  std::string description;
  std::string graph_description;
  int64_t version{0};
  std::string model_path;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
  std::unordered_map<std::string, std::string> custom_metadata;

  /**
   * @brief Returns the number of model inputs.
   */
  [[nodiscard]] size_t inputCount() const noexcept { return inputs.size(); }

  /**
   * @brief Returns the number of model outputs.
   */
  [[nodiscard]] size_t outputCount() const noexcept { return outputs.size(); }

  /**
   * @brief Looks up an input TensorInfo by name. Returns nullptr if not found.
   */
  [[nodiscard]] const TensorInfo* findInput(std::string_view name) const noexcept {
    for (const auto& in : inputs) {
      if (in.name == name) {
        return &in;
      }
    }
    return nullptr;
  }

  /**
   * @brief Looks up an output TensorInfo by name. Returns nullptr if not found.
   */
  [[nodiscard]] const TensorInfo* findOutput(std::string_view name) const noexcept {
    for (const auto& out : outputs) {
      if (out.name == name) {
        return &out;
      }
    }
    return nullptr;
  }

  /**
   * @brief Resolves an input node name to its positional index.
   */
  [[nodiscard]] Result<size_t> getInputIndex(std::string_view name) const noexcept {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].name == name) {
        return Result<size_t>::success(i);
      }
    }
    return Result<size_t>::failure(StatusCode::INVALID_ARGUMENT, "Unknown model input name");
  }

  /**
   * @brief Resolves an output node name to its positional index.
   */
  [[nodiscard]] Result<size_t> getOutputIndex(std::string_view name) const noexcept {
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].name == name) {
        return Result<size_t>::success(i);
      }
    }
    return Result<size_t>::failure(StatusCode::INVALID_ARGUMENT, "Unknown model output name");
  }

  /**
   * @brief Returns true if any model input has a dynamic batch dimension (axis 0 < 0).
   */
  [[nodiscard]] bool hasDynamicBatch() const noexcept {
    for (const auto& in : inputs) {
      if (in.hasDynamicBatch()) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Returns true if any model input contains at least one dynamic dimension.
   */
  [[nodiscard]] bool hasDynamicInputs() const noexcept {
    for (const auto& in : inputs) {
      if (in.hasDynamicShape()) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Returns true if any model output contains at least one dynamic dimension.
   */
  [[nodiscard]] bool hasDynamicOutputs() const noexcept {
    for (const auto& out : outputs) {
      if (out.hasDynamicShape()) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Validates a single input tensor at the given positional index.
   */
  [[nodiscard]] Status validateInput(size_t index, const Tensor& tensor) const noexcept {
    if (index >= inputs.size()) {
      return Status::failure(StatusCode::INVALID_ARGUMENT, "Input index out of range");
    }
    return inputs[index].validateTensor(tensor);
  }

  /**
   * @brief Validates a single input tensor by input node name.
   */
  [[nodiscard]] Status validateInput(std::string_view name, const Tensor& tensor) const noexcept {
    const auto* info = findInput(name);
    if (!info) {
      return Status::failure(StatusCode::INVALID_ARGUMENT, "Unknown model input name");
    }
    return info->validateTensor(tensor);
  }

  /**
   * @brief Zero-overhead validation of a full set of ordered input tensors.
   *
   * Verifies input count, per-tensor data types and shapes, and cross-input symbolic
   * dimension consistency (e.g., ensuring two inputs sharing symbolic dimension "batch_size"
   * have identical runtime values along those axes).
   */
  [[nodiscard]] Status validateInputs(std::span<const Tensor> input_tensors) const noexcept {
    if (input_tensors.size() != inputs.size()) {
      return Status::failure(StatusCode::INVALID_ARGUMENT,
                             "Number of input tensors does not match model input count");
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      auto status = inputs[i].validateTensor(input_tensors[i]);
      if (!status.isOk()) {
        return status;
      }
    }

    // Verify symbolic dimension consistency across inputs without heap allocation
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& info_a = inputs[i];
      const auto& shape_a = input_tensors[i].shape();
      for (size_t dim_a = 0; dim_a < info_a.symbolic_dims.size(); ++dim_a) {
        const auto& sym_a = info_a.symbolic_dims[dim_a];
        if (sym_a.empty()) {
          continue;
        }
        const int64_t val_a = shape_a[dim_a];
        for (size_t j = i; j < inputs.size(); ++j) {
          const auto& info_b = inputs[j];
          const auto& shape_b = input_tensors[j].shape();
          const size_t start_dim = (i == j) ? (dim_a + 1) : 0;
          for (size_t dim_b = start_dim; dim_b < info_b.symbolic_dims.size(); ++dim_b) {
            if (info_b.symbolic_dims[dim_b] == sym_a && shape_b[dim_b] != val_a) {
              return Status::failure(
                  StatusCode::INVALID_DIMENSIONS,
                  "Symbolic dimension size mismatch across model inputs at runtime");
            }
          }
        }
      }
    }

    return Status::ok();
  }

  /**
   * @brief Formats a human-readable summary of the model metadata and IO schema.
   */
  [[nodiscard]] std::string toString() const {
    std::ostringstream ss;
    ss << "ModelInfo(graph=\"" << graph_name << "\", producer=\"" << producer_name
       << "\", domain=\"" << domain << "\", version=" << version << ", inputs=[";
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << inputs[i].toString();
    }
    ss << "], outputs=[";
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (i > 0) {
        ss << ", ";
      }
      ss << outputs[i].toString();
    }
    ss << "])";
    return ss.str();
  }
};

inline std::ostream& operator<<(std::ostream& os, const ModelInfo& info) {
  return os << info.toString();
}

}  // namespace pyreth
