#include <json/json.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <pyreth/server/protocol.hpp>
#include <sstream>
#include <string>

namespace pyreth::server {

namespace {

std::string toUpper(std::string_view s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    result.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return result;
}

Result<KServeRequest> parseKServeRequestInternal(std::string_view json_str, ArenaAllocator* arena) {
  if (json_str.empty()) {
    return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT, "Empty JSON payload");
  }

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool ok = reader->parse(json_str.data(), json_str.data() + json_str.size(), &root, &errs);
  if (!ok || !root.isObject()) {
    return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT, "Malformed JSON payload");
  }

  if (!root.isMember("inputs")) {
    return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT,
                                          "Missing required 'inputs' array in JSON root");
  }

  const auto& inputs_val = root["inputs"];
  if (!inputs_val.isArray()) {
    return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT,
                                          "'inputs' field must be a JSON array");
  }

  KServeRequest request;
  request.inputs.reserve(inputs_val.size());

  for (Json::ArrayIndex i = 0; i < inputs_val.size(); ++i) {
    const auto& node = inputs_val[i];
    if (!node.isObject()) {
      return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT,
                                            "Input descriptor item must be a JSON object");
    }

    if (!node.isMember("name") || !node.isMember("shape") || !node.isMember("datatype") ||
        !node.isMember("data")) {
      return Result<KServeRequest>::failure(
          StatusCode::INVALID_ARGUMENT,
          "Missing mandatory fields in input descriptor ('name', 'shape', 'datatype', 'data')");
    }

    if (!node["name"].isString() || !node["shape"].isArray() || !node["datatype"].isString() ||
        !node["data"].isArray()) {
      return Result<KServeRequest>::failure(
          StatusCode::INVALID_ARGUMENT,
          "Invalid property types in input descriptor: expected string name, array shape, string "
          "datatype, array data");
    }

    std::string name = node["name"].asString();
    std::string datatype_str = node["datatype"].asString();

    auto dtype_res = kserveStringToDataType(datatype_str);
    if (!dtype_res.isOk()) {
      return Result<KServeRequest>::failure(dtype_res.error().code(), dtype_res.error().message());
    }
    DataType dtype = dtype_res.value();

    std::vector<int64_t> dims;
    const auto& shape_val = node["shape"];
    dims.reserve(shape_val.size());
    for (Json::ArrayIndex d = 0; d < shape_val.size(); ++d) {
      if (!shape_val[d].isInt64() && !shape_val[d].isInt()) {
        return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT,
                                              "Shape dimensions must be integer values");
      }
      int64_t dim_val = shape_val[d].asInt64();
      if (dim_val < 0) {
        return Result<KServeRequest>::failure(
            StatusCode::INVALID_DIMENSIONS,
            "Input tensor shape cannot have negative dimensions during inference");
      }
      dims.push_back(dim_val);
    }
    Shape shape(dims);

    const auto& data_val = node["data"];
    int64_t expected_elements = shape.numElements();
    if (static_cast<int64_t>(data_val.size()) != expected_elements) {
      return Result<KServeRequest>::failure(
          StatusCode::INVALID_DIMENSIONS,
          "Element count mismatch between shape product and data array length");
    }

    Result<Tensor> tensor_res =
        arena ? Tensor::create(shape, dtype, *arena) : Tensor::create(shape, dtype);
    if (!tensor_res.isOk()) {
      return Result<KServeRequest>::failure(tensor_res.error().code(),
                                            tensor_res.error().message());
    }
    Tensor tensor = std::move(tensor_res.value());

    switch (dtype) {
      case DataType::Float32: {
        float* ptr = tensor.data<float>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asFloat();
        }
        break;
      }
      case DataType::Float64: {
        double* ptr = tensor.data<double>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asDouble();
        }
        break;
      }
      case DataType::Int32: {
        int32_t* ptr = tensor.data<int32_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asInt();
        }
        break;
      }
      case DataType::Int64: {
        int64_t* ptr = tensor.data<int64_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asInt64();
        }
        break;
      }
      case DataType::Int8: {
        int8_t* ptr = tensor.data<int8_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = static_cast<int8_t>(data_val[idx].asInt());
        }
        break;
      }
      case DataType::UInt8: {
        uint8_t* ptr = tensor.data<uint8_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = static_cast<uint8_t>(data_val[idx].asUInt());
        }
        break;
      }
      case DataType::Bool: {
        bool* ptr = tensor.data<bool>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asBool();
        }
        break;
      }
      case DataType::Int16: {
        int16_t* ptr = tensor.data<int16_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = static_cast<int16_t>(data_val[idx].asInt());
        }
        break;
      }
      case DataType::UInt16: {
        uint16_t* ptr = tensor.data<uint16_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = static_cast<uint16_t>(data_val[idx].asUInt());
        }
        break;
      }
      case DataType::UInt32: {
        uint32_t* ptr = tensor.data<uint32_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = static_cast<uint32_t>(data_val[idx].asUInt64());
        }
        break;
      }
      case DataType::UInt64: {
        uint64_t* ptr = tensor.data<uint64_t>();
        for (Json::ArrayIndex idx = 0; idx < data_val.size(); ++idx) {
          ptr[idx] = data_val[idx].asUInt64();
        }
        break;
      }
      default:
        return Result<KServeRequest>::failure(StatusCode::INVALID_ARGUMENT,
                                              "Unsupported tensor datatype during deserialization");
    }

    request.inputs.push_back(KServeTensorDesc{std::move(name), shape, dtype, std::move(tensor)});
  }

  return Result<KServeRequest>::success(std::move(request));
}

}  // namespace

Result<DataType> kserveStringToDataType(std::string_view str) {
  std::string upper = toUpper(str);
  if (upper == "FP32" || upper == "FLOAT" || upper == "TYPE_FP32") {
    return Result<DataType>::success(DataType::Float32);
  }
  if (upper == "FP64" || upper == "DOUBLE" || upper == "TYPE_FP64") {
    return Result<DataType>::success(DataType::Float64);
  }
  if (upper == "INT32" || upper == "TYPE_INT32") {
    return Result<DataType>::success(DataType::Int32);
  }
  if (upper == "INT64" || upper == "TYPE_INT64") {
    return Result<DataType>::success(DataType::Int64);
  }
  if (upper == "INT8" || upper == "TYPE_INT8") {
    return Result<DataType>::success(DataType::Int8);
  }
  if (upper == "UINT8" || upper == "TYPE_UINT8") {
    return Result<DataType>::success(DataType::UInt8);
  }
  if (upper == "BOOL" || upper == "BOOLEAN" || upper == "TYPE_BOOL") {
    return Result<DataType>::success(DataType::Bool);
  }
  if (upper == "INT16" || upper == "TYPE_INT16") {
    return Result<DataType>::success(DataType::Int16);
  }
  if (upper == "UINT16" || upper == "TYPE_UINT16") {
    return Result<DataType>::success(DataType::UInt16);
  }
  if (upper == "UINT32" || upper == "TYPE_UINT32") {
    return Result<DataType>::success(DataType::UInt32);
  }
  if (upper == "UINT64" || upper == "TYPE_UINT64") {
    return Result<DataType>::success(DataType::UInt64);
  }
  if (upper == "FP16" || upper == "FLOAT16" || upper == "TYPE_FP16") {
    return Result<DataType>::success(DataType::Float16);
  }
  if (upper == "BF16" || upper == "BFLOAT16" || upper == "TYPE_BF16") {
    return Result<DataType>::success(DataType::BFloat16);
  }
  return Result<DataType>::failure(StatusCode::INVALID_ARGUMENT, "Unsupported datatype string");
}

std::string dataTypeToKServeString(DataType dtype) {
  switch (dtype) {
    case DataType::Float32:
      return "FP32";
    case DataType::Float64:
      return "FP64";
    case DataType::Int32:
      return "INT32";
    case DataType::Int64:
      return "INT64";
    case DataType::Int8:
      return "INT8";
    case DataType::UInt8:
      return "UINT8";
    case DataType::Bool:
      return "BOOL";
    case DataType::Int16:
      return "INT16";
    case DataType::UInt16:
      return "UINT16";
    case DataType::UInt32:
      return "UINT32";
    case DataType::UInt64:
      return "UINT64";
    case DataType::Float16:
      return "FP16";
    case DataType::BFloat16:
      return "BF16";
    default:
      return "UNKNOWN";
  }
}

std::vector<Tensor> KServeRequest::extractTensors() const {
  std::vector<Tensor> tensors;
  tensors.reserve(inputs.size());
  for (const auto& item : inputs) {
    tensors.push_back(item.tensor);
  }
  return tensors;
}

std::vector<std::string> KServeRequest::extractNames() const {
  std::vector<std::string> names;
  names.reserve(inputs.size());
  for (const auto& item : inputs) {
    names.push_back(item.name);
  }
  return names;
}

Result<KServeRequest> parseKServeRequest(std::string_view json_str, ArenaAllocator& arena) {
  return parseKServeRequestInternal(json_str, &arena);
}

Result<KServeRequest> parseKServeRequest(std::string_view json_str) {
  return parseKServeRequestInternal(json_str, nullptr);
}

Result<std::vector<Tensor>> parseKServeJson(std::string_view json_str, ArenaAllocator& arena,
                                            std::vector<std::string>* input_names) {
  auto req_res = parseKServeRequest(json_str, arena);
  if (!req_res.isOk()) {
    return Result<std::vector<Tensor>>::failure(req_res.error().code(), req_res.error().message());
  }
  if (input_names) {
    *input_names = req_res.value().extractNames();
  }
  return Result<std::vector<Tensor>>::success(req_res.value().extractTensors());
}

Result<std::vector<Tensor>> parseKServeJson(std::string_view json_str,
                                            std::vector<std::string>* input_names) {
  auto req_res = parseKServeRequest(json_str);
  if (!req_res.isOk()) {
    return Result<std::vector<Tensor>>::failure(req_res.error().code(), req_res.error().message());
  }
  if (input_names) {
    *input_names = req_res.value().extractNames();
  }
  return Result<std::vector<Tensor>>::success(req_res.value().extractTensors());
}

Result<std::string> serializeKServeResponse(std::string_view model_name,
                                            std::span<const std::string> output_names,
                                            std::span<const Tensor> outputs,
                                            std::string_view model_version) {
  Json::Value root;
  root["model_name"] = std::string(model_name);
  if (!model_version.empty()) {
    root["model_version"] = std::string(model_version);
  }
  root["outputs"] = Json::arrayValue;

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    Json::Value out_val;
    out_val["name"] = (i < output_names.size()) ? output_names[i] : ("output_" + std::to_string(i));

    Json::Value shape_val(Json::arrayValue);
    for (int64_t d : tensor.shape().dims()) {
      shape_val.append(static_cast<Json::Value::Int64>(d));
    }
    out_val["shape"] = shape_val;
    out_val["datatype"] = dataTypeToKServeString(tensor.dtype());

    Json::Value data_val(Json::arrayValue);
    int64_t n = tensor.numElements();

    switch (tensor.dtype()) {
      case DataType::Float32: {
        const float* ptr = tensor.data<float>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::Float64: {
        const double* ptr = tensor.data<double>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::Int32: {
        const int32_t* ptr = tensor.data<int32_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::Int64: {
        const int64_t* ptr = tensor.data<int64_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(static_cast<Json::Value::Int64>(ptr[j]));
        }
        break;
      }
      case DataType::Int8: {
        const int8_t* ptr = tensor.data<int8_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(static_cast<int>(ptr[j]));
        }
        break;
      }
      case DataType::UInt8: {
        const uint8_t* ptr = tensor.data<uint8_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(static_cast<unsigned int>(ptr[j]));
        }
        break;
      }
      case DataType::Bool: {
        const bool* ptr = tensor.data<bool>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::Int16: {
        const int16_t* ptr = tensor.data<int16_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::UInt16: {
        const uint16_t* ptr = tensor.data<uint16_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::UInt32: {
        const uint32_t* ptr = tensor.data<uint32_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(ptr[j]);
        }
        break;
      }
      case DataType::UInt64: {
        const uint64_t* ptr = tensor.data<uint64_t>();
        for (int64_t j = 0; j < n; ++j) {
          data_val.append(static_cast<Json::Value::UInt64>(ptr[j]));
        }
        break;
      }
      default:
        return Result<std::string>::failure(StatusCode::INVALID_ARGUMENT,
                                            "Unsupported tensor datatype during serialization");
    }

    out_val["data"] = data_val;
    root["outputs"].append(out_val);
  }

  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  return Result<std::string>::success(Json::writeString(writer, root));
}

Result<std::string> serializeKServeResponse(const KServeResponse& response) {
  std::vector<std::string> names;
  std::vector<Tensor> tensors;
  names.reserve(response.outputs.size());
  tensors.reserve(response.outputs.size());
  for (const auto& out : response.outputs) {
    names.push_back(out.name);
    tensors.push_back(out.tensor);
  }
  return serializeKServeResponse(response.model_name, names, tensors, response.model_version);
}

Result<std::string> serializeKServeJson(std::string_view model_name,
                                        std::span<const std::string> output_names,
                                        std::span<const Tensor> outputs,
                                        std::string_view model_version) {
  return serializeKServeResponse(model_name, output_names, outputs, model_version);
}

Result<void> validateKServeRequest(const KServeRequest& req, const ModelInfo& info) {
  if (req.inputs.size() != info.inputs.size()) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                 "Input count mismatch with model specification");
  }

  for (size_t i = 0; i < req.inputs.size(); ++i) {
    const auto& inp = req.inputs[i];
    const TensorInfo* target_info = nullptr;

    if (!inp.name.empty()) {
      target_info = info.findInput(inp.name);
    }
    if (!target_info && i < info.inputs.size()) {
      target_info = &info.inputs[i];
    }

    if (!target_info) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Input tensor not found in model schema");
    }

    if (inp.datatype != target_info->dtype) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Input tensor datatype mismatch with model schema");
    }

    auto shape_status = target_info->validateShape(inp.shape);
    if (!shape_status.isOk()) {
      return Result<void>::failure(shape_status.error().code(), shape_status.error().message());
    }
  }

  return Result<void>::success();
}

Result<void> validateKServeRequest(std::span<const Tensor> tensors,
                                   std::span<const std::string> names, const ModelInfo& info) {
  if (tensors.size() != info.inputs.size()) {
    return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                 "Input tensor count mismatch with model specification");
  }

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const TensorInfo* target_info = nullptr;
    if (i < names.size() && !names[i].empty()) {
      target_info = info.findInput(names[i]);
    }
    if (!target_info && i < info.inputs.size()) {
      target_info = &info.inputs[i];
    }

    if (!target_info) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Input tensor not found in model schema");
    }

    if (tensor.dtype() != target_info->dtype) {
      return Result<void>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Input tensor datatype mismatch with model schema");
    }

    auto shape_status = target_info->validateShape(tensor.shape());
    if (!shape_status.isOk()) {
      return Result<void>::failure(shape_status.error().code(), shape_status.error().message());
    }
  }

  return Result<void>::success();
}

}  // namespace pyreth::server
