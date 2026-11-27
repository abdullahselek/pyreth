#include <cstring>
#include <pyreth/server/binary_protocol.hpp>

namespace pyreth::server {

Result<BinaryTensorHeader> parseBinaryHeader(std::span<const uint8_t> bytes) {
  if (bytes.size() < kBinaryTensorHeaderSize) {
    return Result<BinaryTensorHeader>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Truncated binary tensor payload: header must be at least 64 bytes");
  }

  BinaryTensorHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));

  if (header.magic != kBinaryTensorMagic) {
    return Result<BinaryTensorHeader>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Invalid magic in binary tensor header: expected 0x50595254 ('PYRT')");
  }

  if (header.dtype > static_cast<uint8_t>(DataType::Bool)) {
    return Result<BinaryTensorHeader>::failure(
        StatusCode::INVALID_ARGUMENT, "Unsupported or invalid data type in binary tensor header");
  }

  if (header.ndim == 0 || header.ndim > 4) {
    return Result<BinaryTensorHeader>::failure(
        StatusCode::INVALID_DIMENSIONS,
        "Invalid tensor rank in binary tensor header: ndim must be between 1 and 4");
  }

  for (size_t i = 0; i < header.ndim; ++i) {
    if (header.dims[i] <= 0) {
      return Result<BinaryTensorHeader>::failure(
          StatusCode::INVALID_DIMENSIONS,
          "Invalid dimensions in binary tensor header: dimensions must be positive");
    }
  }

  return Result<BinaryTensorHeader>::success(header);
}

Result<Tensor> deserializeBinaryTensor(std::span<const uint8_t> bytes,
                                       std::shared_ptr<void> owner) {
  auto header_res = parseBinaryHeader(bytes);
  if (!header_res.isOk()) {
    return Result<Tensor>::failure(header_res.error());
  }

  const auto& header = header_res.value();
  std::vector<int64_t> dims(header.dims, header.dims + header.ndim);
  Shape shape(std::move(dims));
  auto dtype = static_cast<DataType>(header.dtype);

  size_t elem_size = dataTypeSize(dtype);
  int64_t num_elements = shape.numElements();
  size_t expected_payload_bytes = static_cast<size_t>(num_elements) * elem_size;
  size_t actual_payload_bytes = bytes.size() - kBinaryTensorHeaderSize;

  if (actual_payload_bytes != expected_payload_bytes) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Payload byte size mismatch: header expects " +
                                       std::to_string(expected_payload_bytes) + " bytes, got " +
                                       std::to_string(actual_payload_bytes) + " bytes");
  }

  void* data_ptr = const_cast<uint8_t*>(bytes.data() + kBinaryTensorHeaderSize);
  return Tensor::fromBuffer(data_ptr, shape, dtype, {}, std::move(owner));
}

Result<std::vector<uint8_t>> serializeBinaryTensor(const Tensor& tensor) {
  if (!tensor.isValid()) {
    return Result<std::vector<uint8_t>>::failure(StatusCode::INVALID_ARGUMENT,
                                                 "Cannot serialize null or invalid Tensor");
  }

  if (tensor.ndim() == 0 || tensor.ndim() > 4) {
    return Result<std::vector<uint8_t>>::failure(
        StatusCode::INVALID_DIMENSIONS,
        "Binary protocol supports tensors with rank between 1 and 4");
  }

  BinaryTensorHeader header;
  header.magic = kBinaryTensorMagic;
  header.dtype = static_cast<uint8_t>(tensor.dtype());
  header.ndim = static_cast<uint8_t>(tensor.ndim());
  header.reserved = 0;
  std::memset(header.dims, 0, sizeof(header.dims));
  std::memset(header.padding, 0, sizeof(header.padding));

  for (size_t i = 0; i < tensor.ndim(); ++i) {
    header.dims[i] = tensor.shape()[i];
  }

  size_t tensor_bytes = tensor.totalBytes();
  std::vector<uint8_t> out_buffer(kBinaryTensorHeaderSize + tensor_bytes);
  std::memcpy(out_buffer.data(), &header, sizeof(header));

  if (tensor.isContiguous()) {
    std::memcpy(out_buffer.data() + kBinaryTensorHeaderSize, tensor.data(), tensor_bytes);
  } else {
    auto contig_res = tensor.toContiguous();
    if (!contig_res.isOk()) {
      return Result<std::vector<uint8_t>>::failure(contig_res.error());
    }
    std::memcpy(out_buffer.data() + kBinaryTensorHeaderSize, contig_res.value().data(),
                tensor_bytes);
  }

  return Result<std::vector<uint8_t>>::success(std::move(out_buffer));
}

}  // namespace pyreth::server
