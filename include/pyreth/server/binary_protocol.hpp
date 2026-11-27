#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace pyreth::server {

inline constexpr uint32_t kBinaryTensorMagic = 0x50595254;  // "PYRT"
inline constexpr size_t kBinaryTensorHeaderSize = 64;
inline constexpr std::string_view kBinaryContentType = "application/octet-stream";

#pragma pack(push, 1)
/**
 * @struct BinaryTensorHeader
 * @brief 64-byte aligned SIMD header prepending raw binary tensor buffers.
 *
 * Layout:
 *   [0..3]   uint32_t magic (0x50595254 "PYRT")
 *   [4]      uint8_t  dtype (DataType enum value: Float32=0, Float16=1, etc.)
 *   [5]      uint8_t  ndim  (Number of dimensions, 1 <= ndim <= 4)
 *   [6..7]   uint16_t reserved (zero-padding to 8 bytes)
 *   [8..39]  int64_t  dims[4] (Dimensions up to 4D inline)
 *   [40..63] uint8_t  padding[24] (Zero-padding to 64 bytes)
 */
struct PYRETH_API BinaryTensorHeader {
  uint32_t magic{kBinaryTensorMagic};
  uint8_t dtype{0};
  uint8_t ndim{0};
  uint16_t reserved{0};
  int64_t dims[4]{0, 0, 0, 0};
  uint8_t padding[24]{0};
};
#pragma pack(pop)

static_assert(sizeof(BinaryTensorHeader) == 64, "BinaryTensorHeader must be exactly 64 bytes");

/**
 * @brief Parses and validates the 64-byte binary tensor header from raw bytes.
 *
 * @param bytes Input byte stream containing at least 64 bytes.
 * @return Result containing BinaryTensorHeader or diagnostic Error.
 */
PYRETH_API Result<BinaryTensorHeader> parseBinaryHeader(std::span<const uint8_t> bytes);

/**
 * @brief Zero-copy deserialization of a raw binary tensor payload.
 *
 * Wraps payload memory starting at byte 64 directly into a pyreth::Tensor without copying.
 *
 * @param bytes Input byte stream with 64-byte header and raw tensor data.
 * @param owner Optional shared ownership handle for backing buffer lifetime.
 * @return Result containing zero-copy Tensor or diagnostic Error.
 */
PYRETH_API Result<Tensor> deserializeBinaryTensor(std::span<const uint8_t> bytes,
                                                  std::shared_ptr<void> owner = nullptr);

/**
 * @brief Serializes a Tensor into standard 64-byte header + contiguous raw buffer format.
 *
 * @param tensor Source tensor to serialize.
 * @return Result containing serialized binary buffer.
 */
PYRETH_API Result<std::vector<uint8_t>> serializeBinaryTensor(const Tensor& tensor);

}  // namespace pyreth::server
