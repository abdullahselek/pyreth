#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/result.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pyreth {

/**
 * @enum DataType
 * @brief Discrete enumeration of supported data types for Tensor elements.
 */
enum class DataType : uint8_t {
  Float32 = 0,
  Float16,
  BFloat16,
  Float64,
  Int8,
  UInt8,
  Int16,
  UInt16,
  Int32,
  UInt32,
  Int64,
  UInt64,
  Bool,
};

/**
 * @brief Return byte size for each DataType.
 */
[[nodiscard]] constexpr size_t dataTypeSize(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::Float32:
      return 4;
    case DataType::Float16:
    case DataType::BFloat16:
      return 2;
    case DataType::Float64:
      return 8;
    case DataType::Int8:
    case DataType::UInt8:
    case DataType::Bool:
      return 1;
    case DataType::Int16:
    case DataType::UInt16:
      return 2;
    case DataType::Int32:
    case DataType::UInt32:
      return 4;
    case DataType::Int64:
    case DataType::UInt64:
      return 8;
    default:
      return 0;
  }
}

/**
 * @brief Canonical string representation for DataType.
 */
[[nodiscard]] constexpr std::string_view dataTypeToString(DataType dtype) noexcept {
  switch (dtype) {
    case DataType::Float32:
      return "Float32";
    case DataType::Float16:
      return "Float16";
    case DataType::BFloat16:
      return "BFloat16";
    case DataType::Float64:
      return "Float64";
    case DataType::Int8:
      return "Int8";
    case DataType::UInt8:
      return "UInt8";
    case DataType::Int16:
      return "Int16";
    case DataType::UInt16:
      return "UInt16";
    case DataType::Int32:
      return "Int32";
    case DataType::UInt32:
      return "UInt32";
    case DataType::Int64:
      return "Int64";
    case DataType::UInt64:
      return "UInt64";
    case DataType::Bool:
      return "Bool";
    default:
      return "Unknown";
  }
}

inline std::ostream& operator<<(std::ostream& os, DataType dtype) {
  return os << dataTypeToString(dtype);
}

/**
 * @brief Type-safe mapping from C++ types to Pyreth DataType.
 */
template <typename T>
[[nodiscard]] constexpr DataType dataTypeOf() noexcept {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, float>) {
    return DataType::Float32;
  } else if constexpr (std::is_same_v<U, double>) {
    return DataType::Float64;
  } else if constexpr (std::is_same_v<U, int8_t> ||
                       (std::is_same_v<U, char> && std::is_signed_v<char>)) {
    return DataType::Int8;
  } else if constexpr (std::is_same_v<U, uint8_t> || (std::is_same_v<U, unsigned char>)) {
    return DataType::UInt8;
  } else if constexpr (std::is_same_v<U, int16_t> || std::is_same_v<U, short>) {
    return DataType::Int16;
  } else if constexpr (std::is_same_v<U, uint16_t> || std::is_same_v<U, unsigned short>) {
    return DataType::UInt16;
  } else if constexpr (std::is_same_v<U, int32_t> || (sizeof(int) == 4 && std::is_same_v<U, int>)) {
    return DataType::Int32;
  } else if constexpr (std::is_same_v<U, uint32_t> ||
                       (sizeof(unsigned int) == 4 && std::is_same_v<U, unsigned int>)) {
    return DataType::UInt32;
  } else if constexpr (std::is_same_v<U, int64_t> ||
                       (sizeof(long) == 8 && std::is_same_v<U, long>) ||
                       (sizeof(long long) == 8 && std::is_same_v<U, long long>)) {
    return DataType::Int64;
  } else if constexpr (std::is_same_v<U, uint64_t> ||
                       (sizeof(unsigned long) == 8 && std::is_same_v<U, unsigned long>) ||
                       (sizeof(unsigned long long) == 8 && std::is_same_v<U, unsigned long long>)) {
    return DataType::UInt64;
  } else if constexpr (std::is_same_v<U, bool>) {
    return DataType::Bool;
  } else {
    static_assert(!sizeof(U), "Unsupported C++ type for pyreth::dataTypeOf<T>()");
  }
}

/**
 * @class Shape
 * @brief Represents the dimensions of an n-dimensional tensor.
 */
class PYRETH_API Shape {
public:
  Shape() = default;

  Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

  explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

  explicit Shape(std::span<const int64_t> dims) : dims_(dims.begin(), dims.end()) {}

  template <std::integral IntType>
  Shape(std::initializer_list<IntType> dims) {
    dims_.reserve(dims.size());
    for (auto d : dims) {
      dims_.push_back(static_cast<int64_t>(d));
    }
  }

  [[nodiscard]] size_t rank() const noexcept { return dims_.size(); }
  [[nodiscard]] size_t ndim() const noexcept { return dims_.size(); }

  [[nodiscard]] int64_t operator[](size_t index) const { return dims_.at(index); }
  [[nodiscard]] int64_t& operator[](size_t index) { return dims_.at(index); }

  [[nodiscard]] int64_t dim(size_t index) const { return dims_.at(index); }

  [[nodiscard]] const std::vector<int64_t>& dims() const noexcept { return dims_; }
  [[nodiscard]] std::span<const int64_t> span() const noexcept { return dims_; }
  [[nodiscard]] const int64_t* data() const noexcept { return dims_.data(); }

  [[nodiscard]] int64_t numElements() const noexcept {
    if (dims_.empty()) {
      return 1;  // Rank-0 scalar holds 1 element
    }
    int64_t total = 1;
    for (int64_t d : dims_) {
      if (d == 0)
        return 0;
      total *= d;
    }
    return total;
  }

  [[nodiscard]] bool empty() const noexcept { return numElements() == 0; }
  [[nodiscard]] bool isScalar() const noexcept { return dims_.empty(); }

  [[nodiscard]] bool operator==(const Shape& other) const noexcept { return dims_ == other.dims_; }
  [[nodiscard]] bool operator!=(const Shape& other) const noexcept { return dims_ != other.dims_; }

  [[nodiscard]] std::string toString() const;

  static Shape scalar() { return Shape(); }

private:
  std::vector<int64_t> dims_;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.toString();
}

/**
 * @brief Compute standard row-major contiguous strides for a shape.
 */
[[nodiscard]] PYRETH_API std::vector<int64_t> computeContiguousStrides(const Shape& shape);

/**
 * @brief Check whether a combination of shape and strides forms a contiguous row-major layout.
 */
[[nodiscard]] PYRETH_API bool isContiguousLayout(const Shape& shape,
                                                 const std::vector<int64_t>& strides);

/**
 * @enum StorageType
 * @brief Backing allocation strategy for a Tensor.
 */
enum class StorageType : uint8_t { OWNED = 0, ARENA, POOLED, EXTERNAL };

[[nodiscard]] constexpr std::string_view storageTypeToString(StorageType type) noexcept {
  switch (type) {
    case StorageType::OWNED:
      return "OWNED";
    case StorageType::ARENA:
      return "ARENA";
    case StorageType::POOLED:
      return "POOLED";
    case StorageType::EXTERNAL:
      return "EXTERNAL";
    default:
      return "UNKNOWN";
  }
}

/**
 * @class TensorStorage
 * @brief Abstract interface for tensor memory backing with lifetime tracking.
 */
class PYRETH_API TensorStorage {
public:
  virtual ~TensorStorage() = default;

  [[nodiscard]] virtual void* data() noexcept = 0;
  [[nodiscard]] virtual const void* data() const noexcept = 0;
  [[nodiscard]] virtual size_t sizeBytes() const noexcept = 0;
  [[nodiscard]] virtual StorageType type() const noexcept = 0;

  [[nodiscard]] virtual bool isArenaBacked() const noexcept {
    return type() == StorageType::ARENA || type() == StorageType::POOLED;
  }
};

/**
 * @class OwnedStorage
 * @brief Standalone 64-byte aligned SIMD heap allocation.
 */
class PYRETH_API OwnedStorage : public TensorStorage {
public:
  explicit OwnedStorage(size_t size_bytes);
  ~OwnedStorage() override;

  OwnedStorage(const OwnedStorage&) = delete;
  OwnedStorage& operator=(const OwnedStorage&) = delete;

  [[nodiscard]] void* data() noexcept override { return data_; }
  [[nodiscard]] const void* data() const noexcept override { return data_; }
  [[nodiscard]] size_t sizeBytes() const noexcept override { return size_bytes_; }
  [[nodiscard]] StorageType type() const noexcept override { return StorageType::OWNED; }

private:
  void* data_{nullptr};
  size_t size_bytes_{0};
};

/**
 * @class ArenaStorage
 * @brief Backed by an ArenaAllocator. Automatically pins the arena on creation
 * and unpins on destruction to enforce the strict lifetime contract.
 */
class PYRETH_API ArenaStorage : public TensorStorage {
public:
  ArenaStorage(ArenaAllocator& arena, size_t size_bytes);
  ~ArenaStorage() override;

  ArenaStorage(const ArenaStorage&) = delete;
  ArenaStorage& operator=(const ArenaStorage&) = delete;

  [[nodiscard]] void* data() noexcept override { return data_; }
  [[nodiscard]] const void* data() const noexcept override { return data_; }
  [[nodiscard]] size_t sizeBytes() const noexcept override { return size_bytes_; }
  [[nodiscard]] StorageType type() const noexcept override { return StorageType::ARENA; }
  [[nodiscard]] ArenaAllocator& arena() noexcept { return arena_; }

private:
  ArenaAllocator& arena_;
  void* data_{nullptr};
  size_t size_bytes_{0};
};

/**
 * @class PooledArenaStorage
 * @brief Holds a leased arena from MemoryPool. MemoryPool::Lease is retained
 * until all references are destroyed, automatically resetting and returning
 * the arena in O(1) time.
 */
class PYRETH_API PooledArenaStorage : public TensorStorage {
public:
  PooledArenaStorage(MemoryPool::Lease lease, size_t size_bytes);
  ~PooledArenaStorage() override = default;

  PooledArenaStorage(const PooledArenaStorage&) = delete;
  PooledArenaStorage& operator=(const PooledArenaStorage&) = delete;

  [[nodiscard]] void* data() noexcept override { return data_; }
  [[nodiscard]] const void* data() const noexcept override { return data_; }
  [[nodiscard]] size_t sizeBytes() const noexcept override { return size_bytes_; }
  [[nodiscard]] StorageType type() const noexcept override { return StorageType::POOLED; }
  [[nodiscard]] ArenaAllocator& arena() noexcept { return lease_.get(); }

private:
  MemoryPool::Lease lease_;
  void* data_{nullptr};
  size_t size_bytes_{0};
};

/**
 * @class ExternalStorage
 * @brief Zero-copy view over external memory with an optional keep-alive owner.
 */
class PYRETH_API ExternalStorage : public TensorStorage {
public:
  ExternalStorage(void* data, size_t size_bytes, std::shared_ptr<void> owner = nullptr);
  ~ExternalStorage() override = default;

  ExternalStorage(const ExternalStorage&) = delete;
  ExternalStorage& operator=(const ExternalStorage&) = delete;

  [[nodiscard]] void* data() noexcept override { return data_; }
  [[nodiscard]] const void* data() const noexcept override { return data_; }
  [[nodiscard]] size_t sizeBytes() const noexcept override { return size_bytes_; }
  [[nodiscard]] StorageType type() const noexcept override { return StorageType::EXTERNAL; }
  [[nodiscard]] const std::shared_ptr<void>& owner() const noexcept { return owner_; }

private:
  void* data_{nullptr};
  size_t size_bytes_{0};
  std::shared_ptr<void> owner_{nullptr};
};

/**
 * @class Tensor
 * @brief Multi-dimensional array abstraction with strided memory layout,
 * SIMD alignment, and strict zero-copy lifetime guarantees.
 */
class PYRETH_API Tensor {
public:
  Tensor() = default;

  Tensor(Shape shape, DataType dtype, std::shared_ptr<TensorStorage> storage,
         std::vector<int64_t> strides = {}, size_t byte_offset = 0);

  // Copy operations produce shallow views sharing the underlying storage
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);

  // Move operations transfer ownership
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;

  ~Tensor() = default;

  // Factory methods
  static Result<Tensor> create(const Shape& shape, DataType dtype);
  static Result<Tensor> create(const Shape& shape, DataType dtype, ArenaAllocator& arena);
  static Result<Tensor> create(const Shape& shape, DataType dtype, MemoryPool& pool);
  static Result<Tensor> create(const Shape& shape, DataType dtype, MemoryPool::Lease lease);
  static Result<Tensor> fromBuffer(void* data, const Shape& shape, DataType dtype,
                                   std::vector<int64_t> strides = {},
                                   std::shared_ptr<void> owner = nullptr);

  // Metadata accessors
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] DataType dtype() const noexcept { return dtype_; }
  [[nodiscard]] const std::vector<int64_t>& strides() const noexcept { return strides_; }
  [[nodiscard]] size_t rank() const noexcept { return shape_.rank(); }
  [[nodiscard]] size_t ndim() const noexcept { return shape_.ndim(); }
  [[nodiscard]] int64_t numElements() const noexcept { return shape_.numElements(); }
  [[nodiscard]] size_t totalBytes() const noexcept {
    return static_cast<size_t>(shape_.numElements()) * elementSize();
  }
  [[nodiscard]] size_t elementSize() const noexcept { return dataTypeSize(dtype_); }
  [[nodiscard]] size_t byteOffset() const noexcept { return byte_offset_; }
  [[nodiscard]] bool isContiguous() const noexcept { return is_contiguous_; }
  [[nodiscard]] bool isValid() const noexcept { return storage_ != nullptr && data_ != nullptr; }
  [[nodiscard]] explicit operator bool() const noexcept { return isValid(); }

  [[nodiscard]] StorageType storageType() const noexcept {
    return storage_ ? storage_->type() : StorageType::OWNED;
  }
  [[nodiscard]] bool isArenaBacked() const noexcept {
    return storage_ && storage_->isArenaBacked();
  }
  [[nodiscard]] long useCount() const noexcept { return storage_.use_count(); }
  [[nodiscard]] const std::shared_ptr<TensorStorage>& storage() const noexcept { return storage_; }

  [[nodiscard]] bool isAligned(size_t alignment = DEFAULT_ALIGNMENT) const noexcept {
    return reinterpret_cast<uintptr_t>(data_) % alignment == 0;
  }

  // Data pointers
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }

  template <typename T>
  [[nodiscard]] T* data() noexcept {
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  [[nodiscard]] const T* data() const noexcept {
    return reinterpret_cast<const T*>(data_);
  }

  // Element access
  template <typename T>
  [[nodiscard]] T& at(std::span<const int64_t> indices) {
    return *elementAddress<T>(indices);
  }

  template <typename T>
  [[nodiscard]] const T& at(std::span<const int64_t> indices) const {
    return *elementAddress<T>(indices);
  }

  template <typename T>
  [[nodiscard]] T& at(std::initializer_list<int64_t> indices) {
    return at<T>(std::span<const int64_t>(indices.begin(), indices.size()));
  }

  template <typename T>
  [[nodiscard]] const T& at(std::initializer_list<int64_t> indices) const {
    return at<T>(std::span<const int64_t>(indices.begin(), indices.size()));
  }

  template <typename T, typename First, typename Second, typename... Rest>
    requires(std::is_integral_v<First> && std::is_integral_v<Second> &&
             (... && std::is_integral_v<Rest>))
  [[nodiscard]] T& at(First first, Second second, Rest... rest) {
    int64_t coords[] = {static_cast<int64_t>(first), static_cast<int64_t>(second),
                        static_cast<int64_t>(rest)...};
    return at<T>(std::span<const int64_t>(coords, 2 + sizeof...(Rest)));
  }

  template <typename T, typename First, typename Second, typename... Rest>
    requires(std::is_integral_v<First> && std::is_integral_v<Second> &&
             (... && std::is_integral_v<Rest>))
  [[nodiscard]] const T& at(First first, Second second, Rest... rest) const {
    int64_t coords[] = {static_cast<int64_t>(first), static_cast<int64_t>(second),
                        static_cast<int64_t>(rest)...};
    return at<T>(std::span<const int64_t>(coords, 2 + sizeof...(Rest)));
  }

  template <typename T, std::integral IntType>
  [[nodiscard]] T& at(IntType index) {
    if (shape_.rank() == 1) {
      int64_t coord = static_cast<int64_t>(index);
      return at<T>(std::span<const int64_t>(&coord, 1));
    }
    return flatAt<T>(static_cast<int64_t>(index));
  }

  template <typename T, std::integral IntType>
  [[nodiscard]] const T& at(IntType index) const {
    if (shape_.rank() == 1) {
      int64_t coord = static_cast<int64_t>(index);
      return at<T>(std::span<const int64_t>(&coord, 1));
    }
    return flatAt<T>(static_cast<int64_t>(index));
  }

  template <typename T>
  [[nodiscard]] T& flatAt(int64_t flat_index) {
    int64_t total = numElements();
    if (flat_index < 0 || flat_index >= total) {
      throw std::out_of_range("Tensor::flatAt(): flat index out of bounds");
    }
    if (isContiguous()) {
      return data<T>()[flat_index];
    }
    size_t r = shape_.rank();
    int64_t rem = flat_index;
    int64_t strided_offset = 0;
    for (size_t i = 0; i < r; ++i) {
      int64_t stride_i = 1;
      for (size_t j = i + 1; j < r; ++j) {
        stride_i *= shape_[j];
      }
      int64_t coord = rem / stride_i;
      rem %= stride_i;
      strided_offset += coord * strides_[i];
    }
    return *reinterpret_cast<T*>(static_cast<std::byte*>(data_) +
                                 static_cast<size_t>(strided_offset) * elementSize());
  }

  template <typename T>
  [[nodiscard]] const T& flatAt(int64_t flat_index) const {
    return const_cast<Tensor*>(this)->flatAt<T>(flat_index);
  }

  // Tensor transformations
  [[nodiscard]] Result<Tensor> reshape(const Shape& new_shape) const;
  [[nodiscard]] Result<Tensor> slice(size_t dim, int64_t start, int64_t end,
                                     int64_t step = 1) const;
  [[nodiscard]] Result<Tensor> slice(int64_t start, int64_t end) const {
    return slice(0, start, end, 1);
  }

  [[nodiscard]] Result<Tensor> toContiguous(ArenaAllocator& arena) const;
  [[nodiscard]] Result<Tensor> toContiguous() const;
  [[nodiscard]] Tensor clone(ArenaAllocator& arena) const;
  [[nodiscard]] Tensor clone() const;

  void zero() noexcept;

  template <typename T>
  void fill(T value) {
    if (!isValid())
      return;
    int64_t total = numElements();
    if (isContiguous()) {
      T* ptr = data<T>();
      for (int64_t i = 0; i < total; ++i) {
        ptr[i] = value;
      }
    } else {
      for (int64_t i = 0; i < total; ++i) {
        at<T>(i) = value;
      }
    }
  }

  [[nodiscard]] std::string toString() const;

private:
  template <typename T>
  [[nodiscard]] T* elementAddress(std::span<const int64_t> indices) const {
    if (indices.size() != shape_.rank()) {
      throw std::out_of_range("Tensor::at(): index rank mismatch");
    }
    int64_t offset = 0;
    for (size_t d = 0; d < shape_.rank(); ++d) {
      int64_t idx = indices[d];
      if (idx < 0 || idx >= shape_[d]) {
        throw std::out_of_range("Tensor::at(): index out of bounds");
      }
      offset += idx * strides_[d];
    }
    return reinterpret_cast<T*>(static_cast<std::byte*>(data_) +
                                static_cast<size_t>(offset) * elementSize());
  }

  Shape shape_;
  DataType dtype_{DataType::Float32};
  std::vector<int64_t> strides_;
  size_t byte_offset_{0};
  void* data_{nullptr};
  bool is_contiguous_{true};
  std::shared_ptr<TensorStorage> storage_{nullptr};
};

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
  return os << tensor.toString();
}

}  // namespace pyreth
