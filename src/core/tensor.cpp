#include <algorithm>
#include <new>
#include <pyreth/tensor.hpp>
#include <sstream>

namespace pyreth {

// ---------------------------------------------------------------------------
// Strides and Layout Helpers
// ---------------------------------------------------------------------------

std::vector<int64_t> computeContiguousStrides(const Shape& shape) {
  size_t r = shape.rank();
  if (r == 0) {
    return {};
  }
  std::vector<int64_t> strides(r, 1);
  for (int64_t i = static_cast<int64_t>(r) - 2; i >= 0; --i) {
    size_t cur_idx = static_cast<size_t>(i);
    size_t next_idx = static_cast<size_t>(i + 1);
    strides[cur_idx] = strides[next_idx] * std::max<int64_t>(shape[next_idx], 1);
  }
  return strides;
}

bool isContiguousLayout(const Shape& shape, const std::vector<int64_t>& strides) {
  size_t r = shape.rank();
  if (r == 0 || shape.numElements() <= 1) {
    return true;
  }
  if (r != strides.size()) {
    return false;
  }
  int64_t expected_stride = 1;
  for (int64_t i = static_cast<int64_t>(r) - 1; i >= 0; --i) {
    size_t idx = static_cast<size_t>(i);
    if (shape[idx] == 1) {
      continue;
    }
    if (strides[idx] != expected_stride) {
      return false;
    }
    expected_stride *= shape[idx];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Shape Implementation
// ---------------------------------------------------------------------------

std::string Shape::toString() const {
  if (dims_.empty()) {
    return "[]";
  }
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < dims_.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << dims_[i];
  }
  ss << "]";
  return ss.str();
}

// ---------------------------------------------------------------------------
// Storage Implementations
// ---------------------------------------------------------------------------

OwnedStorage::OwnedStorage(size_t size_bytes) : size_bytes_(size_bytes) {
  if (size_bytes_ > 0) {
    data_ = ::operator new(size_bytes_, std::align_val_t(DEFAULT_ALIGNMENT));
  }
}

OwnedStorage::~OwnedStorage() {
  if (data_) {
    ::operator delete(data_, std::align_val_t(DEFAULT_ALIGNMENT));
    data_ = nullptr;
  }
}

ArenaStorage::ArenaStorage(ArenaAllocator& arena, size_t size_bytes)
    : arena_(arena), size_bytes_(size_bytes) {
  arena_.pin();
  if (size_bytes_ > 0) {
    data_ = arena_.allocate(size_bytes_, DEFAULT_ALIGNMENT);
  }
}

ArenaStorage::~ArenaStorage() {
  arena_.unpin();
}

PooledArenaStorage::PooledArenaStorage(MemoryPool::Lease lease, size_t size_bytes)
    : lease_(std::move(lease)), size_bytes_(size_bytes) {
  if (size_bytes_ > 0 && lease_.valid()) {
    data_ = lease_->allocate(size_bytes_, DEFAULT_ALIGNMENT);
  }
}

ExternalStorage::ExternalStorage(void* data, size_t size_bytes, std::shared_ptr<void> owner)
    : data_(data), size_bytes_(size_bytes), owner_(std::move(owner)) {}

// ---------------------------------------------------------------------------
// Tensor Lifecycle Implementation
// ---------------------------------------------------------------------------

Tensor::Tensor(Shape shape, DataType dtype, std::shared_ptr<TensorStorage> storage,
               std::vector<int64_t> strides, size_t byte_offset)
    : shape_(std::move(shape)),
      dtype_(dtype),
      strides_(std::move(strides)),
      byte_offset_(byte_offset),
      storage_(std::move(storage)) {
  if (strides_.empty() && shape_.rank() > 0) {
    strides_ = computeContiguousStrides(shape_);
  }
  is_contiguous_ = isContiguousLayout(shape_, strides_);
  if (storage_) {
    data_ = static_cast<std::byte*>(storage_->data()) + byte_offset_;
  }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      dtype_(other.dtype_),
      strides_(other.strides_),
      byte_offset_(other.byte_offset_),
      data_(other.data_),
      is_contiguous_(other.is_contiguous_),
      storage_(other.storage_) {}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    shape_ = other.shape_;
    dtype_ = other.dtype_;
    strides_ = other.strides_;
    byte_offset_ = other.byte_offset_;
    data_ = other.data_;
    is_contiguous_ = other.is_contiguous_;
    storage_ = other.storage_;
  }
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      strides_(std::move(other.strides_)),
      byte_offset_(other.byte_offset_),
      data_(other.data_),
      is_contiguous_(other.is_contiguous_),
      storage_(std::move(other.storage_)) {
  other.data_ = nullptr;
  other.byte_offset_ = 0;
  other.is_contiguous_ = true;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    strides_ = std::move(other.strides_);
    byte_offset_ = other.byte_offset_;
    data_ = other.data_;
    is_contiguous_ = other.is_contiguous_;
    storage_ = std::move(other.storage_);

    other.data_ = nullptr;
    other.byte_offset_ = 0;
    other.is_contiguous_ = true;
  }
  return *this;
}

// ---------------------------------------------------------------------------
// Tensor Factory Methods
// ---------------------------------------------------------------------------

Result<Tensor> Tensor::create(const Shape& shape, DataType dtype) {
  size_t bytes = static_cast<size_t>(shape.numElements()) * dataTypeSize(dtype);
  try {
    auto storage = std::make_shared<OwnedStorage>(bytes);
    return Result<Tensor>::success(Tensor(shape, dtype, std::move(storage)));
  } catch (...) {
    return Result<Tensor>::failure(StatusCode::ALLOCATION_FAILED,
                                   "Failed to allocate tensor memory");
  }
}

Result<Tensor> Tensor::create(const Shape& shape, DataType dtype, ArenaAllocator& arena) {
  size_t bytes = static_cast<size_t>(shape.numElements()) * dataTypeSize(dtype);
  try {
    auto storage = std::make_shared<ArenaStorage>(arena, bytes);
    return Result<Tensor>::success(Tensor(shape, dtype, std::move(storage)));
  } catch (...) {
    return Result<Tensor>::failure(StatusCode::ALLOCATION_FAILED,
                                   "Failed to allocate tensor from arena");
  }
}

Result<Tensor> Tensor::create(const Shape& shape, DataType dtype, MemoryPool& pool) {
  auto lease = pool.acquire();
  return create(shape, dtype, std::move(lease));
}

Result<Tensor> Tensor::create(const Shape& shape, DataType dtype, MemoryPool::Lease lease) {
  if (!lease.valid()) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT, "Invalid memory pool lease");
  }
  size_t bytes = static_cast<size_t>(shape.numElements()) * dataTypeSize(dtype);
  try {
    auto storage = std::make_shared<PooledArenaStorage>(std::move(lease), bytes);
    return Result<Tensor>::success(Tensor(shape, dtype, std::move(storage)));
  } catch (...) {
    return Result<Tensor>::failure(StatusCode::ALLOCATION_FAILED,
                                   "Failed to allocate tensor from pool lease");
  }
}

Result<Tensor> Tensor::fromBuffer(void* data, const Shape& shape, DataType dtype,
                                  std::vector<int64_t> strides, std::shared_ptr<void> owner) {
  if (!data && shape.numElements() > 0) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Null buffer data for non-empty tensor");
  }
  size_t bytes = static_cast<size_t>(shape.numElements()) * dataTypeSize(dtype);
  auto storage = std::make_shared<ExternalStorage>(data, bytes, std::move(owner));
  return Result<Tensor>::success(Tensor(shape, dtype, std::move(storage), std::move(strides), 0));
}

// ---------------------------------------------------------------------------
// Reshape and Slicing
// ---------------------------------------------------------------------------

Result<Tensor> Tensor::reshape(const Shape& target_shape) const {
  if (!isValid()) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT, "Cannot reshape invalid tensor");
  }
  if (!isContiguous()) {
    return Result<Tensor>::failure(
        StatusCode::INVALID_ARGUMENT,
        "Cannot reshape non-contiguous tensor without copying; call toContiguous() first");
  }

  int64_t total_elements = numElements();
  int negative_dim_index = -1;
  int64_t known_product = 1;
  std::vector<int64_t> resolved_dims;
  resolved_dims.reserve(target_shape.rank());

  for (size_t i = 0; i < target_shape.rank(); ++i) {
    int64_t d = target_shape[i];
    if (d < 0) {
      if (d != -1) {
        return Result<Tensor>::failure(
            StatusCode::INVALID_DIMENSIONS,
            "Invalid negative dimension in reshape (only -1 is allowed)");
      }
      if (negative_dim_index != -1) {
        return Result<Tensor>::failure(StatusCode::INVALID_DIMENSIONS,
                                       "Only one dimension can be -1 in reshape");
      }
      negative_dim_index = static_cast<int>(i);
      resolved_dims.push_back(-1);
    } else {
      known_product *= d;
      resolved_dims.push_back(d);
    }
  }

  if (negative_dim_index != -1) {
    if (known_product == 0) {
      return Result<Tensor>::failure(StatusCode::INVALID_DIMENSIONS,
                                     "Cannot infer dimension when known dimensions product is 0");
    }
    if (total_elements % known_product != 0) {
      return Result<Tensor>::failure(StatusCode::INVALID_DIMENSIONS,
                                     "Total elements not divisible by known dimensions in reshape");
    }
    resolved_dims[static_cast<size_t>(negative_dim_index)] = total_elements / known_product;
  }

  Shape resolved_shape(std::move(resolved_dims));
  if (resolved_shape.numElements() != total_elements) {
    return Result<Tensor>::failure(StatusCode::INVALID_DIMENSIONS,
                                   "Reshape target dimensions do not match tensor element count");
  }

  std::vector<int64_t> new_strides = computeContiguousStrides(resolved_shape);
  return Result<Tensor>::success(
      Tensor(resolved_shape, dtype_, storage_, std::move(new_strides), byte_offset_));
}

Result<Tensor> Tensor::slice(size_t dim, int64_t start, int64_t end, int64_t step) const {
  if (!isValid()) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT, "Cannot slice invalid tensor");
  }
  if (dim >= rank()) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT, "Slice dimension out of range");
  }
  int64_t dim_size = shape_[dim];
  if (start < 0)
    start += dim_size;
  if (end < 0)
    end += dim_size;
  if (step <= 0) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT,
                                   "Slice step must be strictly positive");
  }
  if (start < 0 || end > dim_size || start > end) {
    return Result<Tensor>::failure(StatusCode::INVALID_ARGUMENT, "Slice indices out of bounds");
  }

  std::vector<int64_t> new_dims = shape_.dims();
  new_dims[dim] = (start == end) ? 0 : (end - start + step - 1) / step;
  Shape new_shape(std::move(new_dims));

  std::vector<int64_t> new_strides = strides_;
  new_strides[dim] = strides_[dim] * step;

  size_t offset_delta = static_cast<size_t>(start * strides_[dim]) * elementSize();
  size_t new_byte_offset = byte_offset_ + offset_delta;

  return Result<Tensor>::success(
      Tensor(new_shape, dtype_, storage_, std::move(new_strides), new_byte_offset));
}

// ---------------------------------------------------------------------------
// Copy and Contiguous Transformation
// ---------------------------------------------------------------------------

namespace {

void copyElements(const Tensor& src, Tensor& dst) {
  if (src.isContiguous()) {
    std::memcpy(dst.data(), src.data(), src.totalBytes());
    return;
  }
  size_t r = src.rank();
  if (r == 0) {
    std::memcpy(dst.data(), src.data(), src.totalBytes());
    return;
  }
  int64_t total = src.numElements();
  std::vector<int64_t> coords(r, 0);
  size_t elem_size = src.elementSize();

  for (int64_t i = 0; i < total; ++i) {
    int64_t src_offset = 0;
    for (size_t d = 0; d < r; ++d) {
      src_offset += coords[d] * src.strides()[d];
    }
    const void* src_elem =
        static_cast<const std::byte*>(src.data()) + static_cast<size_t>(src_offset) * elem_size;
    void* dst_elem = static_cast<std::byte*>(dst.data()) + static_cast<size_t>(i) * elem_size;
    std::memcpy(dst_elem, src_elem, elem_size);

    for (int64_t d = static_cast<int64_t>(r) - 1; d >= 0; --d) {
      size_t dim_idx = static_cast<size_t>(d);
      if (++coords[dim_idx] < src.shape()[dim_idx]) {
        break;
      }
      coords[dim_idx] = 0;
    }
  }
}

}  // namespace

Result<Tensor> Tensor::toContiguous(ArenaAllocator& arena) const {
  if (isContiguous()) {
    return Result<Tensor>::success(*this);
  }
  auto res = create(shape_, dtype_, arena);
  if (!res.isOk()) {
    return res;
  }
  Tensor contig = std::move(res).value();
  copyElements(*this, contig);
  return Result<Tensor>::success(std::move(contig));
}

Result<Tensor> Tensor::toContiguous() const {
  if (isContiguous()) {
    return Result<Tensor>::success(*this);
  }
  auto res = create(shape_, dtype_);
  if (!res.isOk()) {
    return res;
  }
  Tensor contig = std::move(res).value();
  copyElements(*this, contig);
  return Result<Tensor>::success(std::move(contig));
}

Tensor Tensor::clone(ArenaAllocator& arena) const {
  auto res = create(shape_, dtype_, arena);
  if (!res.isOk()) {
    throw std::bad_alloc();
  }
  Tensor cloned = std::move(res).value();
  copyElements(*this, cloned);
  return cloned;
}

Tensor Tensor::clone() const {
  auto res = create(shape_, dtype_);
  if (!res.isOk()) {
    throw std::bad_alloc();
  }
  Tensor cloned = std::move(res).value();
  copyElements(*this, cloned);
  return cloned;
}

void Tensor::zero() noexcept {
  if (!isValid())
    return;
  if (isContiguous()) {
    std::memset(data_, 0, totalBytes());
  } else {
    fill<int8_t>(0);
  }
}

std::string Tensor::toString() const {
  std::ostringstream ss;
  ss << "Tensor(shape=" << shape_.toString() << ", dtype=" << dataTypeToString(dtype_)
     << ", contiguous=" << (is_contiguous_ ? "true" : "false") << ", offset=" << byte_offset_
     << ", storage=" << storageTypeToString(storageType()) << ")";
  return ss.str();
}

}  // namespace pyreth
