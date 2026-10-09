#include <gtest/gtest.h>

#include <memory>
#include <numeric>
#include <pyreth/memory_pool.hpp>
#include <pyreth/pyreth.hpp>
#include <pyreth/tensor.hpp>
#include <vector>

using namespace pyreth;

// ===========================================================================
// 1. DataType Tests
// ===========================================================================

TEST(TensorTest, DataTypeSizesAndNames) {
  EXPECT_EQ(dataTypeSize(DataType::Float32), 4);
  EXPECT_EQ(dataTypeSize(DataType::Float16), 2);
  EXPECT_EQ(dataTypeSize(DataType::BFloat16), 2);
  EXPECT_EQ(dataTypeSize(DataType::Float64), 8);
  EXPECT_EQ(dataTypeSize(DataType::Int8), 1);
  EXPECT_EQ(dataTypeSize(DataType::UInt8), 1);
  EXPECT_EQ(dataTypeSize(DataType::Int16), 2);
  EXPECT_EQ(dataTypeSize(DataType::UInt16), 2);
  EXPECT_EQ(dataTypeSize(DataType::Int32), 4);
  EXPECT_EQ(dataTypeSize(DataType::UInt32), 4);
  EXPECT_EQ(dataTypeSize(DataType::Int64), 8);
  EXPECT_EQ(dataTypeSize(DataType::UInt64), 8);
  EXPECT_EQ(dataTypeSize(DataType::Bool), 1);

  EXPECT_EQ(dataTypeToString(DataType::Float32), "Float32");
  EXPECT_EQ(dataTypeToString(DataType::Float16), "Float16");
  EXPECT_EQ(dataTypeToString(DataType::BFloat16), "BFloat16");
  EXPECT_EQ(dataTypeToString(DataType::Int64), "Int64");
  EXPECT_EQ(dataTypeToString(DataType::Bool), "Bool");
}

TEST(TensorTest, DataTypeTraitsMapping) {
  EXPECT_EQ(dataTypeOf<float>(), DataType::Float32);
  EXPECT_EQ(dataTypeOf<double>(), DataType::Float64);
  EXPECT_EQ(dataTypeOf<int8_t>(), DataType::Int8);
  EXPECT_EQ(dataTypeOf<uint8_t>(), DataType::UInt8);
  EXPECT_EQ(dataTypeOf<int16_t>(), DataType::Int16);
  EXPECT_EQ(dataTypeOf<uint16_t>(), DataType::UInt16);
  EXPECT_EQ(dataTypeOf<int32_t>(), DataType::Int32);
  EXPECT_EQ(dataTypeOf<uint32_t>(), DataType::UInt32);
  EXPECT_EQ(dataTypeOf<int64_t>(), DataType::Int64);
  EXPECT_EQ(dataTypeOf<uint64_t>(), DataType::UInt64);
  EXPECT_EQ(dataTypeOf<bool>(), DataType::Bool);
}

// ===========================================================================
// 2. Shape Tests
// ===========================================================================

TEST(TensorTest, ShapeConstructionAndProperties) {
  Shape scalar;
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.ndim(), 0);
  EXPECT_EQ(scalar.numElements(), 1);
  EXPECT_TRUE(scalar.isScalar());
  EXPECT_FALSE(scalar.empty());
  EXPECT_EQ(scalar.toString(), "[]");

  Shape s1({1, 3, 224, 224});
  EXPECT_EQ(s1.rank(), 4);
  EXPECT_EQ(s1.ndim(), 4);
  EXPECT_EQ(s1.numElements(), 1 * 3 * 224 * 224);
  EXPECT_EQ(s1[0], 1);
  EXPECT_EQ(s1[1], 3);
  EXPECT_EQ(s1[2], 224);
  EXPECT_EQ(s1[3], 224);
  EXPECT_FALSE(s1.isScalar());
  EXPECT_FALSE(s1.empty());
  EXPECT_EQ(s1.toString(), "[1, 3, 224, 224]");

  Shape s2(std::vector<int64_t>{1, 3, 224, 224});
  EXPECT_EQ(s1, s2);

  Shape s3({2, 3, 4});
  EXPECT_NE(s1, s3);

  Shape empty_shape({0, 10});
  EXPECT_EQ(empty_shape.numElements(), 0);
  EXPECT_TRUE(empty_shape.empty());
}

// ===========================================================================
// 3. Strides and Contiguity Tests
// ===========================================================================

TEST(TensorTest, ContiguousStridesCalculation) {
  Shape s1({2, 3, 4});
  auto strides1 = computeContiguousStrides(s1);
  ASSERT_EQ(strides1.size(), 3);
  EXPECT_EQ(strides1[0], 12);  // 3 * 4
  EXPECT_EQ(strides1[1], 4);   // 4
  EXPECT_EQ(strides1[2], 1);   // 1
  EXPECT_TRUE(isContiguousLayout(s1, strides1));

  Shape scalar;
  auto strides_scalar = computeContiguousStrides(scalar);
  EXPECT_TRUE(strides_scalar.empty());
  EXPECT_TRUE(isContiguousLayout(scalar, strides_scalar));

  // Non-contiguous strides: transposed matrix
  Shape matrix({3, 4});
  std::vector<int64_t> transposed_strides = {1, 3};  // Column-major
  EXPECT_FALSE(isContiguousLayout(matrix, transposed_strides));
}

// ===========================================================================
// 4. Standalone Owned Tensor Allocation & SIMD Alignment
// ===========================================================================

TEST(TensorTest, OwnedTensorAllocationAndSimdAlignment) {
  auto res = Tensor::create({4, 16}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor tensor = std::move(res).value();

  EXPECT_TRUE(tensor.isValid());
  EXPECT_EQ(tensor.shape(), Shape({4, 16}));
  EXPECT_EQ(tensor.dtype(), DataType::Float32);
  EXPECT_EQ(tensor.numElements(), 64);
  EXPECT_EQ(tensor.totalBytes(), 64 * sizeof(float));
  EXPECT_EQ(tensor.elementSize(), sizeof(float));
  EXPECT_TRUE(tensor.isContiguous());
  EXPECT_EQ(tensor.storageType(), StorageType::OWNED);
  EXPECT_FALSE(tensor.isArenaBacked());

  // Strict 64-byte SIMD alignment check
  EXPECT_TRUE(tensor.isAligned(64));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(tensor.data()) % 64, 0);

  // Fill and indexed access
  tensor.fill(3.14f);
  for (int64_t i = 0; i < 4; ++i) {
    for (int64_t j = 0; j < 16; ++j) {
      EXPECT_FLOAT_EQ(tensor.at<float>(i, j), 3.14f);
    }
  }

  // Linear flat access
  EXPECT_FLOAT_EQ(tensor.at<float>(0), 3.14f);
  EXPECT_FLOAT_EQ(tensor.at<float>(63), 3.14f);

  // Zeroing
  tensor.zero();
  EXPECT_FLOAT_EQ(tensor.at<float>(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(tensor.at<float>(3, 15), 0.0f);
}

// ===========================================================================
// 5. Arena-Backed Tensor & Lifetime Pinning Contract
// ===========================================================================

TEST(TensorTest, ArenaBackedTensorPinningAndLifetimeContract) {
  ArenaAllocator arena(1024 * 1024);
  EXPECT_FALSE(arena.isPinned());
  EXPECT_TRUE(arena.canReset());
  EXPECT_EQ(arena.pinCount(), 0);

  {
    auto res = Tensor::create({8, 8}, DataType::Float32, arena);
    ASSERT_TRUE(res.isOk());
    Tensor tensor = std::move(res).value();

    EXPECT_TRUE(tensor.isValid());
    EXPECT_TRUE(tensor.isArenaBacked());
    EXPECT_EQ(tensor.storageType(), StorageType::ARENA);
    EXPECT_TRUE(tensor.isAligned(64));

    // Arena must be pinned while tensor is alive
    EXPECT_TRUE(arena.isPinned());
    EXPECT_FALSE(arena.canReset());
    EXPECT_EQ(arena.pinCount(), 1);

    // Initial write
    tensor.fill(42.0f);
    EXPECT_FLOAT_EQ(tensor.at<float>(0, 0), 42.0f);

    // Attempting reset while pinned MUST fail and protect buffer!
    auto try_reset = arena.tryReset();
    EXPECT_FALSE(try_reset.isOk());
    EXPECT_EQ(try_reset.error().code(), StatusCode::INVALID_ARGUMENT);

    // Calling arena.reset() is a safe no-op while pinned
    arena.reset();
    EXPECT_FLOAT_EQ(tensor.at<float>(0, 0), 42.0f);  // Memory still intact!

    // Creating a slice also references the arena storage
    {
      auto slice_res = tensor.slice(0, 0, 4);
      ASSERT_TRUE(slice_res.isOk());
      Tensor slice = std::move(slice_res).value();
      EXPECT_TRUE(arena.isPinned());
      EXPECT_EQ(arena.pinCount(), 1);  // Same underlying ArenaStorage
      EXPECT_EQ(tensor.useCount(), 2);
    }

    // Slice out of scope, tensor still alive
    EXPECT_TRUE(arena.isPinned());
    EXPECT_EQ(tensor.useCount(), 1);
  }

  // Tensor is destroyed -> arena is automatically unpinned!
  EXPECT_FALSE(arena.isPinned());
  EXPECT_TRUE(arena.canReset());
  EXPECT_EQ(arena.pinCount(), 0);

  // Now reset succeeds cleanly
  auto reset_res = arena.tryReset();
  EXPECT_TRUE(reset_res.isOk());
}

// ===========================================================================
// 6. MemoryPool-Backed Tensor & Automatic Lease Recycling
// ===========================================================================

TEST(TensorTest, MemoryPoolBackedTensorAutomaticLeaseRecycling) {
  MemoryPool pool(/*initial_count=*/2, /*block_size=*/1024 * 1024);
  EXPECT_EQ(pool.availableCount(), 2);

  {
    auto res = Tensor::create({10, 10}, DataType::Float32, pool);
    ASSERT_TRUE(res.isOk());
    Tensor t1 = std::move(res).value();

    EXPECT_TRUE(t1.isValid());
    EXPECT_EQ(t1.storageType(), StorageType::POOLED);
    EXPECT_TRUE(t1.isArenaBacked());
    EXPECT_TRUE(t1.isAligned(64));

    // One arena leased from pool
    EXPECT_EQ(pool.availableCount(), 1);

    t1.fill(100.0f);
    EXPECT_FLOAT_EQ(t1.at<float>(0, 0), 100.0f);

    {
      // Slicing t1 shares the leased arena storage
      auto slice_res = t1.slice(0, 0, 5);
      ASSERT_TRUE(slice_res.isOk());
      Tensor slice = std::move(slice_res).value();

      EXPECT_EQ(t1.useCount(), 2);
      EXPECT_EQ(pool.availableCount(), 1);  // Lease still held
    }

    // Slice destroyed, t1 still alive -> lease still held
    EXPECT_EQ(pool.availableCount(), 1);
  }

  // t1 destroyed -> lease RAII returns arena to pool in O(1)
  EXPECT_EQ(pool.availableCount(), 2);
}

// ===========================================================================
// 7. External Buffer (Zero-Copy View with Custom Owner)
// ===========================================================================

TEST(TensorTest, ExternalBufferZeroCopyViewAndOwnerRetention) {
  std::vector<float> raw_buffer(20, 1.23f);
  auto keepalive_owner = std::make_shared<int>(999);

  {
    auto res = Tensor::fromBuffer(raw_buffer.data(), Shape({4, 5}), DataType::Float32, {},
                                  keepalive_owner);
    ASSERT_TRUE(res.isOk());
    Tensor view = std::move(res).value();

    EXPECT_TRUE(view.isValid());
    EXPECT_EQ(view.storageType(), StorageType::EXTERNAL);
    EXPECT_EQ(view.data(), raw_buffer.data());
    EXPECT_FLOAT_EQ(view.at<float>(0, 0), 1.23f);
    EXPECT_EQ(keepalive_owner.use_count(), 2);

    // Mutate view in-place -> modifies raw buffer
    view.at<float>(1, 2) = 4.56f;
    EXPECT_FLOAT_EQ(raw_buffer[1 * 5 + 2], 4.56f);
  }

  // View destroyed -> owner refcount drops
  EXPECT_EQ(keepalive_owner.use_count(), 1);
}

// ===========================================================================
// 8. Reshaping Tests
// ===========================================================================

TEST(TensorTest, ReshapingContiguousTensor) {
  auto res = Tensor::create({2, 3, 4}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor tensor = std::move(res).value();

  // Populate sequentially 0..23
  for (int64_t i = 0; i < tensor.numElements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i);
  }

  // Reshape to [6, 4]
  auto r1 = tensor.reshape(Shape({6, 4}));
  ASSERT_TRUE(r1.isOk());
  Tensor t_6x4 = std::move(r1).value();
  EXPECT_EQ(t_6x4.shape(), Shape({6, 4}));
  EXPECT_EQ(t_6x4.data(), tensor.data());  // Zero-copy, same data pointer!
  EXPECT_FLOAT_EQ(t_6x4.at<float>(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(t_6x4.at<float>(5, 3), 23.0f);

  // Reshape to 1D flat [24]
  auto r2 = tensor.reshape(Shape({24}));
  ASSERT_TRUE(r2.isOk());
  EXPECT_EQ(r2.value().shape(), Shape({24}));

  // Reshape with dimension inference (-1)
  auto r3 = tensor.reshape(Shape({-1, 12}));
  ASSERT_TRUE(r3.isOk());
  EXPECT_EQ(r3.value().shape(), Shape({2, 12}));

  auto r4 = tensor.reshape(Shape({4, -1}));
  ASSERT_TRUE(r4.isOk());
  EXPECT_EQ(r4.value().shape(), Shape({4, 6}));

  // Invalid reshape: element count mismatch
  auto r_invalid = tensor.reshape(Shape({5, 5}));
  EXPECT_FALSE(r_invalid.isOk());
  EXPECT_EQ(r_invalid.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Invalid reshape: multiple -1 dimensions
  auto r_multi_neg = tensor.reshape(Shape({-1, -1}));
  EXPECT_FALSE(r_multi_neg.isOk());
  EXPECT_EQ(r_multi_neg.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Invalid reshape: indivisible -1
  auto r_indivisible = tensor.reshape(Shape({-1, 7}));
  EXPECT_FALSE(r_indivisible.isOk());
  EXPECT_EQ(r_indivisible.error().code(), StatusCode::INVALID_DIMENSIONS);
}

// ===========================================================================
// 9. Contiguous Slicing Tests
// ===========================================================================

TEST(TensorTest, ContiguousSlicingDimension0) {
  auto res = Tensor::create({4, 8}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor tensor = std::move(res).value();

  for (int64_t i = 0; i < tensor.numElements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i);
  }

  // Slicing dimension 0 from rows 1 to 3 -> shape [2, 8]
  auto slice_res = tensor.slice(0, 1, 3);
  ASSERT_TRUE(slice_res.isOk());
  Tensor slice = std::move(slice_res).value();

  EXPECT_EQ(slice.shape(), Shape({2, 8}));
  EXPECT_EQ(slice.numElements(), 16);
  EXPECT_EQ(slice.byteOffset(), 1 * 8 * sizeof(float));
  EXPECT_TRUE(slice.isContiguous());  // Dim 0 slice of contiguous matrix IS contiguous!

  // Check element mapping: row 0 of slice is row 1 of tensor (values 8..15)
  EXPECT_FLOAT_EQ(slice.at<float>(0, 0), 8.0f);
  EXPECT_FLOAT_EQ(slice.at<float>(0, 7), 15.0f);
  EXPECT_FLOAT_EQ(slice.at<float>(1, 0), 16.0f);
  EXPECT_FLOAT_EQ(slice.at<float>(1, 7), 23.0f);

  // In-place mutation through slice updates original tensor
  slice.at<float>(0, 0) = 999.0f;
  EXPECT_FLOAT_EQ(tensor.at<float>(1, 0), 999.0f);
}

TEST(TensorTest, NonContiguousSlicingInnerDimension) {
  auto res = Tensor::create({4, 8}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor tensor = std::move(res).value();

  for (int64_t i = 0; i < tensor.numElements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i);
  }

  // Slicing columns 2 to 6 on dimension 1 -> shape [4, 4]
  auto slice_res = tensor.slice(1, 2, 6);
  ASSERT_TRUE(slice_res.isOk());
  Tensor col_slice = std::move(slice_res).value();

  EXPECT_EQ(col_slice.shape(), Shape({4, 4}));
  EXPECT_EQ(col_slice.numElements(), 16);
  EXPECT_FALSE(col_slice.isContiguous());  // Strided column slice is non-contiguous!

  // Check values
  EXPECT_FLOAT_EQ(col_slice.at<float>(0, 0), 2.0f);   // tensor[0, 2]
  EXPECT_FLOAT_EQ(col_slice.at<float>(0, 3), 5.0f);   // tensor[0, 5]
  EXPECT_FLOAT_EQ(col_slice.at<float>(1, 0), 10.0f);  // tensor[1, 2] = 1*8 + 2

  // Non-contiguous tensor cannot be reshaped without copying
  auto reshape_fail = col_slice.reshape(Shape({2, 8}));
  EXPECT_FALSE(reshape_fail.isOk());
  EXPECT_EQ(reshape_fail.error().code(), StatusCode::INVALID_ARGUMENT);

  // Slice bounds validation
  auto oob_dim = tensor.slice(3, 0, 1);
  EXPECT_FALSE(oob_dim.isOk());

  auto oob_range = tensor.slice(0, 5, 6);
  EXPECT_FALSE(oob_range.isOk());

  auto inverted_range = tensor.slice(0, 3, 1);
  EXPECT_FALSE(inverted_range.isOk());
}

// ===========================================================================
// 10. toContiguous() and clone() Tests
// ===========================================================================

TEST(TensorTest, ToContiguousAndClone) {
  auto res = Tensor::create({4, 8}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor tensor = std::move(res).value();

  for (int64_t i = 0; i < tensor.numElements(); ++i) {
    tensor.at<float>(i) = static_cast<float>(i);
  }

  // Non-contiguous slice [4, 4]
  Tensor col_slice = tensor.slice(1, 2, 6).value();
  ASSERT_FALSE(col_slice.isContiguous());

  // Convert to contiguous layout
  auto contig_res = col_slice.toContiguous();
  ASSERT_TRUE(contig_res.isOk());
  Tensor contig = std::move(contig_res).value();

  EXPECT_TRUE(contig.isContiguous());
  EXPECT_EQ(contig.shape(), Shape({4, 4}));
  EXPECT_NE(contig.data(), col_slice.data());  // Newly allocated contiguous memory

  // Verify elements copied in coordinate order
  for (int64_t r = 0; r < 4; ++r) {
    for (int64_t c = 0; c < 4; ++c) {
      EXPECT_FLOAT_EQ(contig.at<float>(r, c), col_slice.at<float>(r, c));
    }
  }

  // Mutating contig does not mutate original tensor
  contig.at<float>(0, 0) = -999.0f;
  EXPECT_FLOAT_EQ(tensor.at<float>(0, 2), 2.0f);

  // clone() creates independent copy
  Tensor cloned = tensor.clone();
  EXPECT_EQ(cloned.shape(), tensor.shape());
  EXPECT_NE(cloned.data(), tensor.data());
  EXPECT_FLOAT_EQ(cloned.at<float>(0, 0), tensor.at<float>(0, 0));

  cloned.at<float>(0, 0) = 888.0f;
  EXPECT_FLOAT_EQ(tensor.at<float>(0, 0), 0.0f);
}

// ===========================================================================
// 11. Move Semantics & Memory Ownership Transfer
// ===========================================================================

TEST(TensorTest, MoveSemanticsAndOwnershipTransfer) {
  auto res = Tensor::create({3, 3}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor t1 = std::move(res).value();
  t1.fill(7.0f);
  void* original_ptr = t1.data();

  // Move constructor
  Tensor t2 = std::move(t1);
  EXPECT_TRUE(t2.isValid());
  EXPECT_EQ(t2.data(), original_ptr);
  EXPECT_FLOAT_EQ(t2.at<float>(0, 0), 7.0f);

  // Moved-from t1 is invalidated
  EXPECT_FALSE(t1.isValid());
  EXPECT_EQ(t1.data(), nullptr);

  // Move assignment
  Tensor t3;
  EXPECT_FALSE(t3.isValid());
  t3 = std::move(t2);
  EXPECT_TRUE(t3.isValid());
  EXPECT_EQ(t3.data(), original_ptr);
  EXPECT_FALSE(t2.isValid());
}

TEST(TensorTest, ShallowCopySharedViewSemantics) {
  auto res = Tensor::create({2, 2}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor t1 = std::move(res).value();
  t1.fill(1.0f);

  // Copy constructor creates shallow view
  Tensor t2 = t1;
  EXPECT_EQ(t1.useCount(), 2);
  EXPECT_EQ(t2.useCount(), 2);
  EXPECT_EQ(t1.data(), t2.data());

  // Mutating t2 modifies t1
  t2.at<float>(0, 1) = 5.0f;
  EXPECT_FLOAT_EQ(t1.at<float>(0, 1), 5.0f);
}

TEST(TensorTest, OneDimensionalAndScalarTensors) {
  // 1D Tensor
  auto res_1d = Tensor::create({5}, DataType::Float32);
  ASSERT_TRUE(res_1d.isOk());
  Tensor t_1d = std::move(res_1d).value();
  EXPECT_EQ(t_1d.rank(), 1);
  EXPECT_EQ(t_1d.numElements(), 5);
  for (int64_t i = 0; i < 5; ++i) {
    t_1d.at<float>(i) = static_cast<float>(i * 10);
  }
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(t_1d.at<float>(i), static_cast<float>(i * 10));
    EXPECT_FLOAT_EQ(t_1d.flatAt<float>(i), static_cast<float>(i * 10));
  }

  // Scalar Tensor (rank 0)
  auto res_scalar = Tensor::create({}, DataType::Float32);
  ASSERT_TRUE(res_scalar.isOk());
  Tensor scalar = std::move(res_scalar).value();
  EXPECT_EQ(scalar.rank(), 0);
  EXPECT_EQ(scalar.numElements(), 1);
  EXPECT_TRUE(scalar.shape().isScalar());
  scalar.fill(123.456f);
  EXPECT_FLOAT_EQ(scalar.flatAt<float>(0), 123.456f);
}

TEST(TensorTest, MultipleArenaBackedTensorsConcurrentPinning) {
  ArenaAllocator arena(1024 * 1024);
  EXPECT_EQ(arena.pinCount(), 0);

  auto res1 = Tensor::create({4, 4}, DataType::Float32, arena);
  ASSERT_TRUE(res1.isOk());
  Tensor t1 = std::move(res1).value();
  EXPECT_EQ(arena.pinCount(), 1);

  {
    auto res2 = Tensor::create({8, 8}, DataType::Int64, arena);
    ASSERT_TRUE(res2.isOk());
    Tensor t2 = std::move(res2).value();
    EXPECT_EQ(arena.pinCount(), 2);

    // Try reset fails with 2 pins
    EXPECT_FALSE(arena.tryReset().isOk());
  }

  // t2 destroyed -> pinCount drops to 1
  EXPECT_EQ(arena.pinCount(), 1);
  EXPECT_FALSE(arena.tryReset().isOk());

  // t1 cloned into the same arena
  {
    Tensor cloned_in_arena = t1.clone(arena);
    EXPECT_EQ(arena.pinCount(), 2);
  }
  EXPECT_EQ(arena.pinCount(), 1);

  // Destruct t1
  t1 = Tensor();  // Invalidated
  EXPECT_EQ(arena.pinCount(), 0);
  EXPECT_TRUE(arena.tryReset().isOk());
}

TEST(TensorTest, ToStringOutput) {
  auto res = Tensor::create({2, 3}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor t = std::move(res).value();
  std::string s = t.toString();
  EXPECT_NE(s.find("Tensor(shape=[2, 3]"), std::string::npos);
  EXPECT_NE(s.find("dtype=Float32"), std::string::npos);
  EXPECT_NE(s.find("contiguous=true"), std::string::npos);
  EXPECT_NE(s.find("storage=OWNED"), std::string::npos);
}

TEST(TensorTest, SelfAssignmentSafety) {
  auto res = Tensor::create({2, 2}, DataType::Float32);
  ASSERT_TRUE(res.isOk());
  Tensor t = std::move(res).value();
  t.fill(9.0f);

  // Copy self-assignment
  Tensor& t_ref = t;
  t = t_ref;
  EXPECT_TRUE(t.isValid());
  EXPECT_FLOAT_EQ(t.at<float>(0, 0), 9.0f);

  // Move self-assignment
  t = std::move(t);
  EXPECT_TRUE(t.isValid());
  EXPECT_FLOAT_EQ(t.at<float>(0, 0), 9.0f);
}
