#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <pyreth/inference_session.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/ort_bridge.hpp>
#include <pyreth/tensor.hpp>
#include <pyreth/thread_pool.hpp>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using namespace pyreth;

// ---------------------------------------------------------------------------
// ONNX Protobuf Builder Helpers (Supporting Initializers & Multi-Layer Graphs)
// ---------------------------------------------------------------------------

enum class OnnxDataType : uint32_t {
  FLOAT = 1,
  INT32 = 6,
  INT64 = 7,
};

struct DimSpec {
  int64_t value{-1};
  std::string param{};

  static DimSpec fixed(int64_t v) { return DimSpec{v, ""}; }
  static DimSpec dynamic(std::string sym = "") { return DimSpec{-1, std::move(sym)}; }
};

struct TensorSpec {
  std::string name;
  OnnxDataType elem_type;
  std::vector<DimSpec> dims;
};

class ProtoBuilder {
public:
  void writeVarint(uint64_t val) {
    while (val >= 0x80) {
      buf_.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
      val >>= 7;
    }
    buf_.push_back(static_cast<uint8_t>(val & 0x7F));
  }

  void writeTag(uint32_t field_number, uint8_t wire_type) {
    writeVarint((static_cast<uint64_t>(field_number) << 3) | wire_type);
  }

  void writeVarintField(uint32_t field_number, uint64_t val) {
    writeTag(field_number, 0);
    writeVarint(val);
  }

  void writeBytesField(uint32_t field_number, std::span<const uint8_t> bytes) {
    writeTag(field_number, 2);
    writeVarint(bytes.size());
    buf_.insert(buf_.end(), bytes.begin(), bytes.end());
  }

  void writeStringField(uint32_t field_number, std::string_view str) {
    writeTag(field_number, 2);
    writeVarint(str.size());
    buf_.insert(buf_.end(), reinterpret_cast<const uint8_t*>(str.data()),
                reinterpret_cast<const uint8_t*>(str.data()) + str.size());
  }

  void writeSubmessage(uint32_t field_number, const ProtoBuilder& sub) {
    writeBytesField(field_number, sub.bytes());
  }

  [[nodiscard]] const std::vector<uint8_t>& bytes() const noexcept { return buf_; }

private:
  std::vector<uint8_t> buf_;
};

ProtoBuilder encodeValueInfo(const TensorSpec& spec) {
  ProtoBuilder shape_proto;
  for (const auto& d : spec.dims) {
    ProtoBuilder dim_proto;
    if (!d.param.empty()) {
      dim_proto.writeStringField(2, d.param);
    } else if (d.value >= 0) {
      dim_proto.writeVarintField(1, static_cast<uint64_t>(d.value));
    }
    shape_proto.writeSubmessage(1, dim_proto);
  }

  ProtoBuilder tensor_type_proto;
  tensor_type_proto.writeVarintField(1, static_cast<uint64_t>(spec.elem_type));
  tensor_type_proto.writeSubmessage(2, shape_proto);

  ProtoBuilder type_proto;
  type_proto.writeSubmessage(1, tensor_type_proto);

  ProtoBuilder value_info;
  value_info.writeStringField(1, spec.name);
  value_info.writeSubmessage(2, type_proto);
  return value_info;
}

ProtoBuilder encodeNode(std::span<const std::string_view> inputs,
                        std::span<const std::string_view> outputs, std::string_view name,
                        std::string_view op_type) {
  ProtoBuilder node;
  for (auto in : inputs) {
    node.writeStringField(1, in);
  }
  for (auto out : outputs) {
    node.writeStringField(2, out);
  }
  node.writeStringField(3, name);
  node.writeStringField(4, op_type);
  return node;
}

// TensorProto initializer encoding:
// field 1: dims (repeated int64 varint)
// field 2: data_type (int32 varint, 1 = FLOAT)
// field 8: name (string)
// field 9: raw_data (bytes)
ProtoBuilder encodeFloatInitializer(std::string_view name, std::span<const int64_t> dims,
                                    std::span<const float> values) {
  ProtoBuilder tensor_proto;
  for (int64_t d : dims) {
    tensor_proto.writeVarintField(1, static_cast<uint64_t>(d));
  }
  tensor_proto.writeVarintField(2, static_cast<uint64_t>(OnnxDataType::FLOAT));
  tensor_proto.writeStringField(8, name);
  std::span<const uint8_t> raw_bytes(reinterpret_cast<const uint8_t*>(values.data()),
                                     values.size() * sizeof(float));
  tensor_proto.writeBytesField(9, raw_bytes);
  return tensor_proto;
}

std::vector<uint8_t> buildModelProto(std::string_view graph_name,
                                     std::span<const TensorSpec> inputs,
                                     std::span<const TensorSpec> outputs,
                                     std::span<const ProtoBuilder> nodes,
                                     std::span<const ProtoBuilder> initializers = {}) {
  ProtoBuilder graph;
  for (const auto& n : nodes) {
    graph.writeSubmessage(1, n);
  }
  graph.writeStringField(2, graph_name);
  for (const auto& init : initializers) {
    graph.writeSubmessage(5, init);  // field 5 = initializer (repeated TensorProto)
  }
  for (const auto& in : inputs) {
    graph.writeSubmessage(11, encodeValueInfo(in));
  }
  for (const auto& out : outputs) {
    graph.writeSubmessage(12, encodeValueInfo(out));
  }

  ProtoBuilder opset;
  opset.writeStringField(1, "");
  opset.writeVarintField(2, 17);

  ProtoBuilder model;
  model.writeVarintField(1, 9);  // ir_version = 9
  model.writeStringField(2, "pyreth_test");
  model.writeSubmessage(7, graph);
  model.writeSubmessage(8, opset);
  return model.bytes();
}

// 1. Identity model: y = x (Float32 [batch_size, 4])
std::vector<uint8_t> buildIdentityModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}}};
  std::string_view in_names[] = {"x"};
  std::string_view out_names[] = {"y"};
  std::vector<ProtoBuilder> nodes = {encodeNode(in_names, out_names, "id_node", "Identity")};
  return buildModelProto("identity_graph", inputs, outputs, nodes);
}

// 2. Dual-Input Dual-Output Arithmetic Model:
//    sum = a + b, prod = a * b (Float32 [batch_size, 4])
std::vector<uint8_t> buildArithmeticModel() {
  std::vector<TensorSpec> inputs = {
      {"a", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}},
      {"b", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}},
  };
  std::vector<TensorSpec> outputs = {
      {"sum", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}},
      {"prod", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}},
  };
  std::string_view add_in[] = {"a", "b"};
  std::string_view add_out[] = {"sum"};
  std::string_view mul_in[] = {"a", "b"};
  std::string_view mul_out[] = {"prod"};
  std::vector<ProtoBuilder> nodes = {
      encodeNode(add_in, add_out, "add_node", "Add"),
      encodeNode(mul_in, mul_out, "mul_node", "Mul"),
  };
  return buildModelProto("arithmetic_graph", inputs, outputs, nodes);
}

// 3. Real Multi-Layer MLP / Linear + Bias + ReLU ONNX Model:
//    y = Relu(MatMul(x, W) + b)
//    x: [batch_size, 4], W: [4, 3], b: [3] -> y: [batch_size, 3]
std::vector<uint8_t> buildMlpLinearReluModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(3)}}};

  // Weight matrix W (4x3, row-major):
  // [ 1.0, -1.0,  0.5]
  // [ 0.0,  2.0, -0.5]
  // [ 2.0,  0.0,  1.0]
  // [-1.0,  1.0,  2.0]
  const int64_t w_dims[] = {4, 3};
  const float w_vals[] = {
      1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -0.5f, 2.0f, 0.0f, 1.0f, -1.0f, 1.0f, 2.0f,
  };

  // Bias vector b (3):
  // [0.5, -1.0, 0.0]
  const int64_t b_dims[] = {3};
  const float b_vals[] = {0.5f, -1.0f, 0.0f};

  std::vector<ProtoBuilder> initializers = {
      encodeFloatInitializer("W", w_dims, w_vals),
      encodeFloatInitializer("b", b_dims, b_vals),
  };

  std::string_view mm_in[] = {"x", "W"};
  std::string_view mm_out[] = {"xw"};
  std::string_view add_in[] = {"xw", "b"};
  std::string_view add_out[] = {"xw_b"};
  std::string_view relu_in[] = {"xw_b"};
  std::string_view relu_out[] = {"y"};

  std::vector<ProtoBuilder> nodes = {
      encodeNode(mm_in, mm_out, "matmul_node", "MatMul"),
      encodeNode(add_in, add_out, "bias_add_node", "Add"),
      encodeNode(relu_in, relu_out, "relu_node", "Relu"),
  };

  return buildModelProto("mlp_linear_relu_graph", inputs, outputs, nodes, initializers);
}

// ---------------------------------------------------------------------------
// Zero-Copy Ort::Value Bridge Unit Tests
// ---------------------------------------------------------------------------

TEST(InferenceExecutionTest, ZeroCopyBridgeTensorToOrtValueSharesMemoryAddress) {
  ArenaAllocator arena(4096);
  auto tensor_res = Tensor::create(Shape{2, 4}, DataType::Float32, arena);
  ASSERT_TRUE(tensor_res.isOk());
  Tensor tensor = std::move(tensor_res).value();

  float* raw_ptr = tensor.data<float>();
  for (int i = 0; i < 8; ++i) {
    raw_ptr[i] = static_cast<float>(i + 1) * 1.5f;
  }

  auto ort_val_res = tensorToOrtValue(tensor);
  ASSERT_TRUE(ort_val_res.isOk()) << ort_val_res.error().message();
  Ort::Value& ort_val = ort_val_res.value();

  ASSERT_TRUE(ort_val.IsTensor());
  // Zero-copy guarantee: Ort::Value points to the exact same memory address as pyreth::Tensor
  EXPECT_EQ(ort_val.GetTensorRawData(), static_cast<const void*>(raw_ptr));

  const float* ort_floats = ort_val.GetTensorData<float>();
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(ort_floats[i], static_cast<float>(i + 1) * 1.5f);
  }
}

TEST(InferenceExecutionTest, ZeroCopyBridgeRejectsInvalidOrNonContiguousTensor) {
  Tensor invalid_tensor;
  auto res_invalid = tensorToOrtValue(invalid_tensor);
  EXPECT_FALSE(res_invalid.isOk());
  EXPECT_EQ(res_invalid.error().code(), StatusCode::INVALID_ARGUMENT);

  // Create a non-contiguous sliced tensor (step=2 along axis 1)
  auto full = Tensor::create(Shape{4, 4}, DataType::Float32).value();
  auto non_contig = full.slice(1, 0, 4, 2).value();
  ASSERT_FALSE(non_contig.isContiguous());

  auto res_non_contig = tensorToOrtValue(non_contig);
  EXPECT_FALSE(res_non_contig.isOk());
  EXPECT_EQ(res_non_contig.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceExecutionTest, ZeroCopyBridgeTensorFromOrtValueTransfersOwnership) {
  Ort::AllocatorWithDefaultOptions allocator;
  const int64_t dims[] = {2, 3};
  Ort::Value ort_val = Ort::Value::CreateTensor<float>(allocator, dims, 2);
  float* original_ort_ptr = ort_val.GetTensorMutableData<float>();
  for (int i = 0; i < 6; ++i) {
    original_ort_ptr[i] = static_cast<float>(i * 10);
  }

  auto tensor_res = tensorFromOrtValue(std::move(ort_val));
  ASSERT_TRUE(tensor_res.isOk()) << tensor_res.error().message();
  Tensor tensor = std::move(tensor_res).value();

  // Zero-copy guarantee: Tensor points to the exact buffer allocated by Ort::Value
  EXPECT_EQ(tensor.data<float>(), original_ort_ptr);
  EXPECT_EQ(tensor.shape(), (Shape{2, 3}));
  EXPECT_EQ(tensor.dtype(), DataType::Float32);
  EXPECT_EQ(tensor.storageType(), StorageType::EXTERNAL);

  // Shallow copies and slices keep the underlying Ort::Value alive
  Tensor sliced = tensor.slice(0, 1, 2).value();
  tensor = Tensor{};  // Drop original handle; sliced view still owns the Ort::Value
  EXPECT_FLOAT_EQ((sliced.at<float>(0, 0)), 30.0f);
  EXPECT_FLOAT_EQ((sliced.at<float>(0, 1)), 40.0f);
  EXPECT_FLOAT_EQ((sliced.at<float>(0, 2)), 50.0f);
}

// ---------------------------------------------------------------------------
// Synchronous End-to-End Inference Tests
// ---------------------------------------------------------------------------

TEST(InferenceExecutionTest, SynchronousRunOnIdentityModel) {
  auto session = InferenceSession::create(buildIdentityModel()).value();

  auto input = Tensor::create(Shape{2, 4}, DataType::Float32).value();
  for (int64_t i = 0; i < 8; ++i) {
    input.data<float>()[i] = static_cast<float>(i + 1) * 2.25f;
  }

  auto out_res = session.run({input});
  ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
  ASSERT_EQ(out_res.value().size(), 1u);

  const Tensor& output = out_res.value()[0];
  EXPECT_EQ(output.shape(), (Shape{2, 4}));
  EXPECT_EQ(output.dtype(), DataType::Float32);
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(output.data<float>()[i], static_cast<float>(i + 1) * 2.25f);
  }
}

TEST(InferenceExecutionTest, SynchronousRunOnArithmeticModelAndNamedInputs) {
  auto session = InferenceSession::create(buildArithmeticModel()).value();

  auto a = Tensor::create(Shape{2, 4}, DataType::Float32).value();
  auto b = Tensor::create(Shape{2, 4}, DataType::Float32).value();
  for (int64_t i = 0; i < 8; ++i) {
    a.data<float>()[i] = static_cast<float>(i + 1);
    b.data<float>()[i] = 10.0f;
  }

  // Test named inputs dictionary execution (in arbitrary map order)
  std::unordered_map<std::string, Tensor> named_inputs = {{"b", b}, {"a", a}};
  auto out_res = session.run(named_inputs);
  ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
  ASSERT_EQ(out_res.value().size(), 2u);

  const Tensor& sum_out = out_res.value()[0];
  const Tensor& prod_out = out_res.value()[1];
  for (int64_t i = 0; i < 8; ++i) {
    const float expected_a = static_cast<float>(i + 1);
    EXPECT_FLOAT_EQ(sum_out.data<float>()[i], expected_a + 10.0f);
    EXPECT_FLOAT_EQ(prod_out.data<float>()[i], expected_a * 10.0f);
  }
}

TEST(InferenceExecutionTest, EndToEndMlpLinearReluModelAcrossDynamicBatches) {
  auto session = InferenceSession::create(buildMlpLinearReluModel()).value();

  // Reference weights:
  // W = [[1, -1, 0.5], [0, 2, -0.5], [2, 0, 1], [-1, 1, 2]]
  // b = [0.5, -1.0, 0.0]
  const float W[4][3] = {
      {1.0f, -1.0f, 0.5f},
      {0.0f, 2.0f, -0.5f},
      {2.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, 2.0f},
  };
  const float b[3] = {0.5f, -1.0f, 0.0f};

  for (int64_t batch_size : {1, 4, 16}) {
    auto x = Tensor::create(Shape{batch_size, 4}, DataType::Float32).value();
    for (int64_t row = 0; row < batch_size; ++row) {
      for (int64_t col = 0; col < 4; ++col) {
        x.at<float>(row, col) = static_cast<float>((row + 1) * (col - 1)) * 0.5f;
      }
    }

    auto out_res = session.run({x});
    ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
    ASSERT_EQ(out_res.value().size(), 1u);

    const Tensor& y = out_res.value()[0];
    EXPECT_EQ(y.shape(), (Shape{batch_size, 3}));
    EXPECT_EQ(y.dtype(), DataType::Float32);

    for (int64_t row = 0; row < batch_size; ++row) {
      for (int64_t out_col = 0; out_col < 3; ++out_col) {
        float acc = b[out_col];
        for (int64_t k = 0; k < 4; ++k) {
          acc += x.at<float>(row, k) * W[k][out_col];
        }
        const float expected = std::max(0.0f, acc);
        EXPECT_NEAR((y.at<float>(row, out_col)), expected, 1e-5f);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Zero-Copy Arena, MemoryPool & Pre-Allocated Output Execution Tests
// ---------------------------------------------------------------------------

TEST(InferenceExecutionTest, ZeroCopyPreallocatedArenaAndInPlaceOutputs) {
  auto session = InferenceSession::create(buildMlpLinearReluModel()).value();

  ArenaAllocator arena(16384);
  auto x = Tensor::create(Shape{4, 4}, DataType::Float32, arena).value();
  x.fill<float>(1.0f);

  // 1. Run with ArenaAllocator output backing
  {
    auto out_res = session.run(std::span<const Tensor>(&x, 1), arena);
    ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
    ASSERT_EQ(out_res.value().size(), 1u);

    const Tensor& y = out_res.value()[0];
    EXPECT_EQ(y.storageType(), StorageType::ARENA);
    EXPECT_TRUE(y.isAligned());
    EXPECT_EQ(y.shape(), (Shape{4, 3}));
    // Arena has 2 pinned tensors: x and y
    EXPECT_EQ(arena.pinCount(), 2u);

    // For x = [1, 1, 1, 1]:
    // x @ W + b = [1+0+2-1 + 0.5, -1+2+0+1 - 1.0, 0.5-0.5+1+2 + 0.0] = [2.5, 1.0, 3.0]
    for (int64_t row = 0; row < 4; ++row) {
      EXPECT_FLOAT_EQ((y.at<float>(row, 0)), 2.5f);
      EXPECT_FLOAT_EQ((y.at<float>(row, 1)), 1.0f);
      EXPECT_FLOAT_EQ((y.at<float>(row, 2)), 3.0f);
    }
  }
  // y is now destroyed -> pinCount drops back to 1 (for x)
  EXPECT_EQ(arena.pinCount(), 1u);

  // 2. Run with pre-allocated output Tensor in-place (zero allocation, pointer preserved)
  auto preallocated_y = Tensor::create(Shape{4, 3}, DataType::Float32, arena).value();
  preallocated_y.zero();
  const void* original_y_ptr = preallocated_y.data();

  Tensor out_span_arr[] = {preallocated_y};
  auto status = session.run(std::span<const Tensor>(&x, 1), std::span<Tensor>(out_span_arr, 1));
  ASSERT_TRUE(status.isOk()) << status.error().message();

  // Verify ORT wrote directly into preallocated_y's buffer without changing its address
  EXPECT_EQ(preallocated_y.data(), original_y_ptr);
  for (int64_t row = 0; row < 4; ++row) {
    EXPECT_FLOAT_EQ((preallocated_y.at<float>(row, 0)), 2.5f);
    EXPECT_FLOAT_EQ((preallocated_y.at<float>(row, 1)), 1.0f);
    EXPECT_FLOAT_EQ((preallocated_y.at<float>(row, 2)), 3.0f);
  }
}

TEST(InferenceExecutionTest, MemoryPoolOutputAllocationAndAutomaticLeaseRecycle) {
  auto session = InferenceSession::create(buildIdentityModel()).value();
  MemoryPool pool(2, 8192);

  auto x = Tensor::create(Shape{3, 4}, DataType::Float32).value();
  x.fill<float>(7.5f);

  {
    auto out_res = session.run(std::span<const Tensor>(&x, 1), pool);
    ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
    const Tensor& y = out_res.value()[0];
    EXPECT_EQ(y.storageType(), StorageType::POOLED);
    EXPECT_EQ(pool.availableCount(), 1u);
    for (int64_t i = 0; i < 12; ++i) {
      EXPECT_FLOAT_EQ(y.data<float>()[i], 7.5f);
    }
  }
  // Once y goes out of scope, the leased arena is reset and returned to the pool
  EXPECT_EQ(pool.availableCount(), 2u);
}

// ---------------------------------------------------------------------------
// Asynchronous Execution (`runAsync`) & Arena Lifetime Pinning Contract Tests
// ---------------------------------------------------------------------------

TEST(InferenceExecutionTest, RunAsyncPinsArenaStorageUntilExecutionCompletes) {
  auto session = InferenceSession::create(buildMlpLinearReluModel()).value();
  ThreadPool thread_pool(4);
  ArenaAllocator arena(16384);

  TaskHandle<Result<std::vector<Tensor>>> handle;
  {
    // Allocate input inside `arena` and immediately let the caller's `Tensor` handle
    // go out of scope right after calling `runAsync`.
    auto x = Tensor::create(Shape{8, 4}, DataType::Float32, arena).value();
    x.fill<float>(1.0f);
    EXPECT_TRUE(arena.isPinned());

    auto async_res = session.runAsync(thread_pool, {x});
    ASSERT_TRUE(async_res.isOk()) << async_res.error().message();
    handle = std::move(async_res).value();
    // `x` is destroyed at the end of this block, while `runAsync` holds its own shallow copy!
  }

  auto result = handle.get();
  ASSERT_TRUE(result.isOk()) << result.error().message();
  ASSERT_EQ(result.value().size(), 1u);

  const Tensor& y = result.value()[0];
  EXPECT_EQ(y.shape(), (Shape{8, 3}));
  for (int64_t row = 0; row < 8; ++row) {
    EXPECT_FLOAT_EQ((y.at<float>(row, 0)), 2.5f);
    EXPECT_FLOAT_EQ((y.at<float>(row, 1)), 1.0f);
    EXPECT_FLOAT_EQ((y.at<float>(row, 2)), 3.0f);
  }

  // Once the async task has finished and released its captured input copy,
  // the arena is completely unpinned and can be reset in O(1).
  EXPECT_FALSE(arena.isPinned());
  EXPECT_NO_THROW(arena.reset());
}

TEST(InferenceExecutionTest, ConcurrentRunAsyncAcrossMultipleWorkerThreads) {
  auto session = InferenceSession::create(buildArithmeticModel()).value();
  ThreadPool thread_pool(4);
  MemoryPool mem_pool(8, 16384);

  constexpr int kNumRequests = 32;
  std::vector<TaskHandle<Result<std::vector<Tensor>>>> handles;
  handles.reserve(kNumRequests);

  for (int req = 0; req < kNumRequests; ++req) {
    auto a = Tensor::create(Shape{4, 4}, DataType::Float32).value();
    auto b = Tensor::create(Shape{4, 4}, DataType::Float32).value();
    a.fill<float>(static_cast<float>(req));
    b.fill<float>(2.0f);

    Tensor inputs[] = {a, b};
    auto async_res = session.runAsync(thread_pool, std::span<const Tensor>(inputs, 2), mem_pool);
    ASSERT_TRUE(async_res.isOk());
    handles.push_back(std::move(async_res).value());
  }

  for (int req = 0; req < kNumRequests; ++req) {
    auto res = handles[static_cast<size_t>(req)].get();
    ASSERT_TRUE(res.isOk()) << res.error().message();
    ASSERT_EQ(res.value().size(), 2u);

    const float expected_sum = static_cast<float>(req) + 2.0f;
    const float expected_prod = static_cast<float>(req) * 2.0f;
    for (int64_t i = 0; i < 16; ++i) {
      EXPECT_FLOAT_EQ(res.value()[0].data<float>()[i], expected_sum);
      EXPECT_FLOAT_EQ(res.value()[1].data<float>()[i], expected_prod);
    }
  }
}

}  // namespace
