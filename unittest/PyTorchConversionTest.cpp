#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <pyreth/backend.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/result.hpp>
#include <pyreth/tensor.hpp>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace pyreth;

// ---------------------------------------------------------------------------
// ONNX Protobuf Wire-Format Builder Helpers
// ---------------------------------------------------------------------------

enum class OnnxDataType : uint32_t {
  FLOAT = 1,
  INT8 = 3,
  INT32 = 6,
  INT64 = 7,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
};

struct DimSpec {
  int64_t value{-1};
  std::string param{};

  static DimSpec fixed(int64_t v) { return DimSpec{v, ""}; }
  static DimSpec dynamic(std::string sym = "batch_size") { return DimSpec{-1, std::move(sym)}; }
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

std::vector<uint8_t> buildModelProto(std::string_view producer_name, std::string_view graph_name,
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
    graph.writeSubmessage(5, init);
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
  model.writeStringField(2, producer_name);
  model.writeSubmessage(7, graph);
  model.writeSubmessage(8, opset);
  return model.bytes();
}

// Model 1: Dynamic batch PyTorch MLP model (y = Relu(x * W + b))
// x: [batch_size, 4] -> y: [batch_size, 2]
std::vector<uint8_t> buildPyTorchDynamicMlpModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(2)}}};

  const int64_t w_dims[] = {4, 2};
  const float w_vals[] = {1.0f, 0.5f, -1.0f, 2.0f, 0.0f, 1.0f, 2.0f, -0.5f};
  const int64_t b_dims[] = {2};
  const float b_vals[] = {0.1f, -0.2f};

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
      encodeNode(mm_in, mm_out, "mm", "MatMul"),
      encodeNode(add_in, add_out, "add", "Add"),
      encodeNode(relu_in, relu_out, "relu", "Relu"),
  };

  return buildModelProto("pytorch", "pytorch_mlp_graph", inputs, outputs, nodes, initializers);
}

// Model 2: Multi-input PyTorch Transformer / NLP model
// Inputs: input_ids [batch_size, 16] (int64), attention_mask [batch_size, 16] (int64)
// Outputs: logits [batch_size, 16] (int64 Identity pass-through for IO test)
std::vector<uint8_t> buildPyTorchMultiInputNlpModel() {
  std::vector<TensorSpec> inputs = {
      {"input_ids", OnnxDataType::INT64, {DimSpec::dynamic("batch_size"), DimSpec::fixed(16)}},
      {"attention_mask", OnnxDataType::INT64, {DimSpec::dynamic("batch_size"), DimSpec::fixed(16)}},
  };
  std::vector<TensorSpec> outputs = {
      {"logits", OnnxDataType::INT64, {DimSpec::dynamic("batch_size"), DimSpec::fixed(16)}},
  };

  std::string_view in_names[] = {"input_ids"};
  std::string_view out_names[] = {"logits"};
  std::vector<ProtoBuilder> nodes = {
      encodeNode(in_names, out_names, "id_node", "Identity"),
  };

  return buildModelProto("pytorch", "pytorch_nlp_graph", inputs, outputs, nodes);
}

// Model 3: PyTorch Regression model with 1D output [batch_size, 1]
// x: [batch_size, 8] -> predictions: [batch_size, 1]
std::vector<uint8_t> buildPyTorchRegressionModel() {
  std::vector<TensorSpec> inputs = {
      {"features", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(8)}}};
  std::vector<TensorSpec> outputs = {
      {"predictions", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(1)}}};

  const int64_t w_dims[] = {8, 1};
  const float w_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  std::vector<ProtoBuilder> initializers = {
      encodeFloatInitializer("W", w_dims, w_vals),
  };

  std::string_view mm_in[] = {"features", "W"};
  std::string_view mm_out[] = {"predictions"};
  std::vector<ProtoBuilder> nodes = {
      encodeNode(mm_in, mm_out, "mm", "MatMul"),
  };

  return buildModelProto("pytorch", "pytorch_regression_graph", inputs, outputs, nodes,
                         initializers);
}

// Model 4: 4D PyTorch CNN model [batch_size, 1, 4, 4] -> [batch_size, 1, 4, 4]
std::vector<uint8_t> buildPyTorch4DVisionModel() {
  std::vector<TensorSpec> inputs = {
      {"pixel_values",
       OnnxDataType::FLOAT,
       {DimSpec::dynamic("batch_size"), DimSpec::fixed(1), DimSpec::fixed(4), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"output",
       OnnxDataType::FLOAT,
       {DimSpec::dynamic("batch_size"), DimSpec::fixed(1), DimSpec::fixed(4), DimSpec::fixed(4)}}};

  std::string_view in_names[] = {"pixel_values"};
  std::string_view out_names[] = {"output"};
  std::vector<ProtoBuilder> nodes = {
      encodeNode(in_names, out_names, "relu_node", "Relu"),
  };

  return buildModelProto("pytorch", "pytorch_cnn_graph", inputs, outputs, nodes);
}

}  // namespace

// ===========================================================================
// Phase 3 & Phase 5 Integration Tests
// ===========================================================================

TEST(PyTorchConversionTest, test_cpp_inference_session_loads_exported_dynamic_batch_model) {
  auto model_data = buildPyTorchDynamicMlpModel();
  auto session_res = InferenceSession::create(model_data);
  ASSERT_TRUE(session_res.isOk()) << session_res.error().message();

  auto session = std::move(session_res).value();
  EXPECT_TRUE(session.isValid());

  const auto& info = session.getModelInfo();
  EXPECT_EQ(info.producer_name, "pytorch");
  EXPECT_TRUE(info.hasDynamicBatch());
  EXPECT_EQ(session.getInputCount(), 1u);
  EXPECT_EQ(session.getOutputCount(), 1u);

  // Test execution across variable dynamic batch sizes
  const std::vector<int64_t> test_batches = {1, 2, 4, 8};
  for (int64_t b : test_batches) {
    auto input_res = Tensor::create(Shape{b, 4}, DataType::Float32);
    ASSERT_TRUE(input_res.isOk());
    auto input = std::move(input_res).value();

    // Fill with positive dummy data
    float* in_ptr = input.data<float>();
    for (int64_t i = 0; i < b * 4; ++i) {
      in_ptr[i] = 1.0f;
    }

    auto run_res = session.run({input});
    ASSERT_TRUE(run_res.isOk()) << "Run failed for batch " << b << ": "
                                << run_res.error().message();
    auto outputs = std::move(run_res).value();
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].shape(), (Shape{b, 2}));
    EXPECT_EQ(outputs[0].dtype(), DataType::Float32);
    EXPECT_TRUE(outputs[0].isValid());
  }
}

TEST(PyTorchConversionTest, test_cpp_model_info_reads_io_schema_from_onnx_export) {
  // Test multi-input transformer model schema discovery
  auto nlp_model = buildPyTorchMultiInputNlpModel();
  auto session_nlp_res = InferenceSession::create(nlp_model);
  ASSERT_TRUE(session_nlp_res.isOk());
  auto session_nlp = std::move(session_nlp_res).value();

  const auto& info_nlp = session_nlp.getModelInfo();
  EXPECT_EQ(info_nlp.producer_name, "pytorch");
  EXPECT_EQ(info_nlp.graph_name, "pytorch_nlp_graph");
  EXPECT_EQ(info_nlp.inputCount(), 2u);
  EXPECT_EQ(info_nlp.outputCount(), 1u);
  EXPECT_TRUE(info_nlp.hasDynamicBatch());
  EXPECT_TRUE(info_nlp.hasDynamicInputs());

  // Input 0: input_ids
  EXPECT_EQ(info_nlp.inputs[0].name, "input_ids");
  EXPECT_EQ(info_nlp.inputs[0].dtype, DataType::Int64);
  EXPECT_EQ(info_nlp.inputs[0].rank(), 2u);
  EXPECT_EQ(info_nlp.inputs[0].shape[0], -1);
  EXPECT_EQ(info_nlp.inputs[0].shape[1], 16);
  ASSERT_GE(info_nlp.inputs[0].symbolic_dims.size(), 1u);
  EXPECT_EQ(info_nlp.inputs[0].symbolic_dims[0], "batch_size");

  // Input 1: attention_mask
  EXPECT_EQ(info_nlp.inputs[1].name, "attention_mask");
  EXPECT_EQ(info_nlp.inputs[1].dtype, DataType::Int64);
  EXPECT_EQ(info_nlp.inputs[1].rank(), 2u);
  EXPECT_EQ(info_nlp.inputs[1].shape[0], -1);
  EXPECT_EQ(info_nlp.inputs[1].shape[1], 16);
  ASSERT_GE(info_nlp.inputs[1].symbolic_dims.size(), 1u);
  EXPECT_EQ(info_nlp.inputs[1].symbolic_dims[0], "batch_size");

  // Output: logits
  EXPECT_EQ(info_nlp.outputs[0].name, "logits");
  EXPECT_EQ(info_nlp.outputs[0].dtype, DataType::Int64);
  EXPECT_EQ(info_nlp.outputs[0].rank(), 2u);
  EXPECT_EQ(info_nlp.outputs[0].shape[0], -1);
  EXPECT_EQ(info_nlp.outputs[0].shape[1], 16);

  // Name resolution
  EXPECT_NE(info_nlp.findInput("input_ids"), nullptr);
  EXPECT_NE(info_nlp.findInput("attention_mask"), nullptr);
  EXPECT_EQ(info_nlp.findInput("non_existent"), nullptr);
  EXPECT_NE(info_nlp.findOutput("logits"), nullptr);

  // Test Regression model schema discovery
  auto reg_model = buildPyTorchRegressionModel();
  auto session_reg_res = InferenceSession::create(reg_model);
  ASSERT_TRUE(session_reg_res.isOk());
  auto session_reg = std::move(session_reg_res).value();

  const auto& info_reg = session_reg.getModelInfo();
  EXPECT_EQ(info_reg.inputCount(), 1u);
  EXPECT_EQ(info_reg.outputCount(), 1u);
  EXPECT_EQ(info_reg.inputs[0].name, "features");
  EXPECT_EQ(info_reg.inputs[0].shape, (std::vector<int64_t>{-1, 8}));
  EXPECT_EQ(info_reg.outputs[0].name, "predictions");
  EXPECT_EQ(info_reg.outputs[0].shape, (std::vector<int64_t>{-1, 1}));
  EXPECT_EQ(info_reg.outputs[0].dtype, DataType::Float32);

  // Test 4D Vision model schema discovery
  auto cnn_model = buildPyTorch4DVisionModel();
  auto session_cnn_res = InferenceSession::create(cnn_model);
  ASSERT_TRUE(session_cnn_res.isOk());
  auto session_cnn = std::move(session_cnn_res).value();

  const auto& info_cnn = session_cnn.getModelInfo();
  EXPECT_EQ(info_cnn.inputs[0].name, "pixel_values");
  EXPECT_EQ(info_cnn.inputs[0].rank(), 4u);
  EXPECT_EQ(info_cnn.inputs[0].shape, (std::vector<int64_t>{-1, 1, 4, 4}));
  EXPECT_EQ(info_cnn.outputs[0].shape, (std::vector<int64_t>{-1, 1, 4, 4}));
}

TEST(PyTorchConversionTest, test_cpp_invalid_input_rank_returns_StatusCode_INVALID_ARGUMENT) {
  auto model_data = buildPyTorchDynamicMlpModel();
  auto session = InferenceSession::create(model_data).value();

  // Model expects rank-2 tensor [batch_size, 4]
  // 1. Submit rank-1 tensor [4] -> Rejected
  auto rank1_tensor = Tensor::create(Shape{4}, DataType::Float32).value();
  auto val1 = session.validateInputs({rank1_tensor});
  EXPECT_FALSE(val1.isOk());
  EXPECT_TRUE(val1.error().code() == StatusCode::INVALID_DIMENSIONS ||
              val1.error().code() == StatusCode::INVALID_ARGUMENT);

  auto run1 = session.run({rank1_tensor});
  EXPECT_FALSE(run1.isOk());
  EXPECT_TRUE(run1.error().code() == StatusCode::INVALID_DIMENSIONS ||
              run1.error().code() == StatusCode::INVALID_ARGUMENT);

  // 2. Submit rank-3 tensor [2, 2, 2] -> Rejected
  auto rank3_tensor = Tensor::create(Shape{2, 2, 2}, DataType::Float32).value();
  auto val3 = session.validateInputs({rank3_tensor});
  EXPECT_FALSE(val3.isOk());
  EXPECT_TRUE(val3.error().code() == StatusCode::INVALID_DIMENSIONS ||
              val3.error().code() == StatusCode::INVALID_ARGUMENT);

  auto run3 = session.run({rank3_tensor});
  EXPECT_FALSE(run3.isOk());
  EXPECT_TRUE(run3.error().code() == StatusCode::INVALID_DIMENSIONS ||
              run3.error().code() == StatusCode::INVALID_ARGUMENT);

  // 3. Submit rank-4 tensor to 2D model -> Rejected
  auto rank4_tensor = Tensor::create(Shape{1, 1, 2, 2}, DataType::Float32).value();
  auto run4 = session.run({rank4_tensor});
  EXPECT_FALSE(run4.isOk());
  EXPECT_TRUE(run4.error().code() == StatusCode::INVALID_DIMENSIONS ||
              run4.error().code() == StatusCode::INVALID_ARGUMENT);
}

TEST(PyTorchConversionTest,
     test_cpp_invalid_dtype_for_session_input_returns_backend_or_invalid_argument) {
  auto model_data = buildPyTorchDynamicMlpModel();
  auto session = InferenceSession::create(model_data).value();

  // Model expects Float32. Submit Int32 -> INVALID_ARGUMENT
  auto int32_tensor = Tensor::create(Shape{2, 4}, DataType::Int32).value();
  auto val_i32 = session.validateInputs({int32_tensor});
  EXPECT_FALSE(val_i32.isOk());
  EXPECT_EQ(val_i32.error().code(), StatusCode::INVALID_ARGUMENT);

  auto run_i32 = session.run({int32_tensor});
  EXPECT_FALSE(run_i32.isOk());
  EXPECT_EQ(run_i32.error().code(), StatusCode::INVALID_ARGUMENT);

  // Submit Int64 -> INVALID_ARGUMENT
  auto int64_tensor = Tensor::create(Shape{2, 4}, DataType::Int64).value();
  auto run_i64 = session.run({int64_tensor});
  EXPECT_FALSE(run_i64.isOk());
  EXPECT_EQ(run_i64.error().code(), StatusCode::INVALID_ARGUMENT);

  // For NLP model expecting Int64: Submit Float32 -> INVALID_ARGUMENT
  auto nlp_model = buildPyTorchMultiInputNlpModel();
  auto nlp_session = InferenceSession::create(nlp_model).value();

  auto f32_tensor = Tensor::create(Shape{2, 16}, DataType::Float32).value();
  auto mask_tensor = Tensor::create(Shape{2, 16}, DataType::Int64).value();
  auto run_nlp_wrong_dtype = nlp_session.run({f32_tensor, mask_tensor});
  EXPECT_FALSE(run_nlp_wrong_dtype.isOk());
  EXPECT_EQ(run_nlp_wrong_dtype.error().code(), StatusCode::INVALID_ARGUMENT);

  // Null/uninitialized tensor -> INVALID_ARGUMENT
  Tensor null_tensor;
  auto run_null = session.run({null_tensor});
  EXPECT_FALSE(run_null.isOk());
  EXPECT_EQ(run_null.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PyTorchConversionTest, DynamicBatchCrossInputSymbolicConsistency) {
  auto nlp_model = buildPyTorchMultiInputNlpModel();
  auto session = InferenceSession::create(nlp_model).value();

  // Valid: both input_ids and attention_mask have batch_size = 3
  auto ids_ok = Tensor::create(Shape{3, 16}, DataType::Int64).value();
  auto mask_ok = Tensor::create(Shape{3, 16}, DataType::Int64).value();
  auto run_ok = session.run({ids_ok, mask_ok});
  EXPECT_TRUE(run_ok.isOk());

  // Invalid: input_ids has batch_size = 2, attention_mask has batch_size = 4
  // Must fail due to symbolic dimension mismatch along axis 0
  auto ids_bad = Tensor::create(Shape{2, 16}, DataType::Int64).value();
  auto mask_bad = Tensor::create(Shape{4, 16}, DataType::Int64).value();
  auto val_bad = session.validateInputs({ids_bad, mask_bad});
  EXPECT_FALSE(val_bad.isOk());
  EXPECT_EQ(val_bad.error().code(), StatusCode::INVALID_DIMENSIONS);

  auto run_bad = session.run({ids_bad, mask_bad});
  EXPECT_FALSE(run_bad.isOk());
  EXPECT_EQ(run_bad.error().code(), StatusCode::INVALID_DIMENSIONS);
}

TEST(PyTorchConversionTest, FixedDimensionMismatchRejection) {
  auto model_data = buildPyTorchDynamicMlpModel();
  auto session = InferenceSession::create(model_data).value();

  // Model expects [batch_size, 4]. Provide [2, 5] -> INVALID_DIMENSIONS
  auto wrong_feat = Tensor::create(Shape{2, 5}, DataType::Float32).value();
  auto run_wrong = session.run({wrong_feat});
  EXPECT_FALSE(run_wrong.isOk());
  EXPECT_EQ(run_wrong.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Dynamic batch dimension cannot be zero or negative
  auto zero_batch = Tensor::create(Shape{0, 4}, DataType::Float32).value();
  auto run_zero = session.run({zero_batch});
  EXPECT_FALSE(run_zero.isOk());
  EXPECT_EQ(run_zero.error().code(), StatusCode::INVALID_DIMENSIONS);
}

TEST(PyTorchConversionTest, NamedInputsExecution) {
  auto nlp_model = buildPyTorchMultiInputNlpModel();
  auto session = InferenceSession::create(nlp_model).value();

  auto ids = Tensor::create(Shape{2, 16}, DataType::Int64).value();
  auto mask = Tensor::create(Shape{2, 16}, DataType::Int64).value();

  std::unordered_map<std::string, Tensor> named_map = {
      {"attention_mask", mask},
      {"input_ids", ids},
  };

  auto run_res = session.run(named_map);
  ASSERT_TRUE(run_res.isOk()) << run_res.error().message();
  auto outputs = std::move(run_res).value();
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].shape(), (Shape{2, 16}));

  // Unknown input name -> INVALID_ARGUMENT
  std::unordered_map<std::string, Tensor> bad_named_map = {
      {"unknown_input", mask},
      {"input_ids", ids},
  };
  auto bad_run = session.run(bad_named_map);
  EXPECT_FALSE(bad_run.isOk());
  EXPECT_EQ(bad_run.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(PyTorchConversionTest, CachedExportedOnnxModelExecutionIfPresent) {
  std::string cached_path = "benchmarks/.onnx_cache/regression__prajjwal1__bert_tiny__seq32.onnx";
  if (!std::filesystem::exists(cached_path)) {
    cached_path = "../" + cached_path;
  }
  if (!std::filesystem::exists(cached_path)) {
    cached_path = "../" + cached_path;
  }
  if (!std::filesystem::exists(cached_path)) {
    GTEST_SKIP() << "Cached benchmark model not found; skipping disk test.";
  }

  auto session_res = InferenceSession::create(cached_path);
  ASSERT_TRUE(session_res.isOk()) << session_res.error().message();
  auto session = std::move(session_res).value();

  const auto& info = session.getModelInfo();
  EXPECT_TRUE(info.hasDynamicBatch());
  EXPECT_EQ(session.getInputCount(), 2u);

  // Generate valid inputs matching the bert-tiny schema
  auto in_ids = Tensor::create(Shape{2, 32}, DataType::Int64).value();
  auto att_mask = Tensor::create(Shape{2, 32}, DataType::Int64).value();
  for (int64_t i = 0; i < 64; ++i) {
    in_ids.data<int64_t>()[i] = 101;
    att_mask.data<int64_t>()[i] = 1;
  }

  auto run_res = session.run({in_ids, att_mask});
  ASSERT_TRUE(run_res.isOk()) << run_res.error().message();
  auto outputs = std::move(run_res).value();
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].shape(), (Shape{2, 1}));
  EXPECT_EQ(outputs[0].dtype(), DataType::Float32);
}
