#include <gtest/gtest.h>

#include <cstdint>
#include <pyreth/inference_session.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/tensor.hpp>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace pyreth;

// ---------------------------------------------------------------------------
// Minimal ONNX Protobuf Wire-Format Builder for Benchmark Model Specifications
// ---------------------------------------------------------------------------

enum class OnnxDataType : uint32_t {
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  BFLOAT16 = 16,
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
      dim_proto.writeStringField(2, d.param);  // dim_param
    } else if (d.value >= 0) {
      dim_proto.writeVarintField(1, static_cast<uint64_t>(d.value));  // dim_value
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

ProtoBuilder encodeIdentityNode(std::string_view in_name, std::string_view out_name,
                                std::string_view node_name) {
  ProtoBuilder node;
  node.writeStringField(1, in_name);
  node.writeStringField(2, out_name);
  node.writeStringField(3, node_name);
  node.writeStringField(4, "Identity");
  return node;
}

struct OnnxModelSpec {
  std::string producer_name{"pyreth_test"};
  std::string graph_name{"test_graph"};
  std::string domain{"ai.onnx"};
  std::string description{};
  std::string graph_description{};
  uint64_t model_version{1};
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;
  std::vector<ProtoBuilder> nodes;
  std::vector<std::pair<std::string, std::string>> custom_metadata;
};

std::vector<uint8_t> buildOnnxModel(const OnnxModelSpec& spec) {
  ProtoBuilder graph;
  for (const auto& n : spec.nodes) {
    graph.writeSubmessage(1, n);
  }
  graph.writeStringField(2, spec.graph_name);
  if (!spec.graph_description.empty()) {
    graph.writeStringField(10, spec.graph_description);
  }
  for (const auto& in : spec.inputs) {
    graph.writeSubmessage(11, encodeValueInfo(in));
  }
  for (const auto& out : spec.outputs) {
    graph.writeSubmessage(12, encodeValueInfo(out));
  }

  ProtoBuilder opset;
  opset.writeStringField(1, "");
  opset.writeVarintField(2, 17);

  ProtoBuilder model;
  model.writeVarintField(1, 9);  // ir_version = 9
  model.writeStringField(2, spec.producer_name);
  model.writeStringField(4, spec.domain);
  model.writeVarintField(5, spec.model_version);
  if (!spec.description.empty()) {
    model.writeStringField(6, spec.description);
  }
  model.writeSubmessage(7, graph);
  model.writeSubmessage(8, opset);

  for (const auto& [k, v] : spec.custom_metadata) {
    ProtoBuilder entry;
    entry.writeStringField(1, k);
    entry.writeStringField(2, v);
    model.writeSubmessage(14, entry);
  }

  return model.bytes();
}

// Construct a SqueezeNet 1.1 specification model:
// Input: "data" Float32 [1, 3, 224, 224] -> Output: "squeezenet0_flatten0_reshape0" Float32 [1,
// 1000] (Uses Shape/Size/Cast/Identity-compatible pass-through or matching IO graph)
std::vector<uint8_t> buildSqueezeNetSpecModel() {
  OnnxModelSpec spec;
  spec.producer_name = "pytorch";
  spec.graph_name = "squeezenet1_1";
  spec.domain = "ai.onnx";
  spec.model_version = 4;
  spec.description = "SqueezeNet 1.1 ImageNet-1K Classification Model";
  spec.graph_description = "SqueezeNet 1.1 Main Computation Graph";
  spec.custom_metadata = {
      {"architecture", "SqueezeNet1.1"},
      {"dataset", "ImageNet-1K"},
      {"num_classes", "1000"},
  };
  spec.inputs = {
      TensorSpec{"data",
                 OnnxDataType::FLOAT,
                 {DimSpec::fixed(1), DimSpec::fixed(3), DimSpec::fixed(224), DimSpec::fixed(224)}},
  };
  spec.outputs = {
      TensorSpec{"squeezenet0_flatten0_reshape0",
                 OnnxDataType::FLOAT,
                 {DimSpec::fixed(1), DimSpec::fixed(3), DimSpec::fixed(224), DimSpec::fixed(224)}},
  };
  spec.nodes = {
      encodeIdentityNode("data", "squeezenet0_flatten0_reshape0", "squeezenet_head"),
  };
  return buildOnnxModel(spec);
}

// Construct a ResNet-50 Dynamic-Batch specification model:
// Input: "input" Float32 ["batch_size"(-1), 3, 224, 224]
// Output: "logits" Float32 ["batch_size"(-1), 3, 224, 224]
std::vector<uint8_t> buildDynamicResNet50SpecModel() {
  OnnxModelSpec spec;
  spec.producer_name = "onnx.hub";
  spec.graph_name = "resnet50_v2_dynamic";
  spec.domain = "ai.onnx";
  spec.model_version = 2;
  spec.description = "ResNet-50 v2 with Dynamic Batch Axis";
  spec.inputs = {
      TensorSpec{"input",
                 OnnxDataType::FLOAT,
                 {DimSpec::dynamic("batch_size"), DimSpec::fixed(3), DimSpec::fixed(224),
                  DimSpec::fixed(224)}},
  };
  spec.outputs = {
      TensorSpec{"logits",
                 OnnxDataType::FLOAT,
                 {DimSpec::dynamic("batch_size"), DimSpec::fixed(3), DimSpec::fixed(224),
                  DimSpec::fixed(224)}},
  };
  spec.nodes = {
      encodeIdentityNode("input", "logits", "resnet_out"),
  };
  return buildOnnxModel(spec);
}

// Construct a Multi-Input / Multi-Output Dynamic Sequence Transformer (BERT-style) model:
// Inputs:
//   - "input_ids": Int64 ["batch_size"(-1), "sequence_length"(-1)]
//   - "attention_mask": Int64 ["batch_size"(-1), "sequence_length"(-1)]
//   - "token_embeddings": Float32 ["batch_size"(-1), "sequence_length"(-1), 768]
// Outputs:
//   - "last_hidden_state": Float32 ["batch_size"(-1), "sequence_length"(-1), 768]
//   - "active_token_ids": Int64 ["batch_size"(-1), "sequence_length"(-1)]
std::vector<uint8_t> buildDynamicTransformerSpecModel() {
  OnnxModelSpec spec;
  spec.producer_name = "huggingface_optimum";
  spec.graph_name = "bert_base_uncased";
  spec.domain = "ai.onnx";
  spec.model_version = 1;
  spec.inputs = {
      TensorSpec{"input_ids",
                 OnnxDataType::INT64,
                 {DimSpec::dynamic("batch_size"), DimSpec::dynamic("sequence_length")}},
      TensorSpec{"attention_mask",
                 OnnxDataType::INT64,
                 {DimSpec::dynamic("batch_size"), DimSpec::dynamic("sequence_length")}},
      TensorSpec{"token_embeddings",
                 OnnxDataType::FLOAT,
                 {DimSpec::dynamic("batch_size"), DimSpec::dynamic("sequence_length"),
                  DimSpec::fixed(768)}},
  };
  spec.outputs = {
      TensorSpec{"last_hidden_state",
                 OnnxDataType::FLOAT,
                 {DimSpec::dynamic("batch_size"), DimSpec::dynamic("sequence_length"),
                  DimSpec::fixed(768)}},
      TensorSpec{"active_token_ids",
                 OnnxDataType::INT64,
                 {DimSpec::dynamic("batch_size"), DimSpec::dynamic("sequence_length")}},
  };
  ProtoBuilder add_node;
  add_node.writeStringField(1, "input_ids");
  add_node.writeStringField(1, "attention_mask");
  add_node.writeStringField(2, "active_token_ids");
  add_node.writeStringField(3, "mask_add");
  add_node.writeStringField(4, "Add");

  spec.nodes = {
      encodeIdentityNode("token_embeddings", "last_hidden_state", "encoder_out"),
      add_node,
  };
  return buildOnnxModel(spec);
}

// ---------------------------------------------------------------------------
// Benchmark Model Metadata & IO Discovery Tests
// ---------------------------------------------------------------------------

TEST(ModelInfoTest, SqueezeNetFixedBatchMetadataAndIoDiscovery) {
  const auto model_bytes = buildSqueezeNetSpecModel();
  const auto res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& session = res.value();
  const auto& info = session.getModelInfo();

  EXPECT_EQ(info.producer_name, "pytorch");
  EXPECT_EQ(info.graph_name, "squeezenet1_1");
  EXPECT_EQ(info.domain, "ai.onnx");
  EXPECT_EQ(info.version, 4);
  EXPECT_EQ(info.description, "SqueezeNet 1.1 ImageNet-1K Classification Model");
  EXPECT_EQ(info.graph_description, "SqueezeNet 1.1 Main Computation Graph");

  ASSERT_EQ(info.custom_metadata.size(), 3u);
  EXPECT_EQ(info.custom_metadata.at("architecture"), "SqueezeNet1.1");
  EXPECT_EQ(info.custom_metadata.at("dataset"), "ImageNet-1K");
  EXPECT_EQ(info.custom_metadata.at("num_classes"), "1000");

  EXPECT_EQ(info.inputCount(), 1u);
  EXPECT_EQ(info.outputCount(), 1u);
  EXPECT_FALSE(info.hasDynamicBatch());
  EXPECT_FALSE(info.hasDynamicInputs());
  EXPECT_FALSE(info.hasDynamicOutputs());

  const auto& in_info = session.getInputs()[0];
  EXPECT_EQ(in_info.name, "data");
  EXPECT_EQ(in_info.dtype, DataType::Float32);
  EXPECT_EQ(in_info.rank(), 4u);
  EXPECT_EQ(in_info.shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_FALSE(in_info.hasDynamicShape());
  EXPECT_FALSE(in_info.hasDynamicBatch());
  EXPECT_EQ(in_info.fixedElementCount(), 1 * 3 * 224 * 224);

  const auto& out_info = session.getOutputs()[0];
  EXPECT_EQ(out_info.name, "squeezenet0_flatten0_reshape0");
  EXPECT_EQ(out_info.dtype, DataType::Float32);
  EXPECT_EQ(out_info.rank(), 4u);
}

TEST(ModelInfoTest, ResNet50DynamicBatchMetadataAndSymbolicDims) {
  const auto model_bytes = buildDynamicResNet50SpecModel();
  const auto res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& session = res.value();
  const auto& info = session.getModelInfo();

  EXPECT_EQ(info.producer_name, "onnx.hub");
  EXPECT_EQ(info.graph_name, "resnet50_v2_dynamic");
  EXPECT_TRUE(info.hasDynamicBatch());
  EXPECT_TRUE(info.hasDynamicInputs());
  EXPECT_TRUE(info.hasDynamicOutputs());

  auto in_res = session.getInputInfo("input");
  ASSERT_TRUE(in_res.isOk());
  const auto& in = in_res.value();

  EXPECT_EQ(in.name, "input");
  EXPECT_EQ(in.dtype, DataType::Float32);
  EXPECT_EQ(in.shape, (std::vector<int64_t>{-1, 3, 224, 224}));
  ASSERT_EQ(in.symbolic_dims.size(), 4u);
  EXPECT_EQ(in.symbolic_dims[0], "batch_size");
  EXPECT_TRUE(in.symbolic_dims[1].empty());
  EXPECT_TRUE(in.symbolic_dims[2].empty());
  EXPECT_TRUE(in.symbolic_dims[3].empty());

  EXPECT_TRUE(in.hasDynamicShape());
  EXPECT_TRUE(in.hasDynamicBatch());
  EXPECT_TRUE(in.isDynamicDim(0));
  EXPECT_FALSE(in.isDynamicDim(1));
  EXPECT_FALSE(in.isDynamicDim(2));
  EXPECT_FALSE(in.isDynamicDim(3));
  EXPECT_FALSE(in.isDynamicDim(99));
  EXPECT_EQ(in.fixedElementCount(), 3 * 224 * 224);

  auto out_res = session.getOutputInfo("logits");
  ASSERT_TRUE(out_res.isOk());
  const auto& out = out_res.value();
  EXPECT_EQ(out.symbolic_dims[0], "batch_size");
  EXPECT_TRUE(out.hasDynamicBatch());
}

TEST(ModelInfoTest, MultiInputDynamicSequenceTransformerDiscovery) {
  const auto model_bytes = buildDynamicTransformerSpecModel();
  const auto res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& session = res.value();
  const auto& info = session.getModelInfo();

  ASSERT_EQ(info.inputCount(), 3u);
  ASSERT_EQ(info.outputCount(), 2u);
  EXPECT_TRUE(info.hasDynamicBatch());
  EXPECT_TRUE(info.hasDynamicInputs());
  EXPECT_TRUE(info.hasDynamicOutputs());

  // Input 0: input_ids
  EXPECT_EQ(info.inputs[0].name, "input_ids");
  EXPECT_EQ(info.inputs[0].dtype, DataType::Int64);
  EXPECT_EQ(info.inputs[0].shape, (std::vector<int64_t>{-1, -1}));
  EXPECT_EQ(info.inputs[0].symbolic_dims,
            (std::vector<std::string>{"batch_size", "sequence_length"}));

  // Input 1: attention_mask
  EXPECT_EQ(info.inputs[1].name, "attention_mask");
  EXPECT_EQ(info.inputs[1].dtype, DataType::Int64);
  EXPECT_EQ(info.inputs[1].shape, (std::vector<int64_t>{-1, -1}));
  EXPECT_EQ(info.inputs[1].symbolic_dims,
            (std::vector<std::string>{"batch_size", "sequence_length"}));

  // Input 2: token_embeddings
  EXPECT_EQ(info.inputs[2].name, "token_embeddings");
  EXPECT_EQ(info.inputs[2].dtype, DataType::Float32);
  EXPECT_EQ(info.inputs[2].shape, (std::vector<int64_t>{-1, -1, 768}));
  EXPECT_EQ(info.inputs[2].symbolic_dims,
            (std::vector<std::string>{"batch_size", "sequence_length", ""}));
  EXPECT_EQ(info.inputs[2].fixedElementCount(), 768);

  // Outputs
  EXPECT_EQ(info.outputs[0].name, "last_hidden_state");
  EXPECT_EQ(info.outputs[0].dtype, DataType::Float32);
  EXPECT_EQ(info.outputs[0].shape, (std::vector<int64_t>{-1, -1, 768}));

  EXPECT_EQ(info.outputs[1].name, "active_token_ids");
  EXPECT_EQ(info.outputs[1].dtype, DataType::Int64);
  EXPECT_EQ(info.outputs[1].shape, (std::vector<int64_t>{-1, -1}));

  // Index lookups
  EXPECT_EQ(info.getInputIndex("input_ids").value(), 0u);
  EXPECT_EQ(info.getInputIndex("attention_mask").value(), 1u);
  EXPECT_EQ(info.getInputIndex("token_embeddings").value(), 2u);
  EXPECT_FALSE(info.getInputIndex("non_existent").isOk());

  EXPECT_EQ(info.getOutputIndex("last_hidden_state").value(), 0u);
  EXPECT_EQ(info.getOutputIndex("active_token_ids").value(), 1u);
  EXPECT_FALSE(info.getOutputIndex("non_existent").isOk());
}

TEST(ModelInfoTest, AllElementDataTypesDiscovery) {
  const std::vector<std::pair<OnnxDataType, DataType>> test_types = {
      {OnnxDataType::FLOAT, DataType::Float32},     {OnnxDataType::FLOAT16, DataType::Float16},
      {OnnxDataType::BFLOAT16, DataType::BFloat16}, {OnnxDataType::DOUBLE, DataType::Float64},
      {OnnxDataType::INT8, DataType::Int8},         {OnnxDataType::UINT8, DataType::UInt8},
      {OnnxDataType::INT16, DataType::Int16},       {OnnxDataType::UINT16, DataType::UInt16},
      {OnnxDataType::INT32, DataType::Int32},       {OnnxDataType::UINT32, DataType::UInt32},
      {OnnxDataType::INT64, DataType::Int64},       {OnnxDataType::UINT64, DataType::UInt64},
      {OnnxDataType::BOOL, DataType::Bool},
  };

  for (const auto& [onnx_dt, expected_dt] : test_types) {
    OnnxModelSpec spec;
    spec.inputs = {TensorSpec{"x", onnx_dt, {DimSpec::fixed(2), DimSpec::fixed(4)}}};
    spec.outputs = {TensorSpec{"y", onnx_dt, {DimSpec::fixed(2), DimSpec::fixed(4)}}};
    spec.nodes = {encodeIdentityNode("x", "y", "id_node")};

    const auto bytes = buildOnnxModel(spec);
    const auto res = InferenceSession::create(bytes);
    ASSERT_TRUE(res.isOk()) << "Failed for DataType: " << dataTypeToString(expected_dt) << " - "
                            << res.error().message();

    const auto& session = res.value();
    EXPECT_EQ(session.getInputs()[0].dtype, expected_dt);
    EXPECT_EQ(session.getOutputs()[0].dtype, expected_dt);
  }
}

// ---------------------------------------------------------------------------
// Zero-Overhead Runtime Shape & Tensor Validation Tests
// ---------------------------------------------------------------------------

TEST(ModelInfoTest, RuntimeValidationOnFixedShapeModel) {
  const auto model_bytes = buildSqueezeNetSpecModel();
  const auto sess_res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(sess_res.isOk());
  const auto& session = sess_res.value();
  const auto& info = session.getModelInfo();

  // Valid input tensor [1, 3, 224, 224] Float32
  auto valid_tensor = Tensor::create(Shape{1, 3, 224, 224}, DataType::Float32).value();
  EXPECT_TRUE(info.validateInput(0, valid_tensor).isOk());
  EXPECT_TRUE(info.validateInput("data", valid_tensor).isOk());
  EXPECT_TRUE(session.validateInputs(std::span<const Tensor>(&valid_tensor, 1)).isOk());

  // Wrong batch size [4, 3, 224, 224] on fixed-batch model -> INVALID_DIMENSIONS
  auto wrong_batch = Tensor::create(Shape{4, 3, 224, 224}, DataType::Float32).value();
  auto status_batch = info.validateInput(0, wrong_batch);
  EXPECT_FALSE(status_batch.isOk());
  EXPECT_EQ(status_batch.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Wrong rank [3, 224, 224] -> INVALID_DIMENSIONS
  auto wrong_rank = Tensor::create(Shape{3, 224, 224}, DataType::Float32).value();
  auto status_rank = info.validateInput(0, wrong_rank);
  EXPECT_FALSE(status_rank.isOk());
  EXPECT_EQ(status_rank.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Wrong DataType (Int32 instead of Float32) -> INVALID_ARGUMENT
  auto wrong_dtype = Tensor::create(Shape{1, 3, 224, 224}, DataType::Int32).value();
  auto status_dtype = info.validateInput(0, wrong_dtype);
  EXPECT_FALSE(status_dtype.isOk());
  EXPECT_EQ(status_dtype.error().code(), StatusCode::INVALID_ARGUMENT);

  // Default-constructed (invalid) Tensor -> INVALID_ARGUMENT
  Tensor empty_tensor;
  auto status_empty = info.validateInput(0, empty_tensor);
  EXPECT_FALSE(status_empty.isOk());
  EXPECT_EQ(status_empty.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(ModelInfoTest, RuntimeValidationOnDynamicBatchModel) {
  const auto model_bytes = buildDynamicResNet50SpecModel();
  const auto sess_res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(sess_res.isOk());
  const auto& session = sess_res.value();

  // Dynamic batch accepts B = 1, 8, 32
  for (int64_t batch_size : {1, 8, 32}) {
    auto tensor = Tensor::create(Shape{batch_size, 3, 224, 224}, DataType::Float32).value();
    EXPECT_TRUE(session.validateInputs(std::span<const Tensor>(&tensor, 1)).isOk());
  }

  // Dynamic dimension = 0 is rejected at runtime -> INVALID_DIMENSIONS
  auto zero_batch = Tensor::create(Shape{0, 3, 224, 224}, DataType::Float32).value();
  auto status_zero = session.validateInputs(std::span<const Tensor>(&zero_batch, 1));
  EXPECT_FALSE(status_zero.isOk());
  EXPECT_EQ(status_zero.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Mismatched spatial dimension [8, 3, 128, 128] -> INVALID_DIMENSIONS
  auto wrong_spatial = Tensor::create(Shape{8, 3, 128, 128}, DataType::Float32).value();
  auto status_spatial = session.validateInputs(std::span<const Tensor>(&wrong_spatial, 1));
  EXPECT_FALSE(status_spatial.isOk());
  EXPECT_EQ(status_spatial.error().code(), StatusCode::INVALID_DIMENSIONS);
}

TEST(ModelInfoTest, CrossInputSymbolicDimensionConsistencyValidation) {
  const auto model_bytes = buildDynamicTransformerSpecModel();
  const auto sess_res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(sess_res.isOk());
  const auto& session = sess_res.value();

  // Matching batch_size=4, sequence_length=16 across all 3 inputs
  auto ids_ok = Tensor::create(Shape{4, 16}, DataType::Int64).value();
  auto mask_ok = Tensor::create(Shape{4, 16}, DataType::Int64).value();
  auto emb_ok = Tensor::create(Shape{4, 16, 768}, DataType::Float32).value();
  std::vector<Tensor> valid_inputs = {ids_ok, mask_ok, emb_ok};
  EXPECT_TRUE(session.validateInputs(valid_inputs).isOk());

  // Mismatched batch_size (4 vs 8) across inputs sharing symbolic dim "batch_size"
  auto mask_bad_batch = Tensor::create(Shape{8, 16}, DataType::Int64).value();
  std::vector<Tensor> bad_batch_inputs = {ids_ok, mask_bad_batch, emb_ok};
  auto status_batch = session.validateInputs(bad_batch_inputs);
  EXPECT_FALSE(status_batch.isOk());
  EXPECT_EQ(status_batch.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Mismatched sequence_length (16 vs 32) across inputs sharing symbolic dim "sequence_length"
  auto emb_bad_seq = Tensor::create(Shape{4, 32, 768}, DataType::Float32).value();
  std::vector<Tensor> bad_seq_inputs = {ids_ok, mask_ok, emb_bad_seq};
  auto status_seq = session.validateInputs(bad_seq_inputs);
  EXPECT_FALSE(status_seq.isOk());
  EXPECT_EQ(status_seq.error().code(), StatusCode::INVALID_DIMENSIONS);

  // Wrong number of inputs (2 instead of 3)
  std::vector<Tensor> wrong_count = {ids_ok, mask_ok};
  auto status_count = session.validateInputs(wrong_count);
  EXPECT_FALSE(status_count.isOk());
  EXPECT_EQ(status_count.error().code(), StatusCode::INVALID_ARGUMENT);
}

TEST(ModelInfoTest, StringFormattingAndQueryEdgeCases) {
  const auto model_bytes = buildDynamicResNet50SpecModel();
  const auto sess_res = InferenceSession::create(model_bytes);
  ASSERT_TRUE(sess_res.isOk());
  const auto& session = sess_res.value();

  const std::string model_str = session.getModelInfo().toString();
  EXPECT_NE(model_str.find("resnet50_v2_dynamic"), std::string::npos);
  EXPECT_NE(model_str.find("batch_size(-1)"), std::string::npos);
  EXPECT_NE(model_str.find("Float32"), std::string::npos);

  EXPECT_FALSE(session.getInputInfo(99).isOk());
  EXPECT_FALSE(session.getInputInfo("unknown_input").isOk());
  EXPECT_FALSE(session.getOutputInfo(99).isOk());
  EXPECT_FALSE(session.getOutputInfo("unknown_output").isOk());

  // Uninitialized session edge cases
  InferenceSession invalid_sess;
  EXPECT_TRUE(invalid_sess.getInputs().empty());
  EXPECT_TRUE(invalid_sess.getOutputs().empty());
  EXPECT_FALSE(invalid_sess.getInputInfo(0).isOk());
  EXPECT_FALSE(invalid_sess.getInputInfo("input").isOk());
  EXPECT_FALSE(invalid_sess.getOutputInfo(0).isOk());
  EXPECT_FALSE(invalid_sess.getOutputInfo("logits").isOk());
  EXPECT_FALSE(invalid_sess.validateInputs({}).isOk());
}

}  // namespace
