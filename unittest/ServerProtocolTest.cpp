#include <gtest/gtest.h>
#include <json/json.h>

#include <cmath>
#include <pyreth/memory_pool.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/result.hpp>
#include <pyreth/server/protocol.hpp>
#include <pyreth/tensor.hpp>
#include <string>
#include <vector>

using namespace pyreth;
using namespace pyreth::server;

TEST(ServerProtocolTest, KServeJsonParsing) {
  const std::string json_payload = R"({
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, 4],
        "datatype": "FP32",
        "data": [1.0, 2.5, 3.25, 4.125]
      }
    ]
  })";

  auto res = parseKServeRequest(json_payload);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& req = res.value();
  ASSERT_EQ(req.inputs.size(), 1u);
  EXPECT_EQ(req.inputs[0].name, "input_ids");
  EXPECT_EQ(req.inputs[0].datatype, DataType::Float32);
  EXPECT_EQ(req.inputs[0].shape.dims(), (std::vector<int64_t>{1, 4}));

  const auto& tensor = req.inputs[0].tensor;
  ASSERT_TRUE(tensor.isValid());
  EXPECT_EQ(tensor.numElements(), 4);
  EXPECT_EQ(tensor.dtype(), DataType::Float32);

  const float* data = tensor.data<float>();
  ASSERT_NE(data, nullptr);
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[1], 2.5f);
  EXPECT_FLOAT_EQ(data[2], 3.25f);
  EXPECT_FLOAT_EQ(data[3], 4.125f);

  // Test convenience overload parseKServeJson
  std::vector<std::string> names;
  auto tensors_res = parseKServeJson(json_payload, &names);
  ASSERT_TRUE(tensors_res.isOk());
  ASSERT_EQ(tensors_res.value().size(), 1u);
  ASSERT_EQ(names.size(), 1u);
  EXPECT_EQ(names[0], "input_ids");
}

TEST(ServerProtocolTest, KServeJsonSerialization) {
  auto tensor_res = Tensor::create(Shape{1, 2}, DataType::Float32);
  ASSERT_TRUE(tensor_res.isOk());
  Tensor tensor = std::move(tensor_res.value());
  tensor.data<float>()[0] = 0.42f;
  tensor.data<float>()[1] = 0.98f;

  std::vector<std::string> names{"output_0"};
  std::vector<Tensor> tensors;
  tensors.push_back(tensor);

  auto json_res = serializeKServeJson("my_model", names, tensors);
  ASSERT_TRUE(json_res.isOk()) << json_res.error().message();

  Json::CharReaderBuilder builder;
  Json::Value root;
  std::string errs;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  bool ok = reader->parse(json_res.value().data(),
                          json_res.value().data() + json_res.value().size(), &root, &errs);
  ASSERT_TRUE(ok) << errs;

  EXPECT_EQ(root["model_name"].asString(), "my_model");
  ASSERT_TRUE(root["outputs"].isArray());
  ASSERT_EQ(root["outputs"].size(), 1u);

  const auto& out = root["outputs"][0];
  EXPECT_EQ(out["name"].asString(), "output_0");
  EXPECT_EQ(out["datatype"].asString(), "FP32");
  ASSERT_EQ(out["shape"].size(), 2u);
  EXPECT_EQ(out["shape"][0].asInt64(), 1);
  EXPECT_EQ(out["shape"][1].asInt64(), 2);

  ASSERT_EQ(out["data"].size(), 2u);
  EXPECT_NEAR(out["data"][0].asFloat(), 0.42f, 1e-5);
  EXPECT_NEAR(out["data"][1].asFloat(), 0.98f, 1e-5);
}

TEST(ServerProtocolTest, DataTypeConversions) {
  // Test FP32, FP64, INT32, INT64, INT8, UINT8, BOOL
  const std::string multi_dtype_json = R"({
    "inputs": [
      {"name": "fp32", "shape": [2], "datatype": "FP32", "data": [1.5, -2.5]},
      {"name": "fp64", "shape": [2], "datatype": "FP64", "data": [3.1415926535, -2.718281828]},
      {"name": "int32", "shape": [2], "datatype": "INT32", "data": [42, -42]},
      {"name": "int64", "shape": [2], "datatype": "INT64", "data": [9000000000, -9000000000]},
      {"name": "int8", "shape": [2], "datatype": "INT8", "data": [127, -128]},
      {"name": "uint8", "shape": [2], "datatype": "UINT8", "data": [0, 255]},
      {"name": "bool", "shape": [2], "datatype": "BOOL", "data": [true, false]}
    ]
  })";

  auto parse_res = parseKServeRequest(multi_dtype_json);
  ASSERT_TRUE(parse_res.isOk()) << parse_res.error().message();
  const auto& req = parse_res.value();
  ASSERT_EQ(req.inputs.size(), 7u);

  // FP32
  EXPECT_EQ(req.inputs[0].datatype, DataType::Float32);
  EXPECT_FLOAT_EQ(req.inputs[0].tensor.data<float>()[0], 1.5f);
  EXPECT_FLOAT_EQ(req.inputs[0].tensor.data<float>()[1], -2.5f);

  // FP64
  EXPECT_EQ(req.inputs[1].datatype, DataType::Float64);
  EXPECT_DOUBLE_EQ(req.inputs[1].tensor.data<double>()[0], 3.1415926535);
  EXPECT_DOUBLE_EQ(req.inputs[1].tensor.data<double>()[1], -2.718281828);

  // INT32
  EXPECT_EQ(req.inputs[2].datatype, DataType::Int32);
  EXPECT_EQ(req.inputs[2].tensor.data<int32_t>()[0], 42);
  EXPECT_EQ(req.inputs[2].tensor.data<int32_t>()[1], -42);

  // INT64
  EXPECT_EQ(req.inputs[3].datatype, DataType::Int64);
  EXPECT_EQ(req.inputs[3].tensor.data<int64_t>()[0], 9000000000LL);
  EXPECT_EQ(req.inputs[3].tensor.data<int64_t>()[1], -9000000000LL);

  // INT8
  EXPECT_EQ(req.inputs[4].datatype, DataType::Int8);
  EXPECT_EQ(req.inputs[4].tensor.data<int8_t>()[0], 127);
  EXPECT_EQ(req.inputs[4].tensor.data<int8_t>()[1], -128);

  // UINT8
  EXPECT_EQ(req.inputs[5].datatype, DataType::UInt8);
  EXPECT_EQ(req.inputs[5].tensor.data<uint8_t>()[0], 0);
  EXPECT_EQ(req.inputs[5].tensor.data<uint8_t>()[1], 255);

  // BOOL
  EXPECT_EQ(req.inputs[6].datatype, DataType::Bool);
  EXPECT_TRUE(req.inputs[6].tensor.data<bool>()[0]);
  EXPECT_FALSE(req.inputs[6].tensor.data<bool>()[1]);

  // Round-trip serialization
  auto names = req.extractNames();
  auto tensors = req.extractTensors();
  auto serialized_res = serializeKServeJson("test_dtypes", names, tensors);
  ASSERT_TRUE(serialized_res.isOk());

  // Re-parse serialized output
  auto reparse_res = parseKServeRequest(serialized_res.value());
  // Because response has "outputs", let's replace "outputs" with "inputs" to test round-trip parse
  std::string reparse_json = serialized_res.value();
  auto pos = reparse_json.find("\"outputs\"");
  ASSERT_NE(pos, std::string::npos);
  reparse_json.replace(pos, 9, "\"inputs\"");

  auto roundtrip_res = parseKServeRequest(reparse_json);
  ASSERT_TRUE(roundtrip_res.isOk()) << roundtrip_res.error().message();
  EXPECT_EQ(roundtrip_res.value().inputs.size(), 7u);
}

TEST(ServerProtocolTest, MalformedJsonHandling) {
  // Empty payload
  {
    auto res = parseKServeRequest("");
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Invalid JSON syntax
  {
    auto res = parseKServeRequest("{ invalid json ");
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Not a JSON object (array root)
  {
    auto res = parseKServeRequest("[1, 2, 3]");
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
  }
}

TEST(ServerProtocolTest, SchemaValidationErrors) {
  // Missing required top-level inputs array
  {
    const std::string payload = R"({"model_name": "test"})";
    auto res = parseKServeRequest(payload);
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(res.error().message().find("Missing required 'inputs'"), std::string::npos);
  }

  // Missing mandatory fields in input descriptor
  {
    const std::string payload = R"({
      "inputs": [
        {"name": "x", "datatype": "FP32"}
      ]
    })";
    auto res = parseKServeRequest(payload);
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
  }

  // Element count mismatch between shape product and data array length
  {
    const std::string payload = R"({
      "inputs": [
        {"name": "x", "shape": [2, 3], "datatype": "FP32", "data": [1.0, 2.0]}
      ]
    })";
    auto res = parseKServeRequest(payload);
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_DIMENSIONS);
    EXPECT_NE(res.error().message().find("Element count mismatch"), std::string::npos);
  }

  // Unsupported datatype string
  {
    const std::string payload = R"({
      "inputs": [
        {"name": "x", "shape": [1], "datatype": "INT128", "data": [42]}
      ]
    })";
    auto res = parseKServeRequest(payload);
    EXPECT_FALSE(res.isOk());
    EXPECT_EQ(res.error().code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(res.error().message().find("Unsupported datatype"), std::string::npos);
  }
}

TEST(ServerProtocolTest, ArenaMemoryAllocation) {
  ArenaAllocator arena(64 * 1024);
  EXPECT_EQ(arena.bytesAllocated(), 0u);

  const std::string payload = R"({
    "inputs": [
      {
        "name": "arena_tensor",
        "shape": [4, 4],
        "datatype": "FP32",
        "data": [
          1, 2, 3, 4,
          5, 6, 7, 8,
          9, 10, 11, 12,
          13, 14, 15, 16
        ]
      }
    ]
  })";

  auto res = parseKServeRequest(payload, arena);
  ASSERT_TRUE(res.isOk()) << res.error().message();

  const auto& tensor = res.value().inputs[0].tensor;
  EXPECT_EQ(tensor.storageType(), StorageType::ARENA);
  EXPECT_GT(arena.bytesAllocated(), 0u);
  EXPECT_TRUE(tensor.isAligned(64));

  const float* data = tensor.data<float>();
  EXPECT_FLOAT_EQ(data[0], 1.0f);
  EXPECT_FLOAT_EQ(data[15], 16.0f);
}
