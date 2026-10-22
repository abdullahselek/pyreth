#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <latch>
#include <numeric>
#include <pyreth/dynamic_batcher.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/tensor.hpp>
#include <pyreth/thread_pool.hpp>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using namespace pyreth;

// ---------------------------------------------------------------------------
// Minimal ONNX Protobuf Builder Helpers for Dynamic Batching Tests
// ---------------------------------------------------------------------------

enum class OnnxDataType : uint32_t {
  FLOAT = 1,
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
  model.writeVarintField(1, 9);
  model.writeStringField(2, "pyreth_dynamic_batcher_test");
  model.writeSubmessage(7, graph);
  model.writeSubmessage(8, opset);
  return model.bytes();
}

// 1. Dynamic-Batch MLP Model: y = Relu(MatMul(x, W) + b)
//    x: [batch_size, 4] -> y: [batch_size, 3]
std::vector<uint8_t> buildMlpModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(4)}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::fixed(3)}}};

  const int64_t w_dims[] = {4, 3};
  const float w_vals[] = {
      1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -0.5f, 2.0f, 0.0f, 1.0f, -1.0f, 1.0f, 2.0f,
  };
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
      encodeNode(add_in, add_out, "add_node", "Add"),
      encodeNode(relu_in, relu_out, "relu_node", "Relu"),
  };
  return buildModelProto("mlp_graph", inputs, outputs, nodes, initializers);
}

void computeExpectedMlpRow(const float* x, float* y) {
  const float w[4][3] = {
      {1.0f, -1.0f, 0.5f},
      {0.0f, 2.0f, -0.5f},
      {2.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, 2.0f},
  };
  const float b[3] = {0.5f, -1.0f, 0.0f};
  for (int j = 0; j < 3; ++j) {
    float acc = b[j];
    for (int i = 0; i < 4; ++i) {
      acc += x[i] * w[i][j];
    }
    y[j] = std::max(0.0f, acc);
  }
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

// 3. Dynamic Sequence Model: y = x (Float32 [batch_size, seq_len])
std::vector<uint8_t> buildDynamicSequenceIdentityModel() {
  std::vector<TensorSpec> inputs = {
      {"x", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::dynamic("seq_len")}}};
  std::vector<TensorSpec> outputs = {
      {"y", OnnxDataType::FLOAT, {DimSpec::dynamic("batch_size"), DimSpec::dynamic("seq_len")}}};
  std::string_view in_names[] = {"x"};
  std::string_view out_names[] = {"y"};
  std::vector<ProtoBuilder> nodes = {encodeNode(in_names, out_names, "id_node", "Identity")};
  return buildModelProto("seq_identity_graph", inputs, outputs, nodes);
}

}  // namespace

// ===========================================================================
// 1. Lifecycle, Configuration & Validation Tests
// ===========================================================================

TEST(DynamicBatcherTest, CreationLifecycleAndMoveSemantics) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();

  // Default constructed batcher is invalid
  DynamicBatcher empty_batcher;
  EXPECT_FALSE(empty_batcher.isValid());
  EXPECT_TRUE(empty_batcher.isShutdown());

  // Invalid configuration rejected by factory
  DynamicBatcherConfig bad_cfg;
  bad_cfg.setMaxBatchSize(0);
  auto bad_res = DynamicBatcher::create(session, bad_cfg);
  EXPECT_FALSE(bad_res.isOk());
  EXPECT_EQ(bad_res.error().code(), StatusCode::INVALID_ARGUMENT);

  DynamicBatcherConfig bad_q_cfg;
  bad_q_cfg.setMaxQueueCapacity(0);
  EXPECT_FALSE(DynamicBatcher::create(session, bad_q_cfg).isOk());

  InferenceSession invalid_session;
  EXPECT_FALSE(DynamicBatcher::create(invalid_session).isOk());

  // Valid creation and move semantics
  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(8).setMaxBatchDelayMs(10).setMaxQueueCapacity(128).setNumWorkers(2);

  auto batcher_res = DynamicBatcher::create(session, cfg);
  ASSERT_TRUE(batcher_res.isOk());
  DynamicBatcher batcher = std::move(batcher_res).value();
  EXPECT_TRUE(batcher.isValid());
  EXPECT_FALSE(batcher.isShutdown());
  EXPECT_EQ(batcher.getConfig().max_batch_size, 8u);
  EXPECT_EQ(batcher.getConfig().max_batch_delay_ms, 10u);

  // Move construct
  DynamicBatcher moved_batcher(std::move(batcher));
  EXPECT_TRUE(moved_batcher.isValid());
  EXPECT_FALSE(batcher.isValid());

  auto x = Tensor::create(Shape{1, 4}, DataType::Float32).value();
  x.fill<float>(1.0f);
  auto out_res = moved_batcher.run({x});
  ASSERT_TRUE(out_res.isOk());
  ASSERT_EQ(out_res.value().size(), 1u);
  EXPECT_EQ(out_res.value()[0].shape(), (Shape{1, 3}));

  moved_batcher.shutdown();
  EXPECT_TRUE(moved_batcher.isShutdown());
}

// ===========================================================================
// 2. Full Batch Coalescing & Zero-Wait Dispatch
// ===========================================================================

TEST(DynamicBatcherTest, CoalescesFullBatchImmediatelyBeforeTimeoutWindow) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();
  ThreadPool pool(2);
  MemoryPool mem_pool(8, 256 * 1024);

  // Set a very long timeout (500ms) so that if the batcher waited for the timeout,
  // the test would take >= 500ms. Reaching max_batch_size=8 must flush immediately.
  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(8).setMaxBatchDelayMs(500);

  DynamicBatcher batcher(session, pool, mem_pool, cfg);

  constexpr size_t kNumRequests = 8;
  std::vector<Tensor> inputs;
  inputs.reserve(kNumRequests);
  for (size_t i = 0; i < kNumRequests; ++i) {
    auto t = Tensor::create(Shape{1, 4}, DataType::Float32).value();
    float* p = t.data<float>();
    p[0] = static_cast<float>(i) + 1.0f;
    p[1] = -0.5f * static_cast<float>(i);
    p[2] = 0.25f * static_cast<float>(i);
    p[3] = 1.5f;
    inputs.push_back(std::move(t));
  }

  const auto start_time = std::chrono::steady_clock::now();

  std::vector<TaskHandle<Result<std::vector<Tensor>>>> handles;
  handles.reserve(kNumRequests);
  for (size_t i = 0; i < kNumRequests; ++i) {
    auto h_res = batcher.submit({inputs[i]});
    ASSERT_TRUE(h_res.isOk()) << h_res.error().message();
    handles.push_back(std::move(h_res).value());
  }

  for (size_t i = 0; i < kNumRequests; ++i) {
    auto out_res = handles[i].get();
    ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
    ASSERT_EQ(out_res.value().size(), 1u);

    const Tensor& y = out_res.value()[0];
    EXPECT_EQ(y.shape(), (Shape{1, 3}));
    EXPECT_TRUE(y.isContiguous());
    EXPECT_EQ(y.storageType(), StorageType::POOLED);

    float expected[3];
    computeExpectedMlpRow(inputs[i].data<float>(), expected);
    const float* actual = y.data<float>();
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(actual[j], expected[j], 1e-5f);
    }
  }

  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now() - start_time)
                              .count();
  EXPECT_LT(elapsed_ms, 250);

  auto m = batcher.metrics();
  EXPECT_EQ(m.requests_submitted, kNumRequests);
  EXPECT_EQ(m.requests_completed, kNumRequests);
  EXPECT_EQ(m.total_batched_samples, kNumRequests);
  EXPECT_GE(m.full_batches, 1u);
}

// ===========================================================================
// 3. Partial Batch Flush on Timeout & Explicit Flush
// ===========================================================================

TEST(DynamicBatcherTest, PartialBatchFlushesOnLatencyWindowExpiration) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();

  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(16).setMaxBatchDelayMs(10);

  DynamicBatcher batcher(session, cfg);

  // Submit only 3 requests (< max_batch_size = 16); they should flush after ~10ms
  constexpr size_t kPartialCount = 3;
  std::vector<TaskHandle<Result<std::vector<Tensor>>>> handles;
  std::vector<Tensor> req_inputs;

  for (size_t i = 0; i < kPartialCount; ++i) {
    auto t = Tensor::create(Shape{1, 4}, DataType::Float32).value();
    float* p = t.data<float>();
    p[0] = 2.0f * static_cast<float>(i + 1);
    p[1] = 1.0f;
    p[2] = -1.0f;
    p[3] = 0.5f;
    req_inputs.push_back(t);

    auto h_res = batcher.submit({t});
    ASSERT_TRUE(h_res.isOk());
    handles.push_back(std::move(h_res).value());
  }

  for (size_t i = 0; i < kPartialCount; ++i) {
    auto res = handles[i].get();
    ASSERT_TRUE(res.isOk());
    const Tensor& y = res.value()[0];
    EXPECT_EQ(y.shape(), (Shape{1, 3}));

    float expected[3];
    computeExpectedMlpRow(req_inputs[i].data<float>(), expected);
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(y.data<float>()[j], expected[j], 1e-5f);
    }
  }

  auto m = batcher.metrics();
  EXPECT_EQ(m.requests_submitted, kPartialCount);
  EXPECT_EQ(m.requests_completed, kPartialCount);
  EXPECT_EQ(m.total_batched_samples, kPartialCount);
  EXPECT_GE(m.timeout_batches, 1u);
}

// ===========================================================================
// 4. Multi-Input / Multi-Output Demultiplexing & Named Inputs
// ===========================================================================

TEST(DynamicBatcherTest, MultiInputMultiOutputAndVariableRequestBatchSizes) {
  auto model_bytes = buildArithmeticModel();
  auto session = InferenceSession::create(model_bytes).value();

  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(8).setMaxBatchDelayMs(15);
  DynamicBatcher batcher(session, cfg);

  // Submit 3 requests with different batch sizes along axis 0: B=1, B=3, B=2 (total = 6)
  const int64_t batch_sizes[] = {1, 3, 2};
  std::vector<Tensor> a_tensors;
  std::vector<Tensor> b_tensors;
  std::vector<TaskHandle<Result<std::vector<Tensor>>>> handles;

  for (size_t req_idx = 0; req_idx < 3; ++req_idx) {
    const int64_t b_sz = batch_sizes[req_idx];
    auto a = Tensor::create(Shape{b_sz, 4}, DataType::Float32).value();
    auto b = Tensor::create(Shape{b_sz, 4}, DataType::Float32).value();

    for (int64_t elem = 0; elem < b_sz * 4; ++elem) {
      a.data<float>()[elem] = static_cast<float>((req_idx + 1) * 10 + elem);
      b.data<float>()[elem] = static_cast<float>(elem + 1) * 0.5f;
    }

    a_tensors.push_back(a);
    b_tensors.push_back(b);

    if (req_idx == 1) {
      // Use named_inputs overload for request 1
      std::unordered_map<std::string, Tensor> named = {{"b", b}, {"a", a}};
      auto h_res = batcher.submit(named);
      ASSERT_TRUE(h_res.isOk());
      handles.push_back(std::move(h_res).value());
    } else {
      auto h_res = batcher.submit({a, b});
      ASSERT_TRUE(h_res.isOk());
      handles.push_back(std::move(h_res).value());
    }
  }

  for (size_t req_idx = 0; req_idx < 3; ++req_idx) {
    const int64_t b_sz = batch_sizes[req_idx];
    auto out_res = handles[req_idx].get();
    ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
    const auto& outs = out_res.value();
    ASSERT_EQ(outs.size(), 2u);

    const Tensor& sum_t = outs[0];
    const Tensor& prod_t = outs[1];
    EXPECT_EQ(sum_t.shape(), (Shape{b_sz, 4}));
    EXPECT_EQ(prod_t.shape(), (Shape{b_sz, 4}));

    const float* a_ptr = a_tensors[req_idx].data<float>();
    const float* b_ptr = b_tensors[req_idx].data<float>();
    const float* sum_ptr = sum_t.data<float>();
    const float* prod_ptr = prod_t.data<float>();

    for (int64_t elem = 0; elem < b_sz * 4; ++elem) {
      EXPECT_FLOAT_EQ(sum_ptr[elem], a_ptr[elem] + b_ptr[elem]);
      EXPECT_FLOAT_EQ(prod_ptr[elem], a_ptr[elem] * b_ptr[elem]);
    }
  }

  EXPECT_EQ(batcher.metrics().total_batched_samples, 6u);
}

// ===========================================================================
// 5. Dynamic Non-Batch Dimension Separation (Sequence Length Bucketing)
// ===========================================================================

TEST(DynamicBatcherTest, IncompatibleTrailingDimensionsSplitIntoSeparateBatches) {
  auto model_bytes = buildDynamicSequenceIdentityModel();
  auto session = InferenceSession::create(model_bytes).value();

  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(8).setMaxBatchDelayMs(50);
  DynamicBatcher batcher(session, cfg);

  // Request 0 & 1 have seq_len = 5; Request 2 has seq_len = 9
  auto t0 = Tensor::create(Shape{1, 5}, DataType::Float32).value();
  auto t1 = Tensor::create(Shape{2, 5}, DataType::Float32).value();
  auto t2 = Tensor::create(Shape{1, 9}, DataType::Float32).value();
  t0.fill<float>(1.0f);
  t1.fill<float>(2.0f);
  t2.fill<float>(3.0f);

  auto h0 = batcher.submit({t0}).value();
  auto h1 = batcher.submit({t1}).value();
  auto h2 = batcher.submit({t2}).value();

  batcher.flush();

  auto r0 = h0.get();
  auto r1 = h1.get();
  auto r2 = h2.get();
  ASSERT_TRUE(r0.isOk());
  ASSERT_TRUE(r1.isOk());
  ASSERT_TRUE(r2.isOk());

  EXPECT_EQ(r0.value()[0].shape(), (Shape{1, 5}));
  EXPECT_EQ(r1.value()[0].shape(), (Shape{2, 5}));
  EXPECT_EQ(r2.value()[0].shape(), (Shape{1, 9}));
  EXPECT_FLOAT_EQ(r0.value()[0].data<float>()[0], 1.0f);
  EXPECT_FLOAT_EQ(r1.value()[0].data<float>()[0], 2.0f);
  EXPECT_FLOAT_EQ(r2.value()[0].data<float>()[0], 3.0f);
}

// ===========================================================================
// 6. Concurrent Asynchronous Requests Simulation (Stress Test)
// ===========================================================================

TEST(DynamicBatcherTest, ConcurrentAsynchronousRequestsSimulation) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();
  ThreadPool pool(4);
  MemoryPool mem_pool(16, 256 * 1024);

  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(16).setMaxBatchDelayMs(5).setMaxQueueCapacity(1024);

  DynamicBatcher batcher(session, pool, mem_pool, cfg);

  constexpr size_t kNumProducers = 8;
  constexpr size_t kRequestsPerProducer = 32;
  constexpr size_t kTotalRequests = kNumProducers * kRequestsPerProducer;

  std::latch start_latch(kNumProducers);
  std::atomic<size_t> success_count{0};
  std::vector<std::jthread> producers;
  producers.reserve(kNumProducers);

  for (size_t tid = 0; tid < kNumProducers; ++tid) {
    producers.emplace_back([&, tid]() {
      ArenaAllocator local_arena(64 * 1024);
      start_latch.arrive_and_wait();

      std::vector<TaskHandle<Result<std::vector<Tensor>>>> handles;
      std::vector<Tensor> local_inputs;
      handles.reserve(kRequestsPerProducer);
      local_inputs.reserve(kRequestsPerProducer);

      for (size_t r = 0; r < kRequestsPerProducer; ++r) {
        auto in_t = Tensor::create(Shape{1, 4}, DataType::Float32, local_arena).value();
        float* ptr = in_t.data<float>();
        const float seed = static_cast<float>(tid * 100 + r);
        ptr[0] = seed * 0.1f;
        ptr[1] = -seed * 0.05f + 1.0f;
        ptr[2] = seed * 0.02f - 0.5f;
        ptr[3] = 0.75f;

        local_inputs.push_back(in_t);
        auto h_res = batcher.submit({in_t});
        ASSERT_TRUE(h_res.isOk()) << h_res.error().message();
        handles.push_back(std::move(h_res).value());
      }

      for (size_t r = 0; r < kRequestsPerProducer; ++r) {
        auto out_res = handles[r].get();
        ASSERT_TRUE(out_res.isOk()) << out_res.error().message();
        const Tensor& y = out_res.value()[0];
        ASSERT_EQ(y.shape(), (Shape{1, 3}));

        float expected[3];
        computeExpectedMlpRow(local_inputs[r].data<float>(), expected);
        const float* actual = y.data<float>();
        for (int j = 0; j < 3; ++j) {
          EXPECT_NEAR(actual[j], expected[j], 1e-4f);
        }
        success_count.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  producers.clear();  // Join all producer threads
  batcher.flush();

  EXPECT_EQ(success_count.load(), kTotalRequests);

  const auto m = batcher.metrics();
  EXPECT_EQ(m.requests_submitted, kTotalRequests);
  EXPECT_EQ(m.requests_completed, kTotalRequests);
  EXPECT_EQ(m.requests_rejected, 0u);
  EXPECT_EQ(m.total_batched_samples, kTotalRequests);
  EXPECT_LT(m.batches_dispatched, kTotalRequests);
  EXPECT_GT(m.averageBatchSize(), 1.0);
}

// ===========================================================================
// 7. Latency vs Throughput Curve Test Under High Request Arrival Rates
// ===========================================================================

TEST(DynamicBatcherTest, LatencyVsThroughputCurveUnderHighArrivalRate) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();
  ThreadPool pool(4);
  MemoryPool mem_pool(16, 512 * 1024);

  const std::vector<size_t> batch_sizes_to_test = {1, 4, 8, 16};
  constexpr size_t kNumProducers = 8;
  constexpr size_t kRequestsPerProducer = 32;
  constexpr size_t kTotalRequests = kNumProducers * kRequestsPerProducer;

  struct CurvePoint {
    size_t max_batch_size{0};
    double throughput_rps{0.0};
    double avg_batch_size{0.0};
    double p50_us{0.0};
    double p95_us{0.0};
    double p99_us{0.0};
    uint64_t batches_dispatched{0};
  };

  std::vector<CurvePoint> curve;
  curve.reserve(batch_sizes_to_test.size());

  for (size_t max_bs : batch_sizes_to_test) {
    DynamicBatcherConfig cfg;
    cfg.setMaxBatchSize(max_bs).setMaxBatchDelayMs(2).setMaxQueueCapacity(2048);

    DynamicBatcher batcher(session, pool, mem_pool, cfg);

    std::vector<double> latencies_us(kTotalRequests, 0.0);
    std::latch start_latch(kNumProducers);

    const auto wall_start = std::chrono::steady_clock::now();

    {
      std::vector<std::jthread> clients;
      clients.reserve(kNumProducers);
      for (size_t p = 0; p < kNumProducers; ++p) {
        clients.emplace_back([&, p]() {
          auto input = Tensor::create(Shape{1, 4}, DataType::Float32).value();
          input.fill<float>(1.0f);

          start_latch.arrive_and_wait();

          struct InFlight {
            size_t global_idx;
            std::chrono::steady_clock::time_point t0;
            TaskHandle<Result<std::vector<Tensor>>> handle;
          };
          std::vector<InFlight> in_flight;
          in_flight.reserve(kRequestsPerProducer);

          for (size_t r = 0; r < kRequestsPerProducer; ++r) {
            const size_t idx = p * kRequestsPerProducer + r;
            const auto t0 = std::chrono::steady_clock::now();
            auto h_res = batcher.submit({input});
            ASSERT_TRUE(h_res.isOk());
            in_flight.push_back(InFlight{idx, t0, std::move(h_res).value()});
          }

          for (auto& item : in_flight) {
            auto out_res = item.handle.get();
            const auto t1 = std::chrono::steady_clock::now();
            ASSERT_TRUE(out_res.isOk());
            latencies_us[item.global_idx] =
                std::chrono::duration<double, std::micro>(t1 - item.t0).count();
          }
        });
      }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    std::sort(latencies_us.begin(), latencies_us.end());
    const auto percentile = [&](double pct) -> double {
      const size_t idx = std::min(
          latencies_us.size() - 1,
          static_cast<size_t>(std::ceil((pct / 100.0) * static_cast<double>(latencies_us.size()))) -
              1);
      return latencies_us[idx];
    };

    const auto m = batcher.metrics();
    CurvePoint pt;
    pt.max_batch_size = max_bs;
    pt.throughput_rps = static_cast<double>(kTotalRequests) / std::max(wall_sec, 1e-9);
    pt.avg_batch_size = m.averageBatchSize();
    pt.p50_us = percentile(50.0);
    pt.p95_us = percentile(95.0);
    pt.p99_us = percentile(99.0);
    pt.batches_dispatched = m.batches_dispatched;
    curve.push_back(pt);

    EXPECT_EQ(m.requests_completed, kTotalRequests);
    EXPECT_GT(pt.throughput_rps, 0.0);
    EXPECT_GT(pt.p50_us, 0.0);
    EXPECT_GE(pt.p95_us, pt.p50_us);
    EXPECT_GE(pt.p99_us, pt.p95_us);
  }

  // Verify that larger max_batch_size achieves significantly fewer ORT dispatches
  // and higher average batch size under concurrent load than max_batch_size = 1.
  EXPECT_DOUBLE_EQ(curve.front().avg_batch_size, 1.0);
  EXPECT_EQ(curve.front().batches_dispatched, kTotalRequests);
  EXPECT_GT(curve.back().avg_batch_size, 1.0);
  EXPECT_LT(curve.back().batches_dispatched, curve.front().batches_dispatched);
}

// ===========================================================================
// 8. Error Handling, Queue Capacity & Shutdown Semantics
// ===========================================================================

TEST(DynamicBatcherTest, ErrorValidationAndShutdownRejection) {
  auto model_bytes = buildMlpModel();
  auto session = InferenceSession::create(model_bytes).value();

  DynamicBatcherConfig cfg;
  cfg.setMaxBatchSize(4).setMaxBatchDelayMs(10);
  DynamicBatcher batcher(session, cfg);

  // 1. Invalid feature dimension ([1, 5] instead of [1, 4])
  auto bad_dim = Tensor::create(Shape{1, 5}, DataType::Float32).value();
  auto res_dim = batcher.submit({bad_dim});
  EXPECT_FALSE(res_dim.isOk());
  EXPECT_EQ(res_dim.error().code(), StatusCode::INVALID_DIMENSIONS);

  // 2. Request batch size (8) exceeds max_batch_size (4)
  auto oversized = Tensor::create(Shape{8, 4}, DataType::Float32).value();
  auto res_over = batcher.submit({oversized});
  EXPECT_FALSE(res_over.isOk());
  EXPECT_EQ(res_over.error().code(), StatusCode::INVALID_DIMENSIONS);

  // 3. Invalid DataType (Int64 instead of Float32)
  auto bad_dtype = Tensor::create(Shape{1, 4}, DataType::Int64).value();
  auto res_dtype = batcher.submit({bad_dtype});
  EXPECT_FALSE(res_dtype.isOk());
  EXPECT_EQ(res_dtype.error().code(), StatusCode::INVALID_ARGUMENT);

  EXPECT_EQ(batcher.metrics().requests_rejected, 3u);

  // 4. Reset metrics
  batcher.resetMetrics();
  EXPECT_EQ(batcher.metrics().requests_rejected, 0u);

  // 5. Post-shutdown submission rejected with INTERNAL_ERROR
  batcher.shutdown();
  EXPECT_TRUE(batcher.isShutdown());

  auto valid_t = Tensor::create(Shape{1, 4}, DataType::Float32).value();
  auto post_shutdown_res = batcher.submit({valid_t});
  EXPECT_FALSE(post_shutdown_res.isOk());
  EXPECT_EQ(post_shutdown_res.error().code(), StatusCode::INTERNAL_ERROR);
}
