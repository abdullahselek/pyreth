#include <benchmark/benchmark.h>

#include <cstdlib>
#include <cstring>
#include <pyreth/dynamic_batcher.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/tensor.hpp>
#include <pyreth/thread_pool.hpp>
#include <pyreth/work_stealing_queue.hpp>
#include <vector>

namespace pyreth::benchmarks {
namespace {

// ===========================================================================
// Tiny Synthetic ONNX Protobuf Builder (Linear + ReLU: Y = Relu(X @ W + B))
// ===========================================================================

void appendVarint(std::vector<uint8_t>& buf, uint64_t val) {
  while (val >= 0x80) {
    buf.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
    val >>= 7;
  }
  buf.push_back(static_cast<uint8_t>(val));
}

void appendTag(std::vector<uint8_t>& buf, uint32_t field_num, uint32_t wire_type) {
  appendVarint(buf, (static_cast<uint64_t>(field_num) << 3) | wire_type);
}

void appendString(std::vector<uint8_t>& buf, uint32_t field_num, std::string_view s) {
  appendTag(buf, field_num, 2);
  appendVarint(buf, s.size());
  buf.insert(buf.end(), s.begin(), s.end());
}

void appendBytes(std::vector<uint8_t>& buf, uint32_t field_num, const std::vector<uint8_t>& sub) {
  appendTag(buf, field_num, 2);
  appendVarint(buf, sub.size());
  buf.insert(buf.end(), sub.begin(), sub.end());
}

void appendInt64(std::vector<uint8_t>& buf, uint32_t field_num, int64_t val) {
  appendTag(buf, field_num, 0);
  appendVarint(buf, static_cast<uint64_t>(val));
}

std::vector<uint8_t> encodeTensorShapeDim(int64_t dim_val, std::string_view dim_param = "") {
  std::vector<uint8_t> d;
  if (!dim_param.empty()) {
    appendString(d, 2, dim_param);
  } else {
    appendInt64(d, 1, dim_val);
  }
  return d;
}

std::vector<uint8_t> encodeValueInfo(std::string_view name, int32_t elem_type,
                                     const std::vector<std::vector<uint8_t>>& dims) {
  std::vector<uint8_t> shape_proto;
  for (const auto& dim : dims) {
    appendBytes(shape_proto, 1, dim);
  }
  std::vector<uint8_t> tensor_type;
  appendInt64(tensor_type, 1, elem_type);
  appendBytes(tensor_type, 2, shape_proto);

  std::vector<uint8_t> type_proto;
  appendBytes(type_proto, 1, tensor_type);

  std::vector<uint8_t> vi;
  appendString(vi, 1, name);
  appendBytes(vi, 2, type_proto);
  return vi;
}

std::vector<uint8_t> encodeFloatInitializer(std::string_view name, const std::vector<int64_t>& dims,
                                            const std::vector<float>& values) {
  std::vector<uint8_t> tp;
  for (int64_t d : dims) {
    appendInt64(tp, 1, d);
  }
  appendInt64(tp, 2, 1);  // FLOAT = 1
  appendString(tp, 8, name);

  std::vector<uint8_t> raw(values.size() * sizeof(float));
  std::memcpy(raw.data(), values.data(), raw.size());
  appendBytes(tp, 9, raw);
  return tp;
}

std::vector<uint8_t> encodeNode(std::string_view op_type, const std::vector<std::string_view>& ins,
                                const std::vector<std::string_view>& outs, std::string_view name) {
  std::vector<uint8_t> node;
  for (auto in : ins) {
    appendString(node, 1, in);
  }
  for (auto out : outs) {
    appendString(node, 2, out);
  }
  appendString(node, 3, name);
  appendString(node, 4, op_type);
  return node;
}

std::vector<uint8_t> buildBenchmarkMlpModel(int64_t in_features = 16, int64_t out_features = 8) {
  std::vector<uint8_t> graph;
  appendString(graph, 2, "bench_mlp_graph");

  std::vector<float> w_vals(static_cast<size_t>(in_features * out_features), 0.125f);
  std::vector<float> b_vals(static_cast<size_t>(out_features), 0.25f);

  appendBytes(graph, 5, encodeFloatInitializer("W", {in_features, out_features}, w_vals));
  appendBytes(graph, 5, encodeFloatInitializer("B", {out_features}, b_vals));

  appendBytes(graph, 1, encodeNode("MatMul", {"x", "W"}, {"xw"}, "matmul_node"));
  appendBytes(graph, 1, encodeNode("Add", {"xw", "B"}, {"xwb"}, "add_node"));
  appendBytes(graph, 1, encodeNode("Relu", {"xwb"}, {"y"}, "relu_node"));

  appendBytes(graph, 11,
              encodeValueInfo(
                  "x", 1, {encodeTensorShapeDim(-1, "batch"), encodeTensorShapeDim(in_features)}));
  appendBytes(graph, 12,
              encodeValueInfo(
                  "y", 1, {encodeTensorShapeDim(-1, "batch"), encodeTensorShapeDim(out_features)}));

  std::vector<uint8_t> opset;
  appendString(opset, 1, "");
  appendInt64(opset, 2, 13);

  std::vector<uint8_t> model;
  appendInt64(model, 1, 7);
  appendString(model, 2, "pyreth_benchmark");
  appendBytes(model, 7, graph);
  appendBytes(model, 8, opset);
  return model;
}

// ===========================================================================
// 1. Memory Allocation Micro-Benchmarks: ArenaAllocator vs std::aligned_alloc
// ===========================================================================

void BM_MallocFree_64B_Aligned_Single(benchmark::State& state) {
  const auto bytes = static_cast<size_t>(state.range(0));
  for (auto _ : state) {
#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(bytes, 64);
    benchmark::DoNotOptimize(ptr);
    _aligned_free(ptr);
#else
    void* ptr = nullptr;
    int rc = posix_memalign(&ptr, 64, bytes);
    benchmark::DoNotOptimize(rc);
    benchmark::DoNotOptimize(ptr);
    std::free(ptr);
#endif
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));
}
BENCHMARK(BM_MallocFree_64B_Aligned_Single)->RangeMultiplier(4)->Range(256, 65536);

void BM_ArenaAllocate_64B_Aligned_Single(benchmark::State& state) {
  const auto bytes = static_cast<size_t>(state.range(0));
  ArenaAllocator arena(std::max<size_t>(bytes * 4, DEFAULT_BLOCK_SIZE));

  for (auto _ : state) {
    void* ptr = arena.allocate(bytes, DEFAULT_ALIGNMENT);
    benchmark::DoNotOptimize(ptr);
    arena.reset();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(bytes));
}
BENCHMARK(BM_ArenaAllocate_64B_Aligned_Single)->RangeMultiplier(4)->Range(256, 65536);

void BM_MallocFree_Batch64_Allocations(benchmark::State& state) {
  constexpr size_t kBatchCount = 64;
  constexpr size_t kAllocSize = 1024;
  void* ptrs[kBatchCount];

  for (auto _ : state) {
    for (size_t i = 0; i < kBatchCount; ++i) {
#if defined(_MSC_VER)
      ptrs[i] = _aligned_malloc(kAllocSize, 64);
#else
      void* p = nullptr;
      int rc = posix_memalign(&p, 64, kAllocSize);
      benchmark::DoNotOptimize(rc);
      ptrs[i] = p;
#endif
    }
    benchmark::DoNotOptimize(ptrs);
    for (size_t i = 0; i < kBatchCount; ++i) {
#if defined(_MSC_VER)
      _aligned_free(ptrs[i]);
#else
      std::free(ptrs[i]);
#endif
    }
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kBatchCount));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(kBatchCount * kAllocSize));
}
BENCHMARK(BM_MallocFree_Batch64_Allocations);

void BM_ArenaAllocate_Batch64_Allocations(benchmark::State& state) {
  constexpr size_t kBatchCount = 64;
  constexpr size_t kAllocSize = 1024;
  ArenaAllocator arena(256 * 1024);
  void* ptrs[kBatchCount];

  for (auto _ : state) {
    for (size_t i = 0; i < kBatchCount; ++i) {
      ptrs[i] = arena.allocate(kAllocSize, DEFAULT_ALIGNMENT);
    }
    benchmark::DoNotOptimize(ptrs);
    arena.reset();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kBatchCount));
  state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(kBatchCount * kAllocSize));
}
BENCHMARK(BM_ArenaAllocate_Batch64_Allocations);

void BM_MemoryPool_AcquireAllocateRelease(benchmark::State& state) {
  MemoryPool pool(4, 64 * 1024);
  for (auto _ : state) {
    auto lease = pool.acquire();
    void* ptr = lease->allocate(1024, DEFAULT_ALIGNMENT);
    benchmark::DoNotOptimize(ptr);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_MemoryPool_AcquireAllocateRelease);

// ===========================================================================
// 2. Lock-Free WorkStealingQueue & ThreadPool Throughput Micro-Benchmarks
// ===========================================================================

void BM_WorkStealingQueue_PushPop_LIFO(benchmark::State& state) {
  WorkStealingQueue<int, 1024> queue;
  int v = 42;
  for (auto _ : state) {
    bool pushed = queue.push(v);
    auto popped = queue.pop();
    benchmark::DoNotOptimize(pushed);
    benchmark::DoNotOptimize(popped);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WorkStealingQueue_PushPop_LIFO);

void BM_WorkStealingQueue_PushSteal_FIFO(benchmark::State& state) {
  WorkStealingQueue<int, 1024> queue;
  int v = 42;
  for (auto _ : state) {
    bool pushed = queue.push(v);
    auto stolen = queue.steal();
    benchmark::DoNotOptimize(pushed);
    benchmark::DoNotOptimize(stolen);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_WorkStealingQueue_PushSteal_FIFO);

void BM_WorkStealingQueue_Burst256_PushPop(benchmark::State& state) {
  WorkStealingQueue<int, 1024> queue;
  constexpr int kBurst = 256;
  for (auto _ : state) {
    for (int i = 0; i < kBurst; ++i) {
      bool ok = queue.push(i);
      benchmark::DoNotOptimize(ok);
    }
    for (int i = 0; i < kBurst; ++i) {
      auto val = queue.pop();
      benchmark::DoNotOptimize(val);
    }
  }
  state.SetItemsProcessed(state.iterations() * kBurst);
}
BENCHMARK(BM_WorkStealingQueue_Burst256_PushPop);

void BM_ThreadPool_SubmitWait(benchmark::State& state) {
  ThreadPool pool(2);
  for (auto _ : state) {
    auto h = pool.submit([]() noexcept { return 1; }).value();
    int r = h.get();
    benchmark::DoNotOptimize(r);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_ThreadPool_SubmitWait);

// ===========================================================================
// 3. Tensor Creation & Zero-Copy View Operations
// ===========================================================================

void BM_TensorCreate_HeapOwned(benchmark::State& state) {
  const int64_t n = state.range(0);
  const Shape shape{n, 16};
  for (auto _ : state) {
    auto t = Tensor::create(shape, DataType::Float32).value();
    benchmark::DoNotOptimize(t.data());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TensorCreate_HeapOwned)->Arg(1)->Arg(16)->Arg(64);

void BM_TensorCreate_ArenaBacked(benchmark::State& state) {
  const int64_t n = state.range(0);
  const Shape shape{n, 16};
  ArenaAllocator arena(256 * 1024);
  for (auto _ : state) {
    {
      auto t = Tensor::create(shape, DataType::Float32, arena).value();
      benchmark::DoNotOptimize(t.data());
    }
    arena.reset();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TensorCreate_ArenaBacked)->Arg(1)->Arg(16)->Arg(64);

void BM_TensorFromBuffer_ZeroCopy(benchmark::State& state) {
  alignas(64) float raw_buffer[64 * 16]{};
  const Shape shape{64, 16};
  for (auto _ : state) {
    auto t = Tensor::fromBuffer(raw_buffer, shape, DataType::Float32).value();
    benchmark::DoNotOptimize(t.data());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TensorFromBuffer_ZeroCopy);

void BM_TensorSliceAndReshape_ZeroCopy(benchmark::State& state) {
  auto base = Tensor::create(Shape{64, 16}, DataType::Float32).value();
  for (auto _ : state) {
    auto sliced = base.slice(0, 8, 24, 1).value();
    auto reshaped = sliced.reshape(Shape{16, 4, 4}).value();
    benchmark::DoNotOptimize(reshaped.data());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_TensorSliceAndReshape_ZeroCopy);

// ===========================================================================
// 4. End-to-End C++ InferenceSession Execution (Heap vs Arena vs In-Place)
// ===========================================================================

void BM_InferenceSession_Run_HeapOutputs(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  auto session = InferenceSession::create(buildBenchmarkMlpModel(16, 8)).value();
  auto x = Tensor::create(Shape{batch_size, 16}, DataType::Float32).value();
  x.fill<float>(0.5f);

  for (auto _ : state) {
    auto outs = session.run({x}).value();
    benchmark::DoNotOptimize(outs[0].data());
  }
  state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_InferenceSession_Run_HeapOutputs)->Arg(1)->Arg(8)->Arg(32)->Arg(64);

void BM_InferenceSession_Run_ArenaOutputs(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  auto session = InferenceSession::create(buildBenchmarkMlpModel(16, 8)).value();
  auto x = Tensor::create(Shape{batch_size, 16}, DataType::Float32).value();
  x.fill<float>(0.5f);
  ArenaAllocator arena(64 * 1024);

  Tensor inputs[] = {x};

  for (auto _ : state) {
    {
      auto outs = session.run(std::span<const Tensor>(inputs, 1), arena).value();
      benchmark::DoNotOptimize(outs[0].data());
    }
    arena.reset();
  }
  state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_InferenceSession_Run_ArenaOutputs)->Arg(1)->Arg(8)->Arg(32)->Arg(64);

void BM_InferenceSession_Run_InPlaceOutputs(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  auto session = InferenceSession::create(buildBenchmarkMlpModel(16, 8)).value();
  auto x = Tensor::create(Shape{batch_size, 16}, DataType::Float32).value();
  auto y = Tensor::create(Shape{batch_size, 8}, DataType::Float32).value();
  x.fill<float>(0.5f);

  Tensor inputs[] = {x};
  Tensor outputs[] = {y};

  for (auto _ : state) {
    auto st = session.run(std::span<const Tensor>(inputs, 1), std::span<Tensor>(outputs, 1));
    benchmark::DoNotOptimize(st);
    benchmark::DoNotOptimize(y.data());
  }
  state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_InferenceSession_Run_InPlaceOutputs)->Arg(1)->Arg(8)->Arg(32)->Arg(64);

}  // namespace
}  // namespace pyreth::benchmarks
