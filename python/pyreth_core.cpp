#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <memory>
#include <pyreth/backend.hpp>
#include <pyreth/common.hpp>
#include <pyreth/dynamic_batcher.hpp>
#include <pyreth/inference_session.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/model_info.hpp>
#include <pyreth/result.hpp>
#include <pyreth/runtime.hpp>
#include <pyreth/tensor.hpp>
#include <pyreth/thread_pool.hpp>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

[[noreturn]] void throwForError(const pyreth::Error& err) {
  const std::string msg(err.message());
  switch (err.code()) {
    case pyreth::StatusCode::MODEL_NOT_FOUND:
      PyErr_SetString(PyExc_FileNotFoundError, msg.c_str());
      throw nb::python_error();
    case pyreth::StatusCode::INVALID_ARGUMENT:
    case pyreth::StatusCode::INVALID_DIMENSIONS:
      throw nb::value_error(msg.c_str());
    case pyreth::StatusCode::ALLOCATION_FAILED:
      PyErr_SetString(PyExc_MemoryError, msg.c_str());
      throw nb::python_error();
    case pyreth::StatusCode::TIMEOUT:
      PyErr_SetString(PyExc_TimeoutError, msg.c_str());
      throw nb::python_error();
    default:
      PyErr_SetString(PyExc_RuntimeError, msg.c_str());
      throw nb::python_error();
  }
}

[[nodiscard]] nb::dlpack::dtype toDlpackDtype(pyreth::DataType dtype) {
  using pyreth::DataType;
  switch (dtype) {
    case DataType::Float32:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
    case DataType::Float64:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 64, 1};
    case DataType::Float16:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    case DataType::Int8:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 8, 1};
    case DataType::Int16:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 16, 1};
    case DataType::Int32:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 32, 1};
    case DataType::Int64:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 64, 1};
    case DataType::UInt8:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1};
    case DataType::UInt16:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 16, 1};
    case DataType::UInt32:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 32, 1};
    case DataType::UInt64:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 64, 1};
    case DataType::Bool:
      return {static_cast<uint8_t>(nb::dlpack::dtype_code::Bool), 8, 1};
    case DataType::BFloat16:
      throw nb::type_error("BFloat16 is not natively supported by standard NumPy dtypes");
    default:
      throw nb::type_error("Unsupported Tensor DataType for NumPy conversion");
  }
}

[[nodiscard]] pyreth::DataType fromDlpackDtype(const nb::dlpack::dtype& dt) {
  using pyreth::DataType;
  if (dt.lanes != 1) {
    throw nb::type_error("Vectorized multi-lane DLPack dtypes are not supported");
  }

  const auto code = static_cast<nb::dlpack::dtype_code>(dt.code);
  switch (code) {
    case nb::dlpack::dtype_code::Float:
      if (dt.bits == 32) {
        return DataType::Float32;
      }
      if (dt.bits == 64) {
        return DataType::Float64;
      }
      if (dt.bits == 16) {
        return DataType::Float16;
      }
      break;
    case nb::dlpack::dtype_code::Bfloat:
      if (dt.bits == 16) {
        return DataType::BFloat16;
      }
      break;
    case nb::dlpack::dtype_code::Int:
      if (dt.bits == 8) {
        return DataType::Int8;
      }
      if (dt.bits == 16) {
        return DataType::Int16;
      }
      if (dt.bits == 32) {
        return DataType::Int32;
      }
      if (dt.bits == 64) {
        return DataType::Int64;
      }
      break;
    case nb::dlpack::dtype_code::UInt:
      if (dt.bits == 8) {
        return DataType::UInt8;
      }
      if (dt.bits == 16) {
        return DataType::UInt16;
      }
      if (dt.bits == 32) {
        return DataType::UInt32;
      }
      if (dt.bits == 64) {
        return DataType::UInt64;
      }
      break;
    case nb::dlpack::dtype_code::Bool:
      if (dt.bits == 8) {
        return DataType::Bool;
      }
      break;
    default:
      break;
  }

  throw nb::type_error("Unsupported NumPy / DLPack dtype for pyreth.Tensor");
}

[[nodiscard]] pyreth::DataType parseDataTypeObject(nb::handle obj) {
  if (nb::isinstance<pyreth::DataType>(obj)) {
    return nb::cast<pyreth::DataType>(obj);
  }

  std::string name;
  if (nb::isinstance<nb::str>(obj)) {
    name = nb::cast<std::string>(obj);
  } else if (nb::hasattr(obj, "name")) {
    name = nb::cast<std::string>(obj.attr("name"));
  } else if (nb::hasattr(obj, "__name__")) {
    name = nb::cast<std::string>(obj.attr("__name__"));
  } else {
    throw nb::type_error("Expected pyreth.DataType, numpy.dtype, or dtype string");
  }

  std::string lower;
  lower.reserve(name.size());
  for (char c : name) {
    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  }

  if (lower == "float32" || lower == "float" || lower == "f4") {
    return pyreth::DataType::Float32;
  }
  if (lower == "float64" || lower == "double" || lower == "f8") {
    return pyreth::DataType::Float64;
  }
  if (lower == "float16" || lower == "half" || lower == "f2") {
    return pyreth::DataType::Float16;
  }
  if (lower == "bfloat16" || lower == "bf16") {
    return pyreth::DataType::BFloat16;
  }
  if (lower == "int8" || lower == "i1") {
    return pyreth::DataType::Int8;
  }
  if (lower == "uint8" || lower == "u1") {
    return pyreth::DataType::UInt8;
  }
  if (lower == "int16" || lower == "short" || lower == "i2") {
    return pyreth::DataType::Int16;
  }
  if (lower == "uint16" || lower == "ushort" || lower == "u2") {
    return pyreth::DataType::UInt16;
  }
  if (lower == "int32" || lower == "int" || lower == "i4") {
    return pyreth::DataType::Int32;
  }
  if (lower == "uint32" || lower == "uint" || lower == "u4") {
    return pyreth::DataType::UInt32;
  }
  if (lower == "int64" || lower == "long" || lower == "i8") {
    return pyreth::DataType::Int64;
  }
  if (lower == "uint64" || lower == "ulong" || lower == "u8") {
    return pyreth::DataType::UInt64;
  }
  if (lower == "bool" || lower == "bool_" || lower == "?") {
    return pyreth::DataType::Bool;
  }

  throw nb::type_error(("Unsupported dtype: " + name).c_str());
}

[[nodiscard]] pyreth::Shape parseShape(nb::handle shape_obj) {
  if (nb::isinstance<pyreth::Shape>(shape_obj)) {
    return nb::cast<pyreth::Shape>(shape_obj);
  }
  if (nb::isinstance<nb::int_>(shape_obj)) {
    return pyreth::Shape({nb::cast<int64_t>(shape_obj)});
  }
  std::vector<int64_t> dims = nb::cast<std::vector<int64_t>>(shape_obj);
  for (int64_t d : dims) {
    if (d < 0) {
      throw nb::value_error("Tensor dimensions must be non-negative");
    }
  }
  return pyreth::Shape(std::move(dims));
}

[[nodiscard]] pyreth::Tensor tensorFromNumpyObject(nb::object obj) {
  if (nb::hasattr(obj, "flags")) {
    nb::object flags = obj.attr("flags");
    if (nb::hasattr(flags, "writeable") && !nb::cast<bool>(flags.attr("writeable"))) {
      throw nb::value_error("Cannot create mutable zero-copy Tensor from a read-only NumPy array");
    }
  }

  if (PyObject_CheckBuffer(obj.ptr()) == 1) {
    Py_buffer view{};
    if (PyObject_GetBuffer(obj.ptr(), &view, PyBUF_SIMPLE) == 0) {
      const bool readonly = (view.readonly != 0);
      PyBuffer_Release(&view);
      if (readonly) {
        throw nb::value_error("Cannot create mutable zero-copy Tensor from a read-only buffer");
      }
    } else {
      PyErr_Clear();
    }
  }

  nb::ndarray<nb::ro> arr;
  try {
    arr = nb::cast<nb::ndarray<nb::ro>>(obj);
  } catch (const nb::cast_error&) {
    throw nb::type_error(
        "Expected a NumPy ndarray (or DLPack-compatible buffer) with a supported numeric dtype");
  }

  if (arr.device_type() != nb::device::cpu::value) {
    throw nb::value_error("Zero-copy NumPy bridge requires a CPU-resident array");
  }

  const pyreth::DataType dtype = fromDlpackDtype(arr.dtype());

  std::vector<int64_t> dims(arr.ndim());
  std::vector<int64_t> strides(arr.ndim());
  for (size_t i = 0; i < arr.ndim(); ++i) {
    dims[i] = static_cast<int64_t>(arr.shape(i));
    strides[i] = arr.stride(i);
    if (strides[i] < 0) {
      throw nb::value_error(
          "Negative strides are not supported for zero-copy Tensor mapping; call "
          "np.ascontiguousarray() first");
    }
  }

  pyreth::Shape shape(std::move(dims));

  PyObject* raw_py_obj = obj.ptr();
  Py_INCREF(raw_py_obj);
  std::shared_ptr<void> owner(raw_py_obj, [](void* ptr) noexcept {
    if (Py_IsInitialized() != 0) {
      nb::gil_scoped_acquire gil;
      Py_DECREF(static_cast<PyObject*>(ptr));
    }
  });

  void* data_ptr = const_cast<void*>(arr.data());
  alignas(64) static std::byte empty_sentinel[64]{};
  if (data_ptr == nullptr && shape.numElements() == 0) {
    data_ptr = empty_sentinel;
  }

  auto res =
      pyreth::Tensor::fromBuffer(data_ptr, shape, dtype, std::move(strides), std::move(owner));
  if (!res.isOk()) {
    throw nb::value_error(std::string(res.error().message()).c_str());
  }
  return std::move(res).value();
}

[[nodiscard]] nb::object tensorToNumpyObject(pyreth::Tensor& self) {
  if (!self.isValid()) {
    throw nb::value_error("Cannot convert an invalid or uninitialized Tensor to a NumPy array");
  }

  const nb::dlpack::dtype dt = toDlpackDtype(self.dtype());

  std::vector<size_t> shape(self.rank());
  for (size_t i = 0; i < self.rank(); ++i) {
    shape[i] = static_cast<size_t>(self.shape()[i]);
  }

  auto* storage_ref = new std::shared_ptr<pyreth::TensorStorage>(self.storage());
  nb::capsule owner(storage_ref, [](void* p) noexcept {
    delete static_cast<std::shared_ptr<pyreth::TensorStorage>*>(p);
  });

  const size_t* shape_ptr = shape.empty() ? nullptr : shape.data();
  const int64_t* strides_ptr = self.strides().empty() ? nullptr : self.strides().data();

  nb::ndarray<nb::numpy> arr(self.data(), self.rank(), shape_ptr, owner, strides_ptr, dt);
  return arr.cast();
}

[[nodiscard]] nb::tuple vecToTuple(const std::vector<int64_t>& vec) {
  nb::list lst;
  for (int64_t v : vec) {
    lst.append(v);
  }
  return nb::tuple(lst);
}

[[nodiscard]] nb::tuple strVecToTuple(const std::vector<std::string>& vec) {
  nb::list lst;
  for (const auto& s : vec) {
    lst.append(s);
  }
  return nb::tuple(lst);
}

[[nodiscard]] pyreth::Tensor coerceToTensor(nb::handle item) {
  if (nb::isinstance<pyreth::Tensor>(item)) {
    return nb::cast<pyreth::Tensor>(item);
  }
  return tensorFromNumpyObject(nb::borrow<nb::object>(item));
}

struct ParsedInputs {
  bool is_named{false};
  std::vector<pyreth::Tensor> positional;
  std::unordered_map<std::string, pyreth::Tensor> named;
};

[[nodiscard]] ParsedInputs parseInputsArgument(nb::handle inputs_arg) {
  ParsedInputs parsed;
  if (nb::isinstance<nb::dict>(inputs_arg)) {
    parsed.is_named = true;
    auto d = nb::borrow<nb::dict>(inputs_arg);
    for (auto [k, v] : d) {
      parsed.named.emplace(nb::cast<std::string>(k), coerceToTensor(v));
    }
    return parsed;
  }

  if (nb::isinstance<pyreth::Tensor>(inputs_arg) ||
      nb::hasattr(inputs_arg, "__array_interface__") || nb::hasattr(inputs_arg, "__dlpack__")) {
    parsed.positional.push_back(coerceToTensor(inputs_arg));
    return parsed;
  }

  if (PySequence_Check(inputs_arg.ptr()) == 1) {
    const Py_ssize_t len = PySequence_Size(inputs_arg.ptr());
    if (len < 0) {
      throw nb::python_error();
    }
    parsed.positional.reserve(static_cast<size_t>(len));
    for (Py_ssize_t i = 0; i < len; ++i) {
      nb::object elem = nb::steal<nb::object>(PySequence_GetItem(inputs_arg.ptr(), i));
      parsed.positional.push_back(coerceToTensor(elem));
    }
    return parsed;
  }

  throw nb::type_error(
      "Expected a Tensor, numpy.ndarray, sequence of Tensors/ndarrays, or dict[str, "
      "Tensor/ndarray]");
}

[[nodiscard]] pyreth::InferenceSession createSessionFromPyObject(
    nb::handle model_path_or_bytes, const pyreth::SessionConfig& config) {
  // 1. Check for bytes / bytearray / memoryview buffer first
  if (nb::isinstance<nb::bytes>(model_path_or_bytes) ||
      nb::isinstance<nb::bytearray>(model_path_or_bytes) ||
      PyMemoryView_Check(model_path_or_bytes.ptr())) {
    Py_buffer view{};
    if (PyObject_GetBuffer(model_path_or_bytes.ptr(), &view, PyBUF_SIMPLE) != 0) {
      throw nb::python_error();
    }
    std::vector<uint8_t> model_bytes(static_cast<const uint8_t*>(view.buf),
                                     static_cast<const uint8_t*>(view.buf) + view.len);
    PyBuffer_Release(&view);

    pyreth::Result<pyreth::InferenceSession> res = [&]() {
      nb::gil_scoped_release release;
      return pyreth::InferenceSession::create(std::span<const uint8_t>(model_bytes), config);
    }();
    if (!res.isOk()) {
      throwForError(res.error());
    }
    return std::move(res).value();
  }

  // 2. Otherwise support str or os.PathLike (__fspath__)
  std::string path_str;
  if (nb::isinstance<nb::str>(model_path_or_bytes)) {
    path_str = nb::cast<std::string>(model_path_or_bytes);
  } else if (nb::hasattr(model_path_or_bytes, "__fspath__")) {
    nb::object fspath = model_path_or_bytes.attr("__fspath__")();
    path_str = nb::cast<std::string>(fspath);
  } else {
    throw nb::type_error("Expected model path (str or os.PathLike) or model bytes");
  }

  pyreth::Result<pyreth::InferenceSession> res = [&]() {
    nb::gil_scoped_release release;
    return pyreth::InferenceSession::create(path_str, config);
  }();
  if (!res.isOk()) {
    throwForError(res.error());
  }
  return std::move(res).value();
}

}  // namespace

NB_MODULE(_core, m) {
  m.doc() = "Pyreth AI inference engine core Python bindings";

  m.attr("__version__") = pyreth::PYRETH_VERSION;

  m.def(
      "get_runtime_version", []() { return pyreth::Backend::getRuntimeVersion(); },
      "Returns the version of the underlying ONNX Runtime engine.");

  m.def(
      "get_best_backend",
      []() {
        return std::string(
            pyreth::executionProviderToString(pyreth::Backend::getBestAvailableProvider()));
      },
      "Returns the name of the best available execution provider.");

  m.def(
      "has_hardware_acceleration", []() { return pyreth::Backend::hasHardwareAcceleration(); },
      "Returns true if hardware acceleration is available.");

  // ---------------------------------------------------------------------------
  // Core Enums
  // ---------------------------------------------------------------------------
  nb::enum_<pyreth::DataType>(m, "DataType", "Supported element data types for pyreth.Tensor.")
      .value("Float32", pyreth::DataType::Float32)
      .value("Float16", pyreth::DataType::Float16)
      .value("BFloat16", pyreth::DataType::BFloat16)
      .value("Float64", pyreth::DataType::Float64)
      .value("Int8", pyreth::DataType::Int8)
      .value("UInt8", pyreth::DataType::UInt8)
      .value("Int16", pyreth::DataType::Int16)
      .value("UInt16", pyreth::DataType::UInt16)
      .value("Int32", pyreth::DataType::Int32)
      .value("UInt32", pyreth::DataType::UInt32)
      .value("Int64", pyreth::DataType::Int64)
      .value("UInt64", pyreth::DataType::UInt64)
      .value("Bool", pyreth::DataType::Bool)
      .export_values();

  nb::enum_<pyreth::StorageType>(m, "StorageType", "Backing memory allocation strategy for Tensor.")
      .value("OWNED", pyreth::StorageType::OWNED)
      .value("ARENA", pyreth::StorageType::ARENA)
      .value("POOLED", pyreth::StorageType::POOLED)
      .value("EXTERNAL", pyreth::StorageType::EXTERNAL)
      .export_values();

  nb::enum_<pyreth::ExecutionProvider>(m, "ExecutionProvider",
                                       "Hardware execution providers supported by Pyreth.")
      .value("CPU", pyreth::ExecutionProvider::CPU)
      .value("COREML", pyreth::ExecutionProvider::COREML)
      .value("CUDA", pyreth::ExecutionProvider::CUDA)
      .value("TENSORRT", pyreth::ExecutionProvider::TENSORRT)
      .value("DIRECTML", pyreth::ExecutionProvider::DIRECTML)
      .value("ROCM", pyreth::ExecutionProvider::ROCM)
      .value("UNKNOWN", pyreth::ExecutionProvider::UNKNOWN);

  nb::enum_<pyreth::OptimizationLevel>(m, "OptimizationLevel",
                                       "ONNX Runtime computation graph optimization levels.")
      .value("Disable", pyreth::OptimizationLevel::Disable)
      .value("Basic", pyreth::OptimizationLevel::Basic)
      .value("Extended", pyreth::OptimizationLevel::Extended)
      .value("All", pyreth::OptimizationLevel::All);

  nb::enum_<pyreth::SessionExecutionMode>(m, "SessionExecutionMode",
                                          "Execution mode for graph operator evaluation.")
      .value("Sequential", pyreth::SessionExecutionMode::Sequential)
      .value("Parallel", pyreth::SessionExecutionMode::Parallel);

  nb::enum_<pyreth::LogLevel>(m, "LogLevel", "Diagnostic logging severity levels.")
      .value("Verbose", pyreth::LogLevel::Verbose)
      .value("Info", pyreth::LogLevel::Info)
      .value("Warning", pyreth::LogLevel::Warning)
      .value("Error", pyreth::LogLevel::Error)
      .value("Fatal", pyreth::LogLevel::Fatal);

  nb::enum_<pyreth::StatusCode>(m, "StatusCode", "Discrete engine status and error codes.")
      .value("OK", pyreth::StatusCode::OK)
      .value("INVALID_ARGUMENT", pyreth::StatusCode::INVALID_ARGUMENT)
      .value("INVALID_DIMENSIONS", pyreth::StatusCode::INVALID_DIMENSIONS)
      .value("MODEL_NOT_FOUND", pyreth::StatusCode::MODEL_NOT_FOUND)
      .value("MODEL_LOAD_FAILED", pyreth::StatusCode::MODEL_LOAD_FAILED)
      .value("BACKEND_ERROR", pyreth::StatusCode::BACKEND_ERROR)
      .value("BACKEND_UNAVAILABLE", pyreth::StatusCode::BACKEND_UNAVAILABLE)
      .value("QUEUE_FULL", pyreth::StatusCode::QUEUE_FULL)
      .value("QUEUE_EMPTY", pyreth::StatusCode::QUEUE_EMPTY)
      .value("ALLOCATION_FAILED", pyreth::StatusCode::ALLOCATION_FAILED)
      .value("TIMEOUT", pyreth::StatusCode::TIMEOUT)
      .value("INTERNAL_ERROR", pyreth::StatusCode::INTERNAL_ERROR)
      .value("UNKNOWN", pyreth::StatusCode::UNKNOWN);

  // ---------------------------------------------------------------------------
  // Backend & Runtime
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::Backend>(m, "Backend", "Hardware execution provider discovery utility.")
      .def_static("get_runtime_version", &pyreth::Backend::getRuntimeVersion,
                  "Query the ONNX Runtime engine version string.")
      .def_static("get_available_providers", &pyreth::Backend::getAvailableProviders,
                  "Return all registered execution provider names.")
      .def_static("get_available_execution_providers",
                  &pyreth::Backend::getAvailableExecutionProviders,
                  "Return all available ExecutionProvider enum values.")
      .def_static(
          "is_provider_available",
          [](pyreth::ExecutionProvider provider) {
            return pyreth::Backend::isProviderAvailable(provider);
          },
          nb::arg("provider"), "Return True if the specified ExecutionProvider is available.")
      .def_static(
          "is_provider_available",
          [](std::string_view provider_name) {
            return pyreth::Backend::isProviderAvailable(provider_name);
          },
          nb::arg("provider_name"), "Return True if the named execution provider is available.")
      .def_static("get_best_available_provider", &pyreth::Backend::getBestAvailableProvider,
                  "Return the highest-priority available ExecutionProvider.")
      .def_static("has_hardware_acceleration", &pyreth::Backend::hasHardwareAcceleration,
                  "Return True if any non-CPU hardware accelerator is available.");

  nb::class_<pyreth::Runtime>(m, "Runtime", "Thread-safe singleton managing the shared Ort::Env.")
      .def_static(
          "instance", []() -> pyreth::Runtime& { return pyreth::Runtime::instance(); },
          nb::rv_policy::reference, "Retrieve the process-wide Runtime singleton.")
      .def(
          "init",
          [](pyreth::Runtime& self, pyreth::LogLevel log_level, std::string_view log_id) {
            auto st = self.init(log_level, log_id);
            if (!st.isOk()) {
              throwForError(st.error());
            }
          },
          nb::arg("log_level") = pyreth::LogLevel::Warning, nb::arg("log_id") = "Pyreth",
          "Initialize the global ONNX Runtime environment.")
      .def_prop_ro("is_initialized", &pyreth::Runtime::isInitialized)
      .def_prop_ro("log_level", &pyreth::Runtime::getLogLevel)
      .def_prop_ro("log_id", &pyreth::Runtime::getLogId)
      .def("shutdown", &pyreth::Runtime::shutdown,
           "Release the global Ort::Env (may be re-initialized lazily).");

  // ---------------------------------------------------------------------------
  // Shape
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::Shape>(m, "Shape", "Represents the dimensions of an N-dimensional Tensor.")
      .def(nb::init<>())
      .def(nb::init<std::vector<int64_t>>(), nb::arg("dims"))
      .def_prop_ro("rank", &pyreth::Shape::rank)
      .def_prop_ro("ndim", &pyreth::Shape::ndim)
      .def_prop_ro("dims", [](const pyreth::Shape& s) { return vecToTuple(s.dims()); })
      .def("num_elements", &pyreth::Shape::numElements)
      .def("is_scalar", &pyreth::Shape::isScalar)
      .def("__len__", &pyreth::Shape::rank)
      .def("__getitem__",
           [](const pyreth::Shape& s, int64_t idx) {
             const auto r = static_cast<int64_t>(s.rank());
             if (idx < 0) {
               idx += r;
             }
             if (idx < 0 || idx >= r) {
               throw nb::index_error("Shape index out of range");
             }
             return s[static_cast<size_t>(idx)];
           })
      .def("__eq__", &pyreth::Shape::operator==)
      .def("__ne__", &pyreth::Shape::operator!=)
      .def("__repr__", [](const pyreth::Shape& s) { return "Shape(" + s.toString() + ")"; });

  // ---------------------------------------------------------------------------
  // ArenaAllocator & MemoryPool
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::ArenaAllocator>(
      m, "ArenaAllocator", "64-byte SIMD-aligned bump allocator with O(1) reset and pin tracking.")
      .def(nb::init<size_t>(), nb::arg("block_size") = pyreth::DEFAULT_BLOCK_SIZE)
      .def("reset", &pyreth::ArenaAllocator::reset,
           "Rewinds the arena in O(1) if not pinned by active Tensor references.")
      .def("force_reset", &pyreth::ArenaAllocator::forceReset,
           "Unconditionally resets the arena regardless of active pins.")
      .def("can_reset", &pyreth::ArenaAllocator::canReset)
      .def_prop_ro("is_pinned", &pyreth::ArenaAllocator::isPinned)
      .def_prop_ro("pin_count", &pyreth::ArenaAllocator::pinCount)
      .def_prop_ro("bytes_allocated", &pyreth::ArenaAllocator::bytesAllocated)
      .def_prop_ro("bytes_reserved", &pyreth::ArenaAllocator::bytesReserved)
      .def_prop_ro("block_count", &pyreth::ArenaAllocator::blockCount);

  nb::class_<pyreth::MemoryPool>(m, "MemoryPool",
                                 "Thread-safe pool of reusable 64-byte aligned ArenaAllocators.")
      .def(nb::init<size_t, size_t>(), nb::arg("initial_count") = 4,
           nb::arg("block_size") = pyreth::DEFAULT_BLOCK_SIZE)
      .def_prop_ro("available_count", &pyreth::MemoryPool::availableCount);

  // ---------------------------------------------------------------------------
  // Tensor
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::Tensor>(
      m, "Tensor",
      "Multi-dimensional array with 64-byte SIMD alignment and zero-copy NumPy interoperability.")
      .def(
          "__init__",
          [](pyreth::Tensor* t, nb::object array_or_shape, nb::object dtype_obj) {
            if (dtype_obj.is_none() && (nb::hasattr(array_or_shape, "__array_interface__") ||
                                        nb::hasattr(array_or_shape, "__dlpack__"))) {
              new (t) pyreth::Tensor(tensorFromNumpyObject(array_or_shape));
              return;
            }
            const pyreth::Shape shape = parseShape(array_or_shape);
            const pyreth::DataType dtype =
                dtype_obj.is_none() ? pyreth::DataType::Float32 : parseDataTypeObject(dtype_obj);
            auto res = pyreth::Tensor::create(shape, dtype);
            if (!res.isOk()) {
              throwForError(res.error());
            }
            new (t) pyreth::Tensor(std::move(res).value());
          },
          nb::arg("array_or_shape"), nb::arg("dtype") = nb::none(),
          "Construct a Tensor either zero-copy from a NumPy array or freshly allocated from a "
          "shape and dtype.")
      .def_static(
          "from_numpy", [](nb::object arr) { return tensorFromNumpyObject(std::move(arr)); },
          nb::arg("array"), "Create a zero-copy pyreth.Tensor sharing memory with a NumPy ndarray.")
      .def_static(
          "create",
          [](nb::handle shape_obj, nb::object dtype_obj, pyreth::ArenaAllocator* arena,
             pyreth::MemoryPool* pool) {
            const pyreth::Shape shape = parseShape(shape_obj);
            const pyreth::DataType dtype =
                dtype_obj.is_none() ? pyreth::DataType::Float32 : parseDataTypeObject(dtype_obj);
            pyreth::Result<pyreth::Tensor> res =
                arena != nullptr ? pyreth::Tensor::create(shape, dtype, *arena)
                                 : (pool != nullptr ? pyreth::Tensor::create(shape, dtype, *pool)
                                                    : pyreth::Tensor::create(shape, dtype));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("shape"), nb::arg("dtype") = pyreth::DataType::Float32,
          nb::arg("arena") = nb::none(), nb::arg("pool") = nb::none(),
          "Allocate a 64-byte aligned Tensor on the heap, in an ArenaAllocator, or from a "
          "MemoryPool.")
      .def_static(
          "zeros",
          [](nb::handle shape_obj, nb::object dtype_obj, pyreth::ArenaAllocator* arena,
             pyreth::MemoryPool* pool) {
            const pyreth::Shape shape = parseShape(shape_obj);
            const pyreth::DataType dtype =
                dtype_obj.is_none() ? pyreth::DataType::Float32 : parseDataTypeObject(dtype_obj);
            pyreth::Result<pyreth::Tensor> res =
                arena != nullptr ? pyreth::Tensor::create(shape, dtype, *arena)
                                 : (pool != nullptr ? pyreth::Tensor::create(shape, dtype, *pool)
                                                    : pyreth::Tensor::create(shape, dtype));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            pyreth::Tensor tensor = std::move(res).value();
            tensor.zero();
            return tensor;
          },
          nb::arg("shape"), nb::arg("dtype") = pyreth::DataType::Float32,
          nb::arg("arena") = nb::none(), nb::arg("pool") = nb::none(),
          "Allocate a zero-initialized 64-byte aligned Tensor.")
      .def(
          "numpy", [](pyreth::Tensor& self) { return tensorToNumpyObject(self); },
          "Return a zero-copy NumPy ndarray view sharing the Tensor's underlying memory.")
      .def(
          "to_numpy", [](pyreth::Tensor& self) { return tensorToNumpyObject(self); },
          "Return a zero-copy NumPy ndarray view sharing the Tensor's underlying memory.")
      .def(
          "__array__",
          [](pyreth::Tensor& self, nb::object dtype, nb::object copy) {
            nb::object arr = tensorToNumpyObject(self);
            if (!dtype.is_none()) {
              nb::object current_dtype = arr.attr("dtype");
              nb::module_ np = nb::module_::import_("numpy");
              nb::object target_dtype = np.attr("dtype")(dtype);
              if (!current_dtype.equal(target_dtype)) {
                if (!copy.is_none() && !nb::cast<bool>(copy)) {
                  throw nb::value_error(
                      "Cannot convert dtype without copying when copy=False in __array__");
                }
                return arr.attr("astype")(target_dtype);
              }
            }
            if (!copy.is_none() && nb::cast<bool>(copy)) {
              return arr.attr("copy")();
            }
            return arr;
          },
          nb::arg("dtype") = nb::none(), nb::arg("copy") = nb::none(),
          "NumPy __array__ protocol for zero-copy np.asarray(tensor) conversion.")
      .def_prop_ro("shape",
                   [](const pyreth::Tensor& self) { return vecToTuple(self.shape().dims()); })
      .def_prop_ro("strides", [](const pyreth::Tensor& self) { return vecToTuple(self.strides()); })
      .def_prop_ro("byte_strides",
                   [](const pyreth::Tensor& self) {
                     nb::list lst;
                     const auto elem_sz = static_cast<int64_t>(self.elementSize());
                     for (int64_t s : self.strides()) {
                       lst.append(s * elem_sz);
                     }
                     return nb::tuple(lst);
                   })
      .def_prop_ro("dtype", &pyreth::Tensor::dtype)
      .def_prop_ro("ndim", &pyreth::Tensor::ndim)
      .def_prop_ro("rank", &pyreth::Tensor::rank)
      .def_prop_ro("size", &pyreth::Tensor::numElements)
      .def_prop_ro("num_elements", &pyreth::Tensor::numElements)
      .def_prop_ro("nbytes", &pyreth::Tensor::totalBytes)
      .def_prop_ro("total_bytes", &pyreth::Tensor::totalBytes)
      .def_prop_ro("itemsize", &pyreth::Tensor::elementSize)
      .def_prop_ro("element_size", &pyreth::Tensor::elementSize)
      .def_prop_ro("byte_offset", &pyreth::Tensor::byteOffset)
      .def_prop_ro(
          "data_ptr",
          [](const pyreth::Tensor& self) { return reinterpret_cast<uintptr_t>(self.data()); })
      .def_prop_ro("is_contiguous", &pyreth::Tensor::isContiguous)
      .def_prop_ro("is_valid", &pyreth::Tensor::isValid)
      .def_prop_ro("is_arena_backed", &pyreth::Tensor::isArenaBacked)
      .def_prop_ro("storage_type", &pyreth::Tensor::storageType)
      .def_prop_ro("use_count", &pyreth::Tensor::useCount)
      .def("is_aligned", &pyreth::Tensor::isAligned,
           nb::arg("alignment") = pyreth::DEFAULT_ALIGNMENT)
      .def(
          "reshape",
          [](const pyreth::Tensor& self, nb::handle new_shape) {
            std::vector<int64_t> dims;
            if (nb::isinstance<pyreth::Shape>(new_shape)) {
              dims = nb::cast<pyreth::Shape>(new_shape).dims();
            } else if (nb::isinstance<nb::int_>(new_shape)) {
              dims = {nb::cast<int64_t>(new_shape)};
            } else {
              dims = nb::cast<std::vector<int64_t>>(new_shape);
            }
            auto res = self.reshape(pyreth::Shape(std::move(dims)));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("new_shape"), "Return a zero-copy reshaped view of the Tensor.")
      .def(
          "slice",
          [](const pyreth::Tensor& self, size_t dim, int64_t start, int64_t end, int64_t step) {
            auto res = self.slice(dim, start, end, step);
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("dim"), nb::arg("start"), nb::arg("end"), nb::arg("step") = 1,
          "Return a zero-copy strided slice view along the specified dimension.")
      .def(
          "to_contiguous",
          [](const pyreth::Tensor& self, pyreth::ArenaAllocator* arena) {
            auto res = arena != nullptr ? self.toContiguous(*arena) : self.toContiguous();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("arena") = nb::none(),
          "Return a contiguous Tensor (no-op shallow view if already contiguous, otherwise "
          "copies).")
      .def(
          "clone",
          [](const pyreth::Tensor& self, pyreth::ArenaAllocator* arena) {
            return arena != nullptr ? self.clone(*arena) : self.clone();
          },
          nb::arg("arena") = nb::none(), "Return a deep copy of the Tensor.")
      .def("zero", &pyreth::Tensor::zero, "Fill the Tensor with zeros in-place.")
      .def(
          "fill",
          [](pyreth::Tensor& self, double value) {
            using pyreth::DataType;
            switch (self.dtype()) {
              case DataType::Float32:
                self.fill<float>(static_cast<float>(value));
                break;
              case DataType::Float64:
                self.fill<double>(value);
                break;
              case DataType::Int8:
                self.fill<int8_t>(static_cast<int8_t>(value));
                break;
              case DataType::UInt8:
                self.fill<uint8_t>(static_cast<uint8_t>(value));
                break;
              case DataType::Int16:
                self.fill<int16_t>(static_cast<int16_t>(value));
                break;
              case DataType::UInt16:
                self.fill<uint16_t>(static_cast<uint16_t>(value));
                break;
              case DataType::Int32:
                self.fill<int32_t>(static_cast<int32_t>(value));
                break;
              case DataType::UInt32:
                self.fill<uint32_t>(static_cast<uint32_t>(value));
                break;
              case DataType::Int64:
                self.fill<int64_t>(static_cast<int64_t>(value));
                break;
              case DataType::UInt64:
                self.fill<uint64_t>(static_cast<uint64_t>(value));
                break;
              case DataType::Bool:
                self.fill<bool>(value != 0.0);
                break;
              default:
                throw nb::type_error("fill() is not supported for this DataType directly");
            }
          },
          nb::arg("value"), "Fill the Tensor with a scalar value in-place.")
      .def("__len__",
           [](const pyreth::Tensor& self) {
             if (self.rank() == 0) {
               throw nb::type_error("len() of a 0-d scalar Tensor");
             }
             return self.shape()[0];
           })
      .def("__repr__", &pyreth::Tensor::toString)
      .def("__str__", &pyreth::Tensor::toString);

  // ---------------------------------------------------------------------------
  // TensorInfo & ModelInfo (Model Metadata & IO Discovery)
  // ---------------------------------------------------------------------------
  auto tensor_info_cls =
      nb::class_<pyreth::TensorInfo>(
          m, "TensorInfo", "Cached metadata describing a single model input or output tensor node.")
          .def(nb::init<>())
          .def_rw("name", &pyreth::TensorInfo::name)
          .def_rw("dtype", &pyreth::TensorInfo::dtype)
          .def_prop_ro("shape",
                       [](const pyreth::TensorInfo& info) { return vecToTuple(info.shape); })
          .def_prop_ro(
              "symbolic_dims",
              [](const pyreth::TensorInfo& info) { return strVecToTuple(info.symbolic_dims); })
          .def_prop_ro("rank", &pyreth::TensorInfo::rank)
          .def_prop_ro("is_scalar", &pyreth::TensorInfo::isScalar)
          .def_prop_ro("has_dynamic_shape", &pyreth::TensorInfo::hasDynamicShape)
          .def_prop_ro("has_dynamic_batch", &pyreth::TensorInfo::hasDynamicBatch)
          .def("is_dynamic_dim", &pyreth::TensorInfo::isDynamicDim, nb::arg("axis"))
          .def("fixed_element_count", &pyreth::TensorInfo::fixedElementCount)
          .def(
              "validate_shape",
              [](const pyreth::TensorInfo& info, nb::handle shape_obj) {
                auto st = info.validateShape(parseShape(shape_obj));
                if (!st.isOk()) {
                  throwForError(st.error());
                }
              },
              nb::arg("shape"),
              "Validate a runtime shape against this TensorInfo schema (raises ValueError on "
              "mismatch).")
          .def(
              "validate_tensor",
              [](const pyreth::TensorInfo& info, nb::handle tensor_obj) {
                auto t = coerceToTensor(tensor_obj);
                auto st = info.validateTensor(t);
                if (!st.isOk()) {
                  throwForError(st.error());
                }
              },
              nb::arg("tensor"),
              "Validate a runtime Tensor or ndarray against this TensorInfo schema.")
          .def("to_string", &pyreth::TensorInfo::toString)
          .def("__eq__", &pyreth::TensorInfo::operator==)
          .def("__ne__", &pyreth::TensorInfo::operator!=)
          .def("__repr__", &pyreth::TensorInfo::toString)
          .def("__str__", &pyreth::TensorInfo::toString);

  m.attr("IOTensorInfo") = tensor_info_cls;

  nb::class_<pyreth::ModelInfo>(
      m, "ModelInfo",
      "Cached model metadata and IO schema for zero-overhead runtime inspection and validation.")
      .def(nb::init<>())
      .def_ro("producer_name", &pyreth::ModelInfo::producer_name)
      .def_ro("graph_name", &pyreth::ModelInfo::graph_name)
      .def_ro("domain", &pyreth::ModelInfo::domain)
      .def_ro("description", &pyreth::ModelInfo::description)
      .def_ro("graph_description", &pyreth::ModelInfo::graph_description)
      .def_ro("version", &pyreth::ModelInfo::version)
      .def_ro("model_path", &pyreth::ModelInfo::model_path)
      .def_ro("inputs", &pyreth::ModelInfo::inputs)
      .def_ro("outputs", &pyreth::ModelInfo::outputs)
      .def_ro("custom_metadata", &pyreth::ModelInfo::custom_metadata)
      .def_prop_ro("input_count", &pyreth::ModelInfo::inputCount)
      .def_prop_ro("output_count", &pyreth::ModelInfo::outputCount)
      .def_prop_ro("has_dynamic_batch", &pyreth::ModelInfo::hasDynamicBatch)
      .def_prop_ro("has_dynamic_inputs", &pyreth::ModelInfo::hasDynamicInputs)
      .def_prop_ro("has_dynamic_outputs", &pyreth::ModelInfo::hasDynamicOutputs)
      .def(
          "find_input",
          [](const pyreth::ModelInfo& info,
             std::string_view name) -> std::optional<pyreth::TensorInfo> {
            const auto* ptr = info.findInput(name);
            return ptr ? std::optional<pyreth::TensorInfo>(*ptr) : std::nullopt;
          },
          nb::arg("name"))
      .def(
          "find_output",
          [](const pyreth::ModelInfo& info,
             std::string_view name) -> std::optional<pyreth::TensorInfo> {
            const auto* ptr = info.findOutput(name);
            return ptr ? std::optional<pyreth::TensorInfo>(*ptr) : std::nullopt;
          },
          nb::arg("name"))
      .def(
          "get_input_index",
          [](const pyreth::ModelInfo& info, std::string_view name) {
            auto res = info.getInputIndex(name);
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return res.value();
          },
          nb::arg("name"))
      .def(
          "get_output_index",
          [](const pyreth::ModelInfo& info, std::string_view name) {
            auto res = info.getOutputIndex(name);
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return res.value();
          },
          nb::arg("name"))
      .def(
          "validate_input",
          [](const pyreth::ModelInfo& info, nb::handle index_or_name, nb::handle tensor_obj) {
            auto t = coerceToTensor(tensor_obj);
            pyreth::Status st = nb::isinstance<nb::str>(index_or_name)
                                    ? info.validateInput(nb::cast<std::string>(index_or_name), t)
                                    : info.validateInput(nb::cast<size_t>(index_or_name), t);
            if (!st.isOk()) {
              throwForError(st.error());
            }
          },
          nb::arg("index_or_name"), nb::arg("tensor"))
      .def(
          "validate_inputs",
          [](const pyreth::ModelInfo& info, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            if (parsed.is_named) {
              if (parsed.named.size() != info.inputs.size()) {
                throw nb::value_error(
                    "Number of named input tensors does not match model input count");
              }
              std::vector<pyreth::Tensor> ordered(info.inputs.size());
              for (const auto& [name, tensor] : parsed.named) {
                auto idx_res = info.getInputIndex(name);
                if (!idx_res.isOk()) {
                  throwForError(idx_res.error());
                }
                ordered[idx_res.value()] = tensor;
              }
              auto st = info.validateInputs(std::span<const pyreth::Tensor>(ordered));
              if (!st.isOk()) {
                throwForError(st.error());
              }
              return;
            }
            auto st = info.validateInputs(std::span<const pyreth::Tensor>(parsed.positional));
            if (!st.isOk()) {
              throwForError(st.error());
            }
          },
          nb::arg("inputs"))
      .def("to_string", &pyreth::ModelInfo::toString)
      .def("__repr__", &pyreth::ModelInfo::toString)
      .def("__str__", &pyreth::ModelInfo::toString);

  // ---------------------------------------------------------------------------
  // ThreadPool, ThreadPoolConfig, ThreadPoolMetrics, ThreadBudget & Future
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::ThreadBudget>(
      m, "ThreadBudget", "Recommended thread partitioning between Pyreth and ONNX Runtime.")
      .def_ro("pyreth_worker_threads", &pyreth::ThreadBudget::pyreth_worker_threads)
      .def_ro("ort_intra_op_threads", &pyreth::ThreadBudget::ort_intra_op_threads)
      .def_ro("ort_inter_op_threads", &pyreth::ThreadBudget::ort_inter_op_threads)
      .def("__repr__", [](const pyreth::ThreadBudget& b) {
        return "ThreadBudget(pyreth_worker_threads=" + std::to_string(b.pyreth_worker_threads) +
               ", ort_intra_op_threads=" + std::to_string(b.ort_intra_op_threads) +
               ", ort_inter_op_threads=" + std::to_string(b.ort_inter_op_threads) + ")";
      });

  nb::class_<pyreth::ThreadPoolConfig>(m, "ThreadPoolConfig",
                                       "Configuration parameters for ThreadPool creation.")
      .def(
          "__init__",
          [](pyreth::ThreadPoolConfig* cfg, size_t num_threads, size_t queue_capacity,
             size_t reserve_threads_for_ort) {
            new (cfg)
                pyreth::ThreadPoolConfig{num_threads, queue_capacity, reserve_threads_for_ort};
          },
          nb::arg("num_threads") = 0, nb::arg("queue_capacity") = 1024,
          nb::arg("reserve_threads_for_ort") = 1)
      .def_rw("num_threads", &pyreth::ThreadPoolConfig::num_threads)
      .def_rw("queue_capacity", &pyreth::ThreadPoolConfig::queue_capacity)
      .def_rw("reserve_threads_for_ort", &pyreth::ThreadPoolConfig::reserve_threads_for_ort);

  nb::class_<pyreth::ThreadPoolMetrics>(m, "ThreadPoolMetrics",
                                        "Runtime execution telemetry for ThreadPool.")
      .def_ro("tasks_submitted", &pyreth::ThreadPoolMetrics::tasks_submitted)
      .def_ro("tasks_executed", &pyreth::ThreadPoolMetrics::tasks_executed)
      .def_ro("tasks_stolen", &pyreth::ThreadPoolMetrics::tasks_stolen)
      .def("to_dict",
           [](const pyreth::ThreadPoolMetrics& m_stat) {
             nb::dict d;
             d["tasks_submitted"] = m_stat.tasks_submitted;
             d["tasks_executed"] = m_stat.tasks_executed;
             d["tasks_stolen"] = m_stat.tasks_stolen;
             return d;
           })
      .def("__repr__", [](const pyreth::ThreadPoolMetrics& m_stat) {
        return "ThreadPoolMetrics(tasks_submitted=" + std::to_string(m_stat.tasks_submitted) +
               ", tasks_executed=" + std::to_string(m_stat.tasks_executed) +
               ", tasks_stolen=" + std::to_string(m_stat.tasks_stolen) + ")";
      });

  nb::class_<pyreth::ThreadPool>(m, "ThreadPool",
                                 "High-throughput lock-free work-stealing thread pool.")
      .def(nb::init<size_t>(), nb::arg("num_threads") = 0)
      .def(nb::init<const pyreth::ThreadPoolConfig&>(), nb::arg("config"))
      .def_prop_ro("num_workers", &pyreth::ThreadPool::numWorkers)
      .def_prop_ro("is_shutdown", &pyreth::ThreadPool::isShutdown)
      .def("metrics", &pyreth::ThreadPool::metrics,
           "Return a snapshot of ThreadPool runtime execution metrics.")
      .def(
          "shutdown",
          [](pyreth::ThreadPool& self) {
            nb::gil_scoped_release release;
            self.shutdown();
          },
          "Cooperatively stop all worker threads and wait for them to join (releases GIL).")
      .def_static("compute_thread_budget", &pyreth::ThreadPool::computeThreadBudget,
                  nb::arg("reserved_for_ort") = 1,
                  "Compute recommended thread partition between Pyreth and ONNX Runtime.")
      .def("__enter__", [](nb::object self) { return self; })
      .def(
          "__exit__",
          [](pyreth::ThreadPool& self, nb::handle, nb::handle, nb::handle) {
            nb::gil_scoped_release release;
            self.shutdown();
          },
          nb::arg("exc_type") = nb::none(), nb::arg("exc_val") = nb::none(),
          nb::arg("exc_tb") = nb::none());

  using InferenceTaskHandle = pyreth::TaskHandle<pyreth::Result<std::vector<pyreth::Tensor>>>;

  nb::class_<InferenceTaskHandle>(
      m, "InferenceFuture",
      "Asynchronous handle for an in-flight inference request or dynamic batch submission.")
      .def_prop_ro("valid", &InferenceTaskHandle::valid)
      .def("is_ready", &InferenceTaskHandle::isReady,
           "Return True if the inference result is ready without blocking.")
      .def("done", &InferenceTaskHandle::isReady,
           "Return True if the inference result is ready without blocking.")
      .def(
          "wait",
          [](const InferenceTaskHandle& self) {
            nb::gil_scoped_release release;
            self.wait();
          },
          "Block until the inference result is ready (explicitly releases Python GIL).")
      .def(
          "wait_for",
          [](const InferenceTaskHandle& self, double timeout_ms) {
            nb::gil_scoped_release release;
            const auto us = static_cast<int64_t>(timeout_ms * 1000.0);
            return self.waitFor(std::chrono::microseconds(std::max<int64_t>(0, us)));
          },
          nb::arg("timeout_ms"),
          "Wait up to timeout_ms milliseconds for the result (releases GIL). Returns True if "
          "ready.")
      .def(
          "get",
          [](InferenceTaskHandle& self) {
            if (!self.valid()) {
              throw nb::value_error(
                  "InferenceFuture is invalid or its result has already been retrieved");
            }
            pyreth::Result<std::vector<pyreth::Tensor>> res = [&]() {
              nb::gil_scoped_release release;
              return self.get();
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          "Block until execution completes (releases GIL) and return the output list[Tensor].")
      .def(
          "result",
          [](InferenceTaskHandle& self) {
            if (!self.valid()) {
              throw nb::value_error(
                  "InferenceFuture is invalid or its result has already been retrieved");
            }
            pyreth::Result<std::vector<pyreth::Tensor>> res = [&]() {
              nb::gil_scoped_release release;
              return self.get();
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          "Alias for get(): block until execution completes (releases GIL) and return "
          "list[Tensor].")
      .def(
          "get_numpy",
          [](InferenceTaskHandle& self) {
            if (!self.valid()) {
              throw nb::value_error(
                  "InferenceFuture is invalid or its result has already been retrieved");
            }
            pyreth::Result<std::vector<pyreth::Tensor>> res = [&]() {
              nb::gil_scoped_release release;
              return self.get();
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            nb::list out_list;
            for (auto& t : res.value()) {
              out_list.append(tensorToNumpyObject(t));
            }
            return out_list;
          },
          "Block until execution completes (releases GIL) and return zero-copy list[np.ndarray].");

  // ---------------------------------------------------------------------------
  // SessionConfig & InferenceSession
  // ---------------------------------------------------------------------------
  nb::class_<pyreth::SessionConfig>(m, "SessionConfig",
                                    "Fine-grained configuration options for an InferenceSession.")
      .def(
          "__init__",
          [](pyreth::SessionConfig* cfg, int intra_op_num_threads, int inter_op_num_threads,
             pyreth::OptimizationLevel graph_optimization_level,
             pyreth::SessionExecutionMode execution_mode, bool enable_cpu_mem_arena,
             bool enable_mem_pattern, pyreth::LogLevel log_severity_level, std::string log_id,
             std::optional<std::vector<pyreth::ExecutionProvider>> providers,
             bool allow_provider_fallback, bool enable_profiling, std::string profile_file_prefix) {
            new (cfg) pyreth::SessionConfig();
            cfg->intra_op_num_threads = intra_op_num_threads;
            cfg->inter_op_num_threads = inter_op_num_threads;
            cfg->graph_optimization_level = graph_optimization_level;
            cfg->execution_mode = execution_mode;
            cfg->enable_cpu_mem_arena = enable_cpu_mem_arena;
            cfg->enable_mem_pattern = enable_mem_pattern;
            cfg->log_severity_level = log_severity_level;
            cfg->log_id = std::move(log_id);
            if (providers.has_value()) {
              cfg->providers = std::move(*providers);
            }
            cfg->allow_provider_fallback = allow_provider_fallback;
            cfg->enable_profiling = enable_profiling;
            cfg->profile_file_prefix = std::move(profile_file_prefix);
          },
          nb::arg("intra_op_num_threads") = 1, nb::arg("inter_op_num_threads") = 1,
          nb::arg("graph_optimization_level") = pyreth::OptimizationLevel::All,
          nb::arg("execution_mode") = pyreth::SessionExecutionMode::Sequential,
          nb::arg("enable_cpu_mem_arena") = true, nb::arg("enable_mem_pattern") = true,
          nb::arg("log_severity_level") = pyreth::LogLevel::Warning,
          nb::arg("log_id") = "PyrethSession", nb::arg("providers") = nb::none(),
          nb::arg("allow_provider_fallback") = true, nb::arg("enable_profiling") = false,
          nb::arg("profile_file_prefix") = "")
      .def_rw("intra_op_num_threads", &pyreth::SessionConfig::intra_op_num_threads)
      .def_rw("inter_op_num_threads", &pyreth::SessionConfig::inter_op_num_threads)
      .def_rw("graph_optimization_level", &pyreth::SessionConfig::graph_optimization_level)
      .def_rw("execution_mode", &pyreth::SessionConfig::execution_mode)
      .def_rw("enable_cpu_mem_arena", &pyreth::SessionConfig::enable_cpu_mem_arena)
      .def_rw("enable_mem_pattern", &pyreth::SessionConfig::enable_mem_pattern)
      .def_rw("log_severity_level", &pyreth::SessionConfig::log_severity_level)
      .def_rw("log_id", &pyreth::SessionConfig::log_id)
      .def_rw("providers", &pyreth::SessionConfig::providers)
      .def_rw("allow_provider_fallback", &pyreth::SessionConfig::allow_provider_fallback)
      .def_rw("enable_profiling", &pyreth::SessionConfig::enable_profiling)
      .def_rw("profile_file_prefix", &pyreth::SessionConfig::profile_file_prefix)
      .def("set_intra_op_num_threads", &pyreth::SessionConfig::setIntraOpNumThreads,
           nb::rv_policy::reference)
      .def("set_inter_op_num_threads", &pyreth::SessionConfig::setInterOpNumThreads,
           nb::rv_policy::reference)
      .def("set_graph_optimization_level", &pyreth::SessionConfig::setGraphOptimizationLevel,
           nb::rv_policy::reference)
      .def("set_execution_mode", &pyreth::SessionConfig::setExecutionMode, nb::rv_policy::reference)
      .def("set_enable_cpu_mem_arena", &pyreth::SessionConfig::setEnableCpuMemArena,
           nb::rv_policy::reference)
      .def("set_enable_mem_pattern", &pyreth::SessionConfig::setEnableMemPattern,
           nb::rv_policy::reference)
      .def("set_log_severity_level", &pyreth::SessionConfig::setLogSeverityLevel,
           nb::rv_policy::reference)
      .def("set_log_id", &pyreth::SessionConfig::setLogId, nb::rv_policy::reference)
      .def("set_providers", &pyreth::SessionConfig::setProviders, nb::rv_policy::reference)
      .def("add_provider", &pyreth::SessionConfig::addProvider, nb::rv_policy::reference)
      .def("set_allow_provider_fallback", &pyreth::SessionConfig::setAllowProviderFallback,
           nb::rv_policy::reference)
      .def("set_enable_profiling", &pyreth::SessionConfig::enableProfiling,
           nb::arg("profile_file_prefix") = "", nb::rv_policy::reference)
      .def("disable_profiling", &pyreth::SessionConfig::disableProfiling, nb::rv_policy::reference);

  nb::class_<pyreth::InferenceSession>(
      m, "InferenceSession",
      "High-performance ONNX Runtime session wrapper with zero-copy execution and GIL release.")
      .def(
          "__init__",
          [](pyreth::InferenceSession* self, nb::handle model_path_or_bytes,
             const pyreth::SessionConfig& config) {
            new (self)
                pyreth::InferenceSession(createSessionFromPyObject(model_path_or_bytes, config));
          },
          nb::arg("model_path_or_bytes"), nb::arg("config") = pyreth::SessionConfig{},
          "Load an ONNX model from a file path (str/PathLike) or in-memory bytes buffer.")
      .def_static(
          "create",
          [](nb::handle model_path_or_bytes, const pyreth::SessionConfig& config) {
            return createSessionFromPyObject(model_path_or_bytes, config);
          },
          nb::arg("model_path_or_bytes"), nb::arg("config") = pyreth::SessionConfig{},
          "Factory method constructing an InferenceSession from a file path or in-memory bytes.")
      .def_prop_ro("is_valid", &pyreth::InferenceSession::isValid)
      .def_prop_ro("config", &pyreth::InferenceSession::getConfig)
      .def_prop_ro("active_providers", &pyreth::InferenceSession::getActiveProviders)
      .def_prop_ro("model_path", &pyreth::InferenceSession::getModelPath)
      .def_prop_ro("input_count", &pyreth::InferenceSession::getInputCount)
      .def_prop_ro("output_count", &pyreth::InferenceSession::getOutputCount)
      .def_prop_ro("input_names", &pyreth::InferenceSession::getInputNames)
      .def_prop_ro("output_names", &pyreth::InferenceSession::getOutputNames)
      .def_prop_ro("model_info", &pyreth::InferenceSession::getModelInfo)
      .def_prop_ro("inputs", &pyreth::InferenceSession::getInputs)
      .def_prop_ro("outputs", &pyreth::InferenceSession::getOutputs)
      .def(
          "get_input_info",
          [](const pyreth::InferenceSession& self, nb::handle index_or_name) {
            auto res = nb::isinstance<nb::str>(index_or_name)
                           ? self.getInputInfo(nb::cast<std::string>(index_or_name))
                           : self.getInputInfo(nb::cast<size_t>(index_or_name));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return res.value();
          },
          nb::arg("index_or_name"), "Retrieve TensorInfo for a model input by index or name.")
      .def(
          "get_output_info",
          [](const pyreth::InferenceSession& self, nb::handle index_or_name) {
            auto res = nb::isinstance<nb::str>(index_or_name)
                           ? self.getOutputInfo(nb::cast<std::string>(index_or_name))
                           : self.getOutputInfo(nb::cast<size_t>(index_or_name));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return res.value();
          },
          nb::arg("index_or_name"), "Retrieve TensorInfo for a model output by index or name.")
      .def(
          "validate_inputs",
          [](const pyreth::InferenceSession& self, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            if (parsed.is_named) {
              const auto& info = self.getModelInfo();
              if (parsed.named.size() != info.inputs.size()) {
                throw nb::value_error(
                    "Number of named input tensors does not match model input count");
              }
              std::vector<pyreth::Tensor> ordered(info.inputs.size());
              for (const auto& [name, tensor] : parsed.named) {
                auto idx_res = info.getInputIndex(name);
                if (!idx_res.isOk()) {
                  throwForError(idx_res.error());
                }
                ordered[idx_res.value()] = tensor;
              }
              auto st = self.validateInputs(std::span<const pyreth::Tensor>(ordered));
              if (!st.isOk()) {
                throwForError(st.error());
              }
              return;
            }
            auto st = self.validateInputs(std::span<const pyreth::Tensor>(parsed.positional));
            if (!st.isOk()) {
              throwForError(st.error());
            }
          },
          nb::arg("inputs"), "Validate input Tensors/ndarrays against the session's ModelInfo.")
      .def(
          "run",
          [](pyreth::InferenceSession& self, nb::handle inputs_arg, pyreth::ArenaAllocator* arena,
             pyreth::MemoryPool* pool, nb::object outputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);

            if (!outputs_arg.is_none()) {
              auto parsed_outputs = parseInputsArgument(outputs_arg);
              if (parsed_outputs.is_named) {
                throw nb::type_error("Pre-allocated outputs must be a positional sequence");
              }
              std::vector<pyreth::Tensor> ordered_inputs = std::move(parsed.positional);
              if (parsed.is_named) {
                const auto& info = self.getModelInfo();
                if (parsed.named.size() != info.inputs.size()) {
                  throw nb::value_error(
                      "Number of named input tensors does not match model input count");
                }
                ordered_inputs.resize(info.inputs.size());
                for (const auto& [name, tensor] : parsed.named) {
                  auto idx_res = info.getInputIndex(name);
                  if (!idx_res.isOk()) {
                    throwForError(idx_res.error());
                  }
                  ordered_inputs[idx_res.value()] = tensor;
                }
              }

              pyreth::Status st = [&]() {
                nb::gil_scoped_release release;
                return self.run(std::span<const pyreth::Tensor>(ordered_inputs),
                                std::span<pyreth::Tensor>(parsed_outputs.positional));
              }();
              if (!st.isOk()) {
                throwForError(st.error());
              }
              return parsed_outputs.positional;
            }

            pyreth::Result<std::vector<pyreth::Tensor>> res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                if (arena == nullptr && pool == nullptr) {
                  return self.run(parsed.named);
                }
                const auto& info = self.getModelInfo();
                if (parsed.named.size() != info.inputs.size()) {
                  return pyreth::Result<std::vector<pyreth::Tensor>>::failure(
                      pyreth::StatusCode::INVALID_ARGUMENT,
                      "Number of named input tensors does not match model input count");
                }
                std::vector<pyreth::Tensor> ordered(info.inputs.size());
                for (const auto& [name, tensor] : parsed.named) {
                  auto idx_res = info.getInputIndex(name);
                  if (!idx_res.isOk()) {
                    return pyreth::Result<std::vector<pyreth::Tensor>>::failure(idx_res.error());
                  }
                  ordered[idx_res.value()] = tensor;
                }
                if (arena != nullptr) {
                  return self.run(std::span<const pyreth::Tensor>(ordered), *arena);
                }
                return self.run(std::span<const pyreth::Tensor>(ordered), *pool);
              }

              if (arena != nullptr) {
                return self.run(std::span<const pyreth::Tensor>(parsed.positional), *arena);
              }
              if (pool != nullptr) {
                return self.run(std::span<const pyreth::Tensor>(parsed.positional), *pool);
              }
              return self.run(std::span<const pyreth::Tensor>(parsed.positional));
            }();

            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("inputs"), nb::arg("arena") = nb::none(), nb::arg("pool") = nb::none(),
          nb::arg("outputs") = nb::none(),
          "Execute synchronous zero-copy inference (explicitly releases Python GIL during ORT "
          "execution).")
      .def(
          "run_numpy",
          [](pyreth::InferenceSession& self, nb::handle inputs_arg, pyreth::ArenaAllocator* arena,
             pyreth::MemoryPool* pool) {
            auto parsed = parseInputsArgument(inputs_arg);
            pyreth::Result<std::vector<pyreth::Tensor>> res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                return self.run(parsed.named);
              }
              if (arena != nullptr) {
                return self.run(std::span<const pyreth::Tensor>(parsed.positional), *arena);
              }
              if (pool != nullptr) {
                return self.run(std::span<const pyreth::Tensor>(parsed.positional), *pool);
              }
              return self.run(std::span<const pyreth::Tensor>(parsed.positional));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            nb::list out_list;
            for (auto& t : res.value()) {
              out_list.append(tensorToNumpyObject(t));
            }
            return out_list;
          },
          nb::arg("inputs"), nb::arg("arena") = nb::none(), nb::arg("pool") = nb::none(),
          "Execute synchronous zero-copy inference (releasing GIL) and return zero-copy "
          "list[np.ndarray].")
      .def(
          "run_async",
          [](pyreth::InferenceSession& self, pyreth::ThreadPool& pool, nb::handle inputs_arg,
             pyreth::MemoryPool* output_pool) {
            auto parsed = parseInputsArgument(inputs_arg);
            std::vector<pyreth::Tensor> ordered_inputs = std::move(parsed.positional);
            if (parsed.is_named) {
              const auto& info = self.getModelInfo();
              if (parsed.named.size() != info.inputs.size()) {
                throw nb::value_error(
                    "Number of named input tensors does not match model input count");
              }
              ordered_inputs.resize(info.inputs.size());
              for (const auto& [name, tensor] : parsed.named) {
                auto idx_res = info.getInputIndex(name);
                if (!idx_res.isOk()) {
                  throwForError(idx_res.error());
                }
                ordered_inputs[idx_res.value()] = tensor;
              }
            }

            auto res = [&]() {
              nb::gil_scoped_release release;
              if (output_pool != nullptr) {
                return self.runAsync(pool, std::span<const pyreth::Tensor>(ordered_inputs),
                                     *output_pool);
              }
              return self.runAsync(pool, std::span<const pyreth::Tensor>(ordered_inputs));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("pool"), nb::arg("inputs"), nb::arg("output_pool") = nb::none(),
          "Dispatch asynchronous zero-copy inference onto a ThreadPool (releases GIL) and return "
          "an InferenceFuture.");

  // ---------------------------------------------------------------------------
  // DynamicBatcher, DynamicBatcherConfig, DynamicBatcherMetrics
  // ---------------------------------------------------------------------------
  auto batcher_cfg_cls =
      nb::class_<pyreth::DynamicBatcherConfig>(
          m, "DynamicBatcherConfig",
          "Configuration parameters for DynamicBatcher request scheduling and batching.")
          .def(
              "__init__",
              [](pyreth::DynamicBatcherConfig* cfg, size_t max_batch_size,
                 uint64_t max_batch_delay_ms, uint64_t max_batch_delay_us,
                 size_t max_queue_capacity, size_t num_workers, size_t arena_block_size) {
                new (cfg) pyreth::DynamicBatcherConfig{max_batch_size,     max_batch_delay_ms,
                                                       max_batch_delay_us, max_queue_capacity,
                                                       num_workers,        arena_block_size};
              },
              nb::arg("max_batch_size") = 16, nb::arg("max_batch_delay_ms") = 5,
              nb::arg("max_batch_delay_us") = 0, nb::arg("max_queue_capacity") = 1024,
              nb::arg("num_workers") = 0, nb::arg("arena_block_size") = pyreth::DEFAULT_BLOCK_SIZE)
          .def_rw("max_batch_size", &pyreth::DynamicBatcherConfig::max_batch_size)
          .def_rw("max_batch_delay_ms", &pyreth::DynamicBatcherConfig::max_batch_delay_ms)
          .def_rw("max_batch_delay_us", &pyreth::DynamicBatcherConfig::max_batch_delay_us)
          .def_rw("max_queue_capacity", &pyreth::DynamicBatcherConfig::max_queue_capacity)
          .def_rw("num_workers", &pyreth::DynamicBatcherConfig::num_workers)
          .def_rw("arena_block_size", &pyreth::DynamicBatcherConfig::arena_block_size)
          .def("set_max_batch_size", &pyreth::DynamicBatcherConfig::setMaxBatchSize,
               nb::rv_policy::reference)
          .def("set_max_batch_delay_ms", &pyreth::DynamicBatcherConfig::setMaxBatchDelayMs,
               nb::rv_policy::reference)
          .def("set_max_batch_delay_us", &pyreth::DynamicBatcherConfig::setMaxBatchDelayUs,
               nb::rv_policy::reference)
          .def("set_max_queue_capacity", &pyreth::DynamicBatcherConfig::setMaxQueueCapacity,
               nb::rv_policy::reference)
          .def("set_num_workers", &pyreth::DynamicBatcherConfig::setNumWorkers,
               nb::rv_policy::reference)
          .def("set_arena_block_size", &pyreth::DynamicBatcherConfig::setArenaBlockSize,
               nb::rv_policy::reference);

  m.attr("BatcherConfig") = batcher_cfg_cls;

  auto batcher_metrics_cls =
      nb::class_<pyreth::DynamicBatcherMetrics>(
          m, "DynamicBatcherMetrics",
          "Runtime throughput, batching efficiency, and latency telemetry for DynamicBatcher.")
          .def_ro("requests_submitted", &pyreth::DynamicBatcherMetrics::requests_submitted)
          .def_ro("requests_completed", &pyreth::DynamicBatcherMetrics::requests_completed)
          .def_ro("requests_rejected", &pyreth::DynamicBatcherMetrics::requests_rejected)
          .def_ro("batches_dispatched", &pyreth::DynamicBatcherMetrics::batches_dispatched)
          .def_ro("full_batches", &pyreth::DynamicBatcherMetrics::full_batches)
          .def_ro("timeout_batches", &pyreth::DynamicBatcherMetrics::timeout_batches)
          .def_ro("total_batched_samples", &pyreth::DynamicBatcherMetrics::total_batched_samples)
          .def_ro("total_batch_wait_us", &pyreth::DynamicBatcherMetrics::total_batch_wait_us)
          .def_ro("total_execution_us", &pyreth::DynamicBatcherMetrics::total_execution_us)
          .def_prop_ro("average_batch_size", &pyreth::DynamicBatcherMetrics::averageBatchSize)
          .def_prop_ro("average_batch_wait_us", &pyreth::DynamicBatcherMetrics::averageBatchWaitUs)
          .def_prop_ro("average_execution_us", &pyreth::DynamicBatcherMetrics::averageExecutionUs)
          .def("to_dict",
               [](const pyreth::DynamicBatcherMetrics& m_stat) {
                 nb::dict d;
                 d["requests_submitted"] = m_stat.requests_submitted;
                 d["requests_completed"] = m_stat.requests_completed;
                 d["requests_rejected"] = m_stat.requests_rejected;
                 d["batches_dispatched"] = m_stat.batches_dispatched;
                 d["full_batches"] = m_stat.full_batches;
                 d["timeout_batches"] = m_stat.timeout_batches;
                 d["total_batched_samples"] = m_stat.total_batched_samples;
                 d["total_batch_wait_us"] = m_stat.total_batch_wait_us;
                 d["total_execution_us"] = m_stat.total_execution_us;
                 d["average_batch_size"] = m_stat.averageBatchSize();
                 d["average_batch_wait_us"] = m_stat.averageBatchWaitUs();
                 d["average_execution_us"] = m_stat.averageExecutionUs();
                 return d;
               })
          .def("__repr__", [](const pyreth::DynamicBatcherMetrics& m_stat) {
            return "DynamicBatcherMetrics(requests_submitted=" +
                   std::to_string(m_stat.requests_submitted) +
                   ", requests_completed=" + std::to_string(m_stat.requests_completed) +
                   ", batches_dispatched=" + std::to_string(m_stat.batches_dispatched) +
                   ", average_batch_size=" + std::to_string(m_stat.averageBatchSize()) + ")";
          });

  m.attr("BatcherMetrics") = batcher_metrics_cls;

  nb::class_<pyreth::DynamicBatcher>(
      m, "DynamicBatcher",
      "High-throughput dynamic batching request scheduler and inference engine.")
      .def(
          "__init__",
          [](pyreth::DynamicBatcher* self, pyreth::InferenceSession& session,
             const pyreth::DynamicBatcherConfig& config, pyreth::ThreadPool* pool,
             pyreth::MemoryPool* memory_pool) {
            pyreth::Result<pyreth::DynamicBatcher> res =
                (pool != nullptr && memory_pool != nullptr)
                    ? pyreth::DynamicBatcher::create(session, *pool, *memory_pool, config)
                    : (pool != nullptr ? pyreth::DynamicBatcher::create(session, *pool, config)
                                       : pyreth::DynamicBatcher::create(session, config));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            new (self) pyreth::DynamicBatcher(std::move(res).value());
          },
          nb::arg("session"), nb::arg("config") = pyreth::DynamicBatcherConfig{},
          nb::arg("pool") = nb::none(), nb::arg("memory_pool") = nb::none(), nb::keep_alive<1, 2>(),
          nb::keep_alive<1, 4>(), nb::keep_alive<1, 5>(),
          "Construct a DynamicBatcher bound to an InferenceSession.")
      .def_static(
          "create",
          [](pyreth::InferenceSession& session, const pyreth::DynamicBatcherConfig& config,
             pyreth::ThreadPool* pool, pyreth::MemoryPool* memory_pool) {
            pyreth::Result<pyreth::DynamicBatcher> res =
                (pool != nullptr && memory_pool != nullptr)
                    ? pyreth::DynamicBatcher::create(session, *pool, *memory_pool, config)
                    : (pool != nullptr ? pyreth::DynamicBatcher::create(session, *pool, config)
                                       : pyreth::DynamicBatcher::create(session, config));
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("session"), nb::arg("config") = pyreth::DynamicBatcherConfig{},
          nb::arg("pool") = nb::none(), nb::arg("memory_pool") = nb::none(), nb::keep_alive<0, 1>(),
          nb::keep_alive<0, 3>(), nb::keep_alive<0, 4>(),
          "Factory method constructing a DynamicBatcher.")
      .def(
          "submit",
          [](pyreth::DynamicBatcher& self, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            auto res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                return self.submit(parsed.named);
              }
              return self.submit(std::span<const pyreth::Tensor>(parsed.positional));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("inputs"),
          "Enqueue an asynchronous inference request (releases GIL) and return an "
          "InferenceFuture.")
      .def(
          "run_async",
          [](pyreth::DynamicBatcher& self, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            auto res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                return self.submit(parsed.named);
              }
              return self.submit(std::span<const pyreth::Tensor>(parsed.positional));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("inputs"),
          "Alias for submit(): enqueue an asynchronous inference request and return an "
          "InferenceFuture.")
      .def(
          "run",
          [](pyreth::DynamicBatcher& self, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            auto res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                return self.run(parsed.named);
              }
              return self.run(std::span<const pyreth::Tensor>(parsed.positional));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            return std::move(res).value();
          },
          nb::arg("inputs"),
          "Submit an inference request to the batcher and block until its batch completes "
          "(explicitly releases Python GIL so concurrent Python threads coalesce into batches).")
      .def(
          "run_numpy",
          [](pyreth::DynamicBatcher& self, nb::handle inputs_arg) {
            auto parsed = parseInputsArgument(inputs_arg);
            auto res = [&]() {
              nb::gil_scoped_release release;
              if (parsed.is_named) {
                return self.run(parsed.named);
              }
              return self.run(std::span<const pyreth::Tensor>(parsed.positional));
            }();
            if (!res.isOk()) {
              throwForError(res.error());
            }
            nb::list out_list;
            for (auto& t : res.value()) {
              out_list.append(tensorToNumpyObject(t));
            }
            return out_list;
          },
          nb::arg("inputs"),
          "Submit an inference request (releasing GIL) and return demultiplexed zero-copy "
          "list[np.ndarray].")
      .def(
          "flush",
          [](pyreth::DynamicBatcher& self) {
            nb::gil_scoped_release release;
            self.flush();
          },
          "Immediately flush all queued requests into batches and wait for completion (releases "
          "GIL).")
      .def(
          "shutdown",
          [](pyreth::DynamicBatcher& self) {
            nb::gil_scoped_release release;
            self.shutdown();
          },
          "Drain remaining queued requests, stop the scheduler thread, and wait for in-flight "
          "batches (releases GIL).")
      .def_prop_ro("is_shutdown", &pyreth::DynamicBatcher::isShutdown)
      .def_prop_ro("is_valid", &pyreth::DynamicBatcher::isValid)
      .def_prop_ro("config", &pyreth::DynamicBatcher::getConfig)
      .def_prop_ro("pending_count", &pyreth::DynamicBatcher::pendingCount)
      .def("metrics", &pyreth::DynamicBatcher::metrics,
           "Return a snapshot of DynamicBatcher runtime throughput and latency metrics.")
      .def("reset_metrics", &pyreth::DynamicBatcher::resetMetrics,
           "Reset all runtime telemetry counters to zero.")
      .def("__enter__", [](nb::object self) { return self; })
      .def(
          "__exit__",
          [](pyreth::DynamicBatcher& self, nb::handle, nb::handle, nb::handle) {
            nb::gil_scoped_release release;
            self.shutdown();
          },
          nb::arg("exc_type") = nb::none(), nb::arg("exc_val") = nb::none(),
          nb::arg("exc_tb") = nb::none());
}
