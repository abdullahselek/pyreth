#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <cctype>
#include <cstdint>
#include <memory>
#include <pyreth/backend.hpp>
#include <pyreth/common.hpp>
#include <pyreth/memory_pool.hpp>
#include <pyreth/tensor.hpp>

namespace nb = nanobind;

namespace {

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
  // Check for NumPy read-only flag before wrapping
  if (nb::hasattr(obj, "flags")) {
    nb::object flags = obj.attr("flags");
    if (nb::hasattr(flags, "writeable") && !nb::cast<bool>(flags.attr("writeable"))) {
      throw nb::value_error("Cannot create mutable zero-copy Tensor from a read-only NumPy array");
    }
  }

  // Validate buffer writeability via Python Buffer Protocol if available
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

  // Retain a strong Python reference tied to the Tensor's ExternalStorage lifetime
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

  // Create a capsule holding a shared_ptr<TensorStorage> so the backing memory
  // (OwnedStorage, ArenaStorage, PooledArenaStorage, or ExternalStorage) remains
  // pinned until all NumPy arrays and views referencing it are garbage-collected.
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
  // DataType & StorageType Enums
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
              throw nb::value_error(std::string(res.error().message()).c_str());
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
              throw nb::value_error(std::string(res.error().message()).c_str());
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
              throw nb::value_error(std::string(res.error().message()).c_str());
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
              throw nb::value_error(std::string(res.error().message()).c_str());
            }
            return std::move(res).value();
          },
          nb::arg("new_shape"), "Return a zero-copy reshaped view of the Tensor.")
      .def(
          "slice",
          [](const pyreth::Tensor& self, size_t dim, int64_t start, int64_t end, int64_t step) {
            auto res = self.slice(dim, start, end, step);
            if (!res.isOk()) {
              throw nb::value_error(std::string(res.error().message()).c_str());
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
              throw nb::value_error(std::string(res.error().message()).c_str());
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
}
