#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>

#include <pyreth/backend.hpp>
#include <pyreth/common.hpp>

namespace nb = nanobind;

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
}
