# Error Handling Guidelines

Pyreth is designed for **low-latency, high-throughput parallel AI model inference**. In high-frequency inference loops and lock-free concurrency pipelines, traditional C++ exception handling (`throw` / `try` / `catch`) introduces non-deterministic stack unwinding penalties and inhibits compiler optimizations.

To guarantee predictable tail latency, Pyreth adopts a **monadic, zero-allocation value-or-error model** inspired by modern C++ functional patterns and `std::expected`.

---

## 1. Core Architecture

The error handling subsystem is defined in [`include/pyreth/result.hpp`](../include/pyreth/result.hpp) and comprises four main primitives:

| Component | Description | Memory Overhead |
| :--- | :--- | :--- |
| `StatusCode` | `uint8_t` enum class categorizing discrete failure modes | 1 byte |
| `Error` | Encapsulates a `StatusCode` and an immutable `std::string_view` description | 24 bytes (stack-allocated) |
| `Status` | Type alias for `Result<void, Error>`, representing success or failure without payload | $\le$ 32 bytes |
| `Result<T, E>` | Monadic template holding either a successful value `T` or an `Error` | Stack-allocated (`std::variant<T, E>`) |

---

## 2. Core Types Reference

### `StatusCode`
An exhaustive enumeration of failure conditions:
* **Argument validation**: `INVALID_ARGUMENT`, `INVALID_DIMENSIONS`
* **Model lifecycle**: `MODEL_NOT_FOUND`, `MODEL_LOAD_FAILED`
* **Hardware backends**: `BACKEND_ERROR`, `BACKEND_UNAVAILABLE`
* **Work queue & concurrency**: `QUEUE_FULL`, `QUEUE_EMPTY`, `TIMEOUT`
* **Resource management**: `ALLOCATION_FAILED`, `INTERNAL_ERROR`, `UNKNOWN`

Each code provides a constant-time, zero-allocation string representation via `statusCodeToString(code)`.

### `Error`
A lightweight, zero-allocation diagnostic object:
```cpp
// Construct with default message:
Error err1(StatusCode::MODEL_NOT_FOUND);

// Construct with specific string literal or static message:
Error err2(StatusCode::INVALID_ARGUMENT, "Input tensor shape mismatch");

// Convenience factory methods:
Error err3 = Error::modelNotFound("models/resnet50.onnx");
Error err4 = Error::queueFull();
```

---

## 3. Usage Patterns

### Returning Values or Errors
Functions should return `Result<T>` when producing an output, or `Status` when performing an action without a return value.

```cpp
#include <pyreth/result.hpp>

// Returning a value or error
pyreth::Result<ModelMetadata> loadMetadata(std::string_view model_path) {
  if (model_path.empty()) {
    return pyreth::Error::invalidArgument("Model path cannot be empty");
  }
  if (!fileExists(model_path)) {
    return pyreth::StatusCode::MODEL_NOT_FOUND; // Implicit conversion from StatusCode
  }

  ModelMetadata meta = parseModel(model_path);
  return meta; // Implicit conversion from value
}

// Returning status without payload
pyreth::Status validateInput(const Tensor& input) {
  if (input.dimensions().empty()) {
    return pyreth::Error::invalidDimensions("Input tensor has no dimensions");
  }
  return pyreth::Status::ok();
}
```

### Checking Results
Use idiomatic conditional checks:

```cpp
auto result = loadMetadata("resnet50.onnx");

if (!result) {
  std::cerr << "Failed: " << result.error().message() << "\n";
  return;
}

// Access value by reference or move
const ModelMetadata& meta = result.value();
ModelMetadata owned_meta = std::move(result).value();

// Or provide a fallback value
ModelMetadata fallback_meta = result.valueOr(defaultMetadata);
```

---

## 4. Monadic Chaining

`Result<T>` and `Status` support functional monadic pipelines:

### `andThen` (FlatMap / Bind)
Executes the next operation only if the current result is successful; short-circuits immediately on error.

```cpp
pyreth::Result<InferenceOutput> output = readInputTensor(input_path)
    .andThen([](const Tensor& tensor) {
        return runPreprocessing(tensor);
    })
    .andThen([](const Tensor& preprocessed) {
        return session.run(preprocessed);
    });
```

### `transform` (Map)
Transforms the contained value using a callable, returning a new `Result<U>`.

```cpp
pyreth::Result<int> score = session.run(tensor)
    .transform([](const InferenceOutput& out) {
        return computeTopClass(out);
    });
```

### `orElse` (Error Recovery)
Interceps an error and provides recovery or fallback:

```cpp
pyreth::Result<Device> device = queryMetalBackend()
    .orElse([](const pyreth::Error& err) -> pyreth::Result<Device> {
        // Fall back to CPU if Metal is unavailable
        return queryCpuBackend();
    });
```

---

## 5. Best Practices & Rules for Pyreth

1. **No Exceptions on the Hot Path**: Never throw C++ exceptions inside tensor manipulation, memory allocation, queue operations, or inference loops.
2. **Zero-Allocation Messages**: `Error` stores `std::string_view`. Ensure string literals or static lifetime strings are passed to avoid dangling references or heap allocation.
3. **Boundaries & Foreign Runtimes**:
   * **Python / nanobind**: Translate `Result::isError()` into appropriate Python exceptions (e.g., `nb::value_error`, `nb::runtime_error`) at the C++ $\to$ Python binding layer.
   * **ONNX Runtime**: Catch and convert any `Ort::Exception` at the raw ORT wrapper boundary into a `pyreth::StatusCode::BACKEND_ERROR` immediately.
