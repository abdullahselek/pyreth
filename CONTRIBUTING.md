# Contributing to Pyreth

Thank you for your interest in contributing to the Pyreth AI Inference Engine! To maintain high performance and code quality, please adhere to the following standards.

## Directory Structure

- `include/pyreth/`: Public API headers.
- `src/core/`: Internal implementation details.
- `unittest/`: C++ unit tests (GoogleTest).
- `docs/`: Technical documentation and architecture guides.
- `python/`: Python bindings (nanobind).
- `tests/python/`: Python integration tests.

## Coding Standards

### Naming Conventions

- **Classes/Structs**: `PascalCase` (e.g., `InferenceSession`).
- **Methods/Functions**: `camelCase` (e.g., `runInference`).
- **Variables**: `snake_case` (e.g., `buffer_size`).
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_THREADS`).
- **Namespaces**: `pyreth` (lowercase).

### Performance & Safety

- **C++ Standard**: C++20.
- **Memory**: Use the `MemoryPool` for tensor data; avoid raw `new/delete` in the hot path.
- **Concurrency**: Prefer the internal `ThreadPool` and work-stealing primitives over raw threads.
- **Headers**: Use `#pragma once` and minimize includes in public headers.

## Pull Request Process

1.  **Modular Changes**: Each PR should focus on a single sub-phase or feature.
2.  **Testing**: Every change must be accompanied by unit tests in `unittest/`.
3.  **Documentation**: Update relevant `.md` files or header comments.
4.  **Formatting**: Ensure code follows `clang-format`.

## Build System

We use CMake. Ensure your changes do not break the build on macOS or Linux.
