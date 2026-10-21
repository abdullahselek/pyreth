```bash
██████╗ ██╗   ██╗██████╗ ███████╗████████╗██╗  ██╗
██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝╚══██╔══╝██║  ██║
██████╔╝ ╚████╔╝ ██████╔╝█████╗     ██║   ███████║
██╔═══╝   ╚██╔╝  ██╔══██╗██╔══╝     ██║   ██╔══██║
██║        ██║   ██║  ██║███████╗   ██║   ██║  ██║
╚═╝        ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝
```

High throughput, low latency parallel processing AI model inference engine.

## Overview

pyreth is designed to serve AI model inference with minimal latency and maximum throughput. It is built around a lightweight core (`EngineCore`) that will grow into a parallel inference pipeline (batching, scheduling, GPU/Metal dispatch), keeping the hot path lock-free and cache-friendly.

## Project Layout

```
├── cmake/            # CMake discovery and helper modules
├── docs/             # Architecture and guidelines documentation
├── include/pyreth/   # Public C++ API headers
├── python/pyreth/    # Python package & nanobind bindings
├── sandbox/          # Dev runner / manual integration checks
├── src/core/         # Engine core implementation
├── tests/python/     # Python pytest integration tests
├── unittest/         # C++ GTest suite
├── CMakeLists.txt    # CMake build configuration
├── pyproject.toml    # Python packaging configuration (uv / scikit-build-core)
└── build/            # Build artifacts (gitignored)
```

## Requirements

- CMake >= 3.25
- C++20 compiler (Apple Clang / Clang, GCC, or MSVC)
- `clang-format` (for C++ code formatting)
- [uv](https://docs.astral.sh/uv/) (for Python environment and dependency management)
- Python >= 3.13 (managed automatically by `uv`)

## Quick Start

Configure, build, and run the unit tests in three steps:

```sh
cmake -B build              # configure (GoogleTest is fetched on first run)
cmake --build build         # build the library, sandbox, and tests
ctest --test-dir build      # run the unit tests
```

## Building

```sh
# Configure (Debug)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug

# Configure (Release)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build

# Build in parallel (4 jobs)
cmake --build build -j 4
```

The build produces:

- `build/EngineCore.a` — static engine library
- `build/Sandbox` — dev runner executable

## Testing

Unit tests live in `unittest/` and are built with GoogleTest (fetched via CMake `FetchContent` on first configure). Every `*.cpp` file in `unittest/` is picked up automatically, and each test case is registered with CTest.

```sh
# Configure with tests enabled
cmake -B build -S . -DBUILD_TESTING=ON

# Build
cmake --build build

# Run all tests from the build directory
cd build && ctest --output-on-failure

# Or from the project root
ctest --test-dir build --output-on-failure

# Verbose test output
ctest --test-dir build --output-on-failure -V
```

## Running the Sandbox
 
```sh
./build/Sandbox
```

## Python Development (`uv`)

Pyreth uses **`uv`** for dependency resolution, virtual environment management, and packaging with `scikit-build-core` and `nanobind`.

### 1. Setup & Build Extension

```sh
# Sync the virtual environment and compile the C++ nanobind extension
uv sync
```

### 2. Run Python Tests

```sh
# Run pytest tests using the uv-managed environment
uv run pytest
```

### 3. Interactive Verification

```sh
uv run python -c "import pyreth; print('Version:', pyreth.__version__, '| Best Backend:', pyreth.get_best_backend())"
```

## Code Style & Formatting

Repository standards are enforced automatically in CI on every push and pull request via the `standards` job before running build/test workflows.

### C++ (`clang-format` & `clang-tidy`)

C++ formatting is governed by [`.clang-format`](.clang-format) based on Google C++20 conventions (2-space indent, 100-character line limit, left pointer alignment `Type* ptr`, strict control-statement braces).

- **Check formatting (dry run, matches CI)**:
  ```sh
  find include src unittest sandbox python \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format --dry-run --Werror
  ```

- **Format in-place**:
  ```sh
  # Format all C++ files across the codebase
  find include src unittest sandbox python \( -name '*.hpp' -o -name '*.cpp' \) -print0 | xargs -0 clang-format -i

  # Format an individual file
  clang-format -i include/pyreth/thread_pool.hpp
  ```

- **Linting (`clang-tidy`)**:
  Static analysis checks and naming conventions are configured in [`.clang-tidy`](.clang-tidy). Generate the compile commands database with CMake first:
  ```sh
  cmake -B build -S . -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  clang-tidy src/core/thread_pool.cpp -p build
  ```

> **Editor / IDE Support**: If you use Neovim / LazyVim (with `conform.nvim` or `clangd`), VS Code, or CLion, the editor will automatically discover and apply [`.clang-format`](.clang-format) upon saving.

### Python (`ruff`)

Python code formatting and linting are managed using [Ruff](https://docs.astral.sh/ruff/) via `uv`:

- **Check linting and formatting**:
  ```sh
  uv run ruff check .
  uv run ruff format --check .
  ```

- **Auto-format and fix**:
  ```sh
  uv run ruff format .
  uv run ruff check --fix .
  ```

## License

See [LICENSE](LICENSE).
