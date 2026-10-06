# pyreth

High throughput, low latency parallel processing AI model inference engine.

## Overview

pyreth is designed to serve AI model inference with minimal latency and maximum throughput. It is built around a lightweight core (`EngineCore`) that will grow into a parallel inference pipeline (batching, scheduling, GPU/Metal dispatch), keeping the hot path lock-free and cache-friendly.

## Project Layout

```
├── include/engine/   # Public headers
├── src/core/         # Engine core implementation
├── sandbox/          # Dev runner / manual integration checks
├── CMakeLists.txt    # Build configuration
└── build/            # Build artifacts (gitignored)
```

## Requirements

- CMake >= 3.25
- C++20 compiler (Apple Clang / Clang or GCC)
- Internet access on first configure (CMake `FetchContent` downloads GoogleTest)

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

## License

See [LICENSE](LICENSE).
