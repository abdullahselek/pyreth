cmake_minimum_required(VERSION 3.25...4.4)

project(pyreth
    VERSION 0.1.0
    LANGUAGES CXX C
)

# Standard C++ configuration
# We use C++20 for modern features like std::jthread and concepts
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position Independent Code (required when static libraries are linked into shared modules on Linux)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ONNX Runtime Integration
include(cmake/OnnxRuntime.cmake)

# Server Dependencies (JsonCpp & Drogon)
include(FetchContent)

set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  jsoncpp
  GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
  GIT_TAG 1.9.5
  GIT_SHALLOW TRUE
  SYSTEM
)

FetchContent_MakeAvailable(jsoncpp)

if(TARGET jsoncpp_static AND NOT TARGET jsoncpp_lib)
  add_library(jsoncpp_lib ALIAS jsoncpp_static)
endif()

# Provide CMake config package for Drogon to find jsoncpp target cleanly
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake/jsoncpp/jsoncppConfig.cmake"
  "set(jsoncpp_FOUND TRUE)\nset(JSONCPP_INCLUDE_DIRS \"${jsoncpp_SOURCE_DIR}/include\")\n"
)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/jsoncpp")

set(BUILD_CTL OFF CACHE BOOL "Build drogon_ctl" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Build drogon testing" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(BUILD_ORM OFF CACHE BOOL "Build drogon ORM" FORCE)
set(BUILD_YAML_CONFIG OFF CACHE BOOL "Build yaml config" FORCE)
set(BUILD_BROTLI OFF CACHE BOOL "Build brotli" FORCE)

# On Linux, Drogon requires UUID (provided by libuuid / uuid-dev).
# If not present on the host system, fetch a self-contained libuuid via FetchContent.
if(NOT WIN32 AND NOT APPLE AND NOT CMAKE_SYSTEM_NAME MATCHES "FreeBSD|OpenBSD")
  find_path(UUID_INCLUDE_DIR NAMES uuid.h PATH_SUFFIXES uuid)
  find_library(UUID_LIBRARY NAMES uuid)
  if(NOT UUID_INCLUDE_DIR OR NOT UUID_LIBRARY)
    FetchContent_Declare(
      libuuid
      GIT_REPOSITORY https://github.com/gershnik/libuuid-cmake.git
      GIT_TAG v2.39.1
      GIT_SHALLOW TRUE
      SYSTEM
    )
    FetchContent_MakeAvailable(libuuid)
    get_target_property(LIBUUID_INC uuid::uuid INTERFACE_INCLUDE_DIRECTORIES)
    set(UUID_INCLUDE_DIR "${LIBUUID_INC}" CACHE PATH "UUID include directory" FORCE)
    set(UUID_INCLUDE_DIRS "${LIBUUID_INC}" CACHE PATH "UUID include directory" FORCE)
    set(UUID_LIBRARY "uuid::uuid" CACHE STRING "UUID library" FORCE)
    set(UUID_LIBRARIES "uuid::uuid" CACHE STRING "UUID library" FORCE)
    set(UUID_FOUND TRUE CACHE BOOL "UUID found" FORCE)
  endif()
endif()

# On Windows and other systems without system ZLIB, fetch ZLIB via FetchContent.
find_package(ZLIB QUIET)
if(NOT ZLIB_FOUND)
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SKIP_INSTALL_ALL ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib.git
    GIT_TAG v1.3.1
    GIT_SHALLOW TRUE
    SYSTEM
  )
  FetchContent_MakeAvailable(zlib)
  if(TARGET zlibstatic AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlibstatic)
  elseif(TARGET zlib AND NOT TARGET ZLIB::ZLIB)
    add_library(ZLIB::ZLIB ALIAS zlib)
  endif()
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cmake/zlib/ZLIBConfig.cmake"
    "set(ZLIB_FOUND TRUE)\nset(ZLIB_INCLUDE_DIRS \"${zlib_SOURCE_DIR};${zlib_BINARY_DIR}\")\nif(NOT TARGET ZLIB::ZLIB)\n  if(TARGET zlibstatic)\n    add_library(ZLIB::ZLIB ALIAS zlibstatic)\n  elseif(TARGET zlib)\n    add_library(ZLIB::ZLIB ALIAS zlib)\n  endif()\nendif()\n"
  )
  list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_BINARY_DIR}/cmake/zlib")
endif()

FetchContent_Declare(
  drogon
  GIT_REPOSITORY https://github.com/drogonframework/drogon.git
  GIT_TAG v1.9.14
  GIT_SHALLOW TRUE
  GIT_SUBMODULES_RECURSE TRUE
  SYSTEM
)

FetchContent_MakeAvailable(drogon)

# Suppress compiler warnings and deprecation errors from Drogon, Trantor, and Zlib sources
foreach(_tgt IN ITEMS drogon trantor zlibstatic)
  if(TARGET ${_tgt})
    if(MSVC)
      target_compile_options(${_tgt} PRIVATE /w)
    else()
      target_compile_options(${_tgt} PRIVATE -w -Wno-error -Wno-deprecated-declarations)
    endif()
    target_compile_definitions(${_tgt} PRIVATE OPENSSL_SUPPRESS_DEPRECATED)
  endif()
endforeach()

# Re-enable BUILD_TESTING for Pyreth unit tests
set(BUILD_TESTING ON CACHE BOOL "Build Pyreth unit tests" FORCE)

# Engine Core Target (Static Library)
# This library contains the core inference logic, thread pool
# and memory management.
add_library(EngineCore STATIC
    src/core/pyreth.cpp
    src/core/backend.cpp
    src/core/runtime.cpp
    src/core/inference_session.cpp
    src/core/dynamic_batcher.cpp
    src/core/thread_pool.cpp
    src/core/memory_pool.cpp
    src/core/tensor.cpp
)

set_target_properties(EngineCore PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(EngineCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

get_target_property(ENGINE_CORE_TYPE EngineCore TYPE)
if(ENGINE_CORE_TYPE STREQUAL "STATIC_LIBRARY")
  target_compile_definitions(EngineCore PUBLIC PYRETH_STATIC_DEFINE)
else()
  target_compile_definitions(EngineCore PRIVATE PYRETH_EXPORT)
endif()

# Unified warning flags across compilers
if(MSVC)
  target_compile_options(EngineCore PRIVATE
        /W4
        /permissive-
  )
else()
  # Apple Clang / Clang warning and architecture flags
  target_compile_options(EngineCore PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
  )
endif()

# Sanitizers (AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer)
option(ENABLE_ASAN "Enable AddressSanitizer (ASan)" OFF)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer (UBSan)" OFF)
option(ENABLE_TSAN "Enable ThreadSanitizer (TSan)" OFF)

if(ENABLE_ASAN AND ENABLE_TSAN)
  message(FATAL_ERROR "ENABLE_ASAN and ENABLE_TSAN cannot be enabled simultaneously.")
endif()

set(PYRETH_SANITIZER_FLAGS "")
if(ENABLE_ASAN)
  list(APPEND PYRETH_SANITIZER_FLAGS "-fsanitize=address")
endif()
if(ENABLE_UBSAN)
  list(APPEND PYRETH_SANITIZER_FLAGS "-fsanitize=undefined" "-fno-sanitize-recover=all")
endif()
if(ENABLE_TSAN)
  list(APPEND PYRETH_SANITIZER_FLAGS "-fsanitize=thread")
endif()

if(PYRETH_SANITIZER_FLAGS)
  if(MSVC)
    if(ENABLE_ASAN)
      target_compile_options(EngineCore PUBLIC /fsanitize=address /Zi)
    endif()
  else()
    target_compile_options(EngineCore PUBLIC
      -fno-omit-frame-pointer
      -g
      ${PYRETH_SANITIZER_FLAGS}
    )
    target_link_options(EngineCore PUBLIC
      ${PYRETH_SANITIZER_FLAGS}
    )
  endif()
endif()

# Pyreth Server Target (Static Library)
add_library(PyrethServer STATIC
    src/server/http_server.cpp
)

set_target_properties(PyrethServer PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(PyrethServer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(PyrethServer PUBLIC PYRETH_STATIC_DEFINE)

target_link_libraries(PyrethServer
    PUBLIC
        EngineCore
        drogon
        jsoncpp_lib
)

if(WIN32)
  target_link_libraries(PyrethServer
    PUBLIC
        ws2_32
        rpcrt4
        crypt32
        advapi32
        shlwapi
        iphlpapi
  )
endif()

if(MSVC)
  target_compile_options(PyrethServer PRIVATE
        /W4
        /permissive-
  )
else()
  target_compile_options(PyrethServer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
  )
endif()

if(PYRETH_SANITIZER_FLAGS)
  if(MSVC)
    if(ENABLE_ASAN)
      target_compile_options(PyrethServer PUBLIC /fsanitize=address /Zi)
    endif()
  else()
    target_compile_options(PyrethServer PUBLIC
      -fno-omit-frame-pointer
      -g
      ${PYRETH_SANITIZER_FLAGS}
    )
    target_link_options(PyrethServer PUBLIC
      ${PYRETH_SANITIZER_FLAGS}
    )
  endif()
endif()

# Testing (CTest)
# BUILD_TESTING defaults to ON (from CTest) so the "Quick Start" flow
# (cmake -B build && ctest --test-dir build) works out of the box.
# Pass -DBUILD_TESTING=OFF to skip fetching and building the test suite.
include(CTest)

if(BUILD_TESTING)
  include(FetchContent)

  # Pin GoogleTest to a known-good release. FetchContent downloads and builds
  # it as part of this project, so no system-wide GTest install is required.
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    SYSTEM
  )

  # Cross-platform knobs: build GTest against a shared CRT on Windows and do
  # not install it alongside our artifacts.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

  # If a target already exists (e.g. a cached or pre-provided GTest), don't
  # re-fetch. Otherwise download, extract, and build it now.
  if(NOT TARGET gtest)
    FetchContent_MakeAvailable(googletest)
  endif()

  # Suppress compiler warnings from GoogleTest internal sources
  foreach(_tgt IN ITEMS gtest gmock gtest_main gmock_main)
    if(TARGET ${_tgt})
      if(MSVC)
        target_compile_options(${_tgt} PRIVATE /w)
      else()
        target_compile_options(${_tgt} PRIVATE -w)
      endif()
    endif()
  endforeach()

  add_subdirectory(unittest)
endif()

# Micro-benchmarking Suite (Google Benchmark)
option(BUILD_BENCHMARKS "Build Google Benchmark micro-benchmark suite" ON)

if(BUILD_BENCHMARKS AND NOT SKBUILD)
  include(FetchContent)

  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    SYSTEM
  )

  if(NOT TARGET benchmark::benchmark)
    FetchContent_MakeAvailable(googlebenchmark)
  endif()

  foreach(_tgt IN ITEMS benchmark benchmark_main)
    if(TARGET ${_tgt})
      if(MSVC)
        target_compile_options(${_tgt} PRIVATE /w)
      else()
        target_compile_options(${_tgt} PRIVATE -w)
      endif()
    endif()
  endforeach()

  add_subdirectory(benchmarks)
endif()

# Executable / Sandbox Runner
add_executable(Sandbox
    sandbox/main.cpp
)

target_link_libraries(EngineCore
    PUBLIC
        onnxruntime::onnxruntime
)

target_link_libraries(Sandbox
    PRIVATE
        EngineCore
)

# On Windows, copy ONNX Runtime DLLs to the executable directory
pyreth_copy_ort_dlls(Sandbox)

# Apple Silicon specific frameworks linkage
if(APPLE)
  target_link_libraries(EngineCore
        PUBLIC
            "-framework Cocoa"
            "-framework IOKit"
            "-framework CoreVideo"
            "-framework Metal"
  )
endif()

# Python Extension Module (nanobind)
# Enabled automatically when invoked by scikit-build-core
# (SKBUILD) or explicitly via -DBUILD_PYTHON_BINDINGS=ON
option(BUILD_PYTHON_BINDINGS "Build Python bindings via nanobind" OFF)

if(SKBUILD OR BUILD_PYTHON_BINDINGS)
  find_package(Python 3.13 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core
    python/pyreth_core.cpp
  )

  target_link_libraries(_core
    PRIVATE
        EngineCore
  )

  # Set RPATH so _core resolves bundled ONNX Runtime libraries
  if(APPLE)
    set_target_properties(_core PROPERTIES
      INSTALL_RPATH "@loader_path"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  else()
    set_target_properties(_core PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()

  pyreth_copy_ort_dlls(_core)

  install(TARGETS _core
    LIBRARY DESTINATION pyreth
    RUNTIME DESTINATION pyreth
  )

  # Install ONNX Runtime shared libraries into the pyreth package
  # for self-contained wheels
  if(WIN32)
    get_target_property(ORT_DLL_PATH onnxruntime::onnxruntime IMPORTED_LOCATION)
    if(ORT_DLL_PATH)
      get_filename_component(ORT_DLL_DIR "${ORT_DLL_PATH}" DIRECTORY)
      file(GLOB ORT_ALL_DLLS "${ORT_DLL_DIR}/*.dll")
      install(FILES ${ORT_ALL_DLLS} DESTINATION pyreth)
    endif()
  else()
    get_target_property(ORT_LIB_PATH onnxruntime::onnxruntime IMPORTED_LOCATION)
    if(ORT_LIB_PATH)
      get_filename_component(ORT_LIB_DIR "${ORT_LIB_PATH}" DIRECTORY)
      file(GLOB ORT_ALL_LIBS "${ORT_LIB_DIR}/*.dylib" "${ORT_LIB_DIR}/*.so*")
      install(FILES ${ORT_ALL_LIBS} DESTINATION pyreth)
    endif()
  endif()
endif()
