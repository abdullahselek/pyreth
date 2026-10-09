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

# Engine Core Target (Static Library)
# This library contains the core inference logic, thread pool
# and memory management.
add_library(EngineCore STATIC
    src/core/pyreth.cpp
    src/core/backend.cpp
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
