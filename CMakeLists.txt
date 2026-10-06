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

# Export compile commands for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Engine Core Target (Static Library)
# This library contains the core inference logic, thread pool
# and memory management.
add_library(EngineCore STATIC
    src/core/pyreth.cpp
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

  add_subdirectory(unittest)
endif()

# Executable / Sandbox Runner
add_executable(Sandbox
    sandbox/main.cpp
)

target_link_libraries(Sandbox
    PRIVATE
        EngineCore
)

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
