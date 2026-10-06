cmake_minimum_required(VERSION 3.25...4.4)

project(pyreth
    VERSION 0.1.0
    LANGUAGES CXX C
)

# Standard C++ configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for tooling (clangd, clang-tidy, etc.)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Engine Core Target (Static Library)
add_library(EngineCore STATIC
    src/core/pyreth.cpp
)

target_include_directories(EngineCore
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Apple Clang / Clang warning and architecture flags
target_compile_options(EngineCore
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
)

# Testing (CTest) — add test targets under a tests/ dir as the project grows
include(CTest)

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
