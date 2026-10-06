# Unit test target for Pyreth.
#
# Every `*.cpp` file dropped into this directory is automatically compiled into
# a single test executable. `test_main.cpp` provides `main()` (so we link only
# `GTest::gtest`, not `GTest::gtest_main`), and `gtest_discover_tests` registers
# each individual test case with CTest for per-test `ctest` visibility.

# CONFIGURE_DEPENDS re-runs the glob when files are added/removed so the build
# stays in sync without a manual re-configure step.
file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(pyreth_tests
    ${TEST_SOURCES}
)

target_link_libraries(pyreth_tests
    PRIVATE
        EngineCore
        GTest::gtest
)

target_include_directories(pyreth_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Match the core library's strict warning set for consistency.
target_compile_options(pyreth_tests
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
)

# Register each GTest case with CTest so it can be run/filtered by name,
# e.g. `ctest --test-dir build -R "Foundation.EngineLifecycle"`.
include(GoogleTest)
gtest_discover_tests(pyreth_tests)
