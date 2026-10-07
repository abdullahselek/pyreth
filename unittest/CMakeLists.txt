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

# Unified warning flags across compilers
if(MSVC)
  target_compile_options(pyreth_tests PRIVATE
        /W4
        /permissive-
  )
else()
  target_compile_options(pyreth_tests
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
)
endif()

# On Windows, copy ONNX Runtime DLLs next to the test executable
pyreth_copy_ort_dlls(pyreth_tests)

# Register each GTest case with CTest so it can be run/filtered by name,
# e.g. `ctest --test-dir build -R "Foundation.EngineLifecycle"`.
include(GoogleTest)

if(WIN32)
  get_target_property(ORT_DLL_PATH onnxruntime::onnxruntime IMPORTED_LOCATION)
  if(ORT_DLL_PATH)
    get_filename_component(ORT_DLL_DIR "${ORT_DLL_PATH}" DIRECTORY)
    gtest_discover_tests(pyreth_tests
      PROPERTIES ENVIRONMENT "PATH=${ORT_DLL_DIR}\;$ENV{PATH}"
    )
  else()
    gtest_discover_tests(pyreth_tests)
  endif()
else()
  gtest_discover_tests(pyreth_tests)
endif()
