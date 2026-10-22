# Micro-benchmark target for Pyreth (Google Benchmark).

add_executable(pyreth_benchmarks
    microbenchmarks.cpp
)

target_link_libraries(pyreth_benchmarks
    PRIVATE
        EngineCore
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(pyreth_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(MSVC)
  target_compile_options(pyreth_benchmarks PRIVATE
        /W4
        /permissive-
  )
else()
  target_compile_options(pyreth_benchmarks
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
  )
endif()

pyreth_copy_ort_dlls(pyreth_benchmarks)

if(BUILD_TESTING)
  add_test(
    NAME PyrethMicrobenchmarksSmoke
    COMMAND pyreth_benchmarks --benchmark_min_time=0.001s
  )
endif()
