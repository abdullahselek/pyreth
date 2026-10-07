# OnnxRuntime.cmake - Discovery and Prebuilt Binary Management for ONNX Runtime

set(ONNXRUNTIME_VERSION "1.19.2" CACHE STRING "ONNX Runtime version to download if not found locally")

# 1. Search for system-installed or user-specified ONNX Runtime
set(ORT_SEARCH_PATHS
    ${ONNXRUNTIME_ROOT}
    $ENV{ONNXRUNTIME_ROOT}
    /opt/homebrew/opt/onnxruntime
    /usr/local/opt/onnxruntime
    /usr/local
    /usr
)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ORT_SEARCH_PATHS}
    PATH_SUFFIXES include onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${ORT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "Found system ONNX Runtime:")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")
    set(ORT_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR}")
    set(ORT_LIBRARY "${ONNXRUNTIME_LIBRARY}")
else()
    # 2. Determine platform and architecture for official prebuilt binary
    set(ORT_PLATFORM "")
    set(ORT_EXT "tgz")

    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(ORT_PLATFORM "osx-arm64")
        else()
            set(ORT_PLATFORM "osx-x86_64")
        endif()
    elseif(UNIX AND NOT APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(ORT_PLATFORM "linux-aarch64")
        else()
            set(ORT_PLATFORM "linux-x64")
        endif()
    elseif(WIN32)
        set(ORT_EXT "zip")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(ORT_PLATFORM "win-arm64")
        else()
            set(ORT_PLATFORM "win-x64")
        endif()
    endif()

    if(ORT_PLATFORM STREQUAL "")
        message(FATAL_ERROR "Unsupported platform for prebuilt ONNX Runtime: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(ORT_ARCHIVE_NAME "onnxruntime-${ORT_PLATFORM}-${ONNXRUNTIME_VERSION}")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ORT_ARCHIVE_NAME}.${ORT_EXT}")

    message(STATUS "Using prebuilt ONNX Runtime ${ONNXRUNTIME_VERSION} for ${ORT_PLATFORM}")
    message(STATUS "Downloading from: ${ORT_URL}")

    include(FetchContent)
    FetchContent_Declare(
        onnxruntime_prebuilt
        URL ${ORT_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    set(ORT_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/include")
    find_library(ORT_LIBRARY
        NAMES onnxruntime
        PATHS "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )
endif()

# 3. Create imported target onnxruntime::onnxruntime
if(NOT TARGET onnxruntime::onnxruntime)
    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        IMPORTED_LOCATION "${ORT_LIBRARY}"
    )

    if(WIN32)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${ORT_LIBRARY}"
        )
    endif()

    get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
    set_target_properties(onnxruntime::onnxruntime PROPERTIES
        INTERFACE_LINK_DIRECTORIES "${ORT_LIB_DIR}"
    )
endif()

message(STATUS "ONNX Runtime configured successfully.")
