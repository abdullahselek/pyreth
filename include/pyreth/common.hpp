#pragma once

namespace pyreth {

/**
 * @brief Current version of the Pyreth engine in SEMVER format.
 */
inline constexpr const char* PYRETH_VERSION = "0.1.0";

/**
 * @def PYRETH_API
 * @brief Macro for exporting/importing symbols in a cross-platform manner.
 * 
 * Uses __declspec(dllexport/dllimport) on Windows and visibility attributes on GCC/Clang.
 */
#if defined(_WIN32) || defined(__CYGWIN__)

#ifdef PYRETH_EXPORT
#define PYRETH_API __declspec(dllexport)
#else
#define PYRETH_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define PYRETH_API __attribute__((visibility("default")))
#else
#define PYRETH_API
#endif
#endif

} // namespace pyreth
