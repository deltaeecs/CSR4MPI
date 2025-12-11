cmake_minimum_required(VERSION 3.16)
project(CSR4MPI LANGUAGES CXX)

# CSR4MPI_VALUE_TYPE is now deprecated - all types are compiled into the library.
# This option is kept for backward compatibility but has no effect.
set(CSR4MPI_VALUE_TYPE "1" CACHE STRING "DEPRECATED: All scalar types are now always compiled. This option has no effect.")
option(CSR4MPI_ENABLE_BLAS "Enable optional BLAS placeholder path" OFF)
option(CSR4MPI_ENABLE_OPENMP "Enable OpenMP parallel kernels" ON)

# Default to ON if standalone, OFF if subproject
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(CSR4MPI_BUILD_TESTS "Build CSR4MPI tests" ON)
else()
    option(CSR4MPI_BUILD_TESTS "Build CSR4MPI tests" OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

if(CSR4MPI_BUILD_TESTS)
    # GTest discovery: prefer system find_package, fallback to FetchContent
    option(CSR4MPI_FORCE_FETCH_GTEST "Force FetchContent for GoogleTest instead of find_package" OFF)

    if(NOT CSR4MPI_FORCE_FETCH_GTEST)
        find_package(GTest QUIET)
    endif()

    if(GTest_FOUND OR GTEST_FOUND)
        message(STATUS "Using system GTest via find_package")
    else()
        message(STATUS "GTest not found via find_package, using FetchContent")
        include(FetchContent)

        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )

        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

        FetchContent_MakeAvailable(googletest)
    endif()

    include(CTest)
    enable_testing()

    add_subdirectory(tests)
endif()

option(CSR4MPI_BUILD_BENCHMARKS "Build CSR4MPI benchmarks" OFF)
if(CSR4MPI_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

option(CSR4MPI_BUILD_TOOLS "Build CSR4MPI tools" OFF)
if(CSR4MPI_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
