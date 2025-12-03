cmake_minimum_required(VERSION 3.16)
project(CSR4MPI LANGUAGES CXX)

set(CSR4MPI_VALUE_TYPE "1" CACHE STRING "Scalar type: 0=float,1=double,2=complex<float>,3=complex<double>")
option(CSR4MPI_ENABLE_BLAS "Enable optional BLAS placeholder path" OFF)
option(CSR4MPI_ENABLE_OPENMP "Enable OpenMP parallel kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)

# GTest discovery: prefer system find_package, fallback to FetchContent
option(CSR4MPI_FORCE_FETCH_GTEST "Force FetchContent for GoogleTest instead of find_package" OFF)

if(NOT CSR4MPI_FORCE_FETCH_GTEST)
    find_package(GTest QUIET)
endif()

if(GTest_FOUND)
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
add_subdirectory(bench)
add_subdirectory(tools)
