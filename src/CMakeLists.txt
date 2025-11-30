set(SRC_LIST
    CSRMatrix.cpp
    Distribution.cpp
    CommPattern.cpp
    MumpsAdapter.cpp
    CSRComm.cpp
    Operations.cpp
    BlasAdapter.cpp
    MatrixMarketLoader.cpp
)

add_library(csr4mpi STATIC ${SRC_LIST})
target_include_directories(csr4mpi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_compile_definitions(csr4mpi PUBLIC CSR4MPI_VALUE_TYPE=${CSR4MPI_VALUE_TYPE})
if(CSR4MPI_ENABLE_BLAS)
    target_compile_definitions(csr4mpi PUBLIC CSR4MPI_USE_BLAS)
endif()

find_package(MPI REQUIRED)
target_compile_definitions(csr4mpi PUBLIC CSR4MPI_USE_MPI)
target_link_libraries(csr4mpi PUBLIC MPI::MPI_CXX)

if(CSR4MPI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(csr4mpi PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(csr4mpi PUBLIC CSR4MPI_USE_OPENMP)
endif()