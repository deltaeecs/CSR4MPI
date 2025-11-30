add_executable(csr4mpi_bench
    bench_operations.cpp
)
target_link_libraries(csr4mpi_bench PRIVATE csr4mpi)
target_compile_definitions(csr4mpi_bench PRIVATE CSR4MPI_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CSR4MPI_VALUE_TYPE=${CSR4MPI_VALUE_TYPE})

add_executable(csr4mpi_bench_spmv
    bench_large_spmv.cpp
)
target_link_libraries(csr4mpi_bench_spmv PRIVATE csr4mpi)
target_compile_definitions(csr4mpi_bench_spmv PRIVATE CSR4MPI_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CSR4MPI_VALUE_TYPE=${CSR4MPI_VALUE_TYPE})
# MPI is required by the library; no extra conditionals needed here