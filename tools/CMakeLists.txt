add_executable(csr4mpi_gen_mm
    gen_mm.cpp
)
target_compile_definitions(csr4mpi_gen_mm PRIVATE CSR4MPI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")