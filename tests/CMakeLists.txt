include(GoogleTest)

# Select MPI launcher per-OS: MS-MPI on Windows, OpenMPI/MPICH otherwise
if (WIN32)
    find_program(MPIEXEC_EXECUTABLE_NT "mpiexec.exe" PATHS
        "$ENV{ProgramFiles}/Microsoft MPI/Bin"
        NO_DEFAULT_PATH)
    if (NOT MPIEXEC_EXECUTABLE_NT)
        find_program(MPIEXEC_EXECUTABLE_NT "mpiexec.exe")
    endif()
    if (MPIEXEC_EXECUTABLE_NT)
        set(MPIEXEC_EXECUTABLE "${MPIEXEC_EXECUTABLE_NT}")
        set(MPIEXEC_NUMPROC_FLAG "-n")
    endif()
else()
    # Linux/macOS: prefer mpirun, fallback to mpiexec
    find_program(MPIEXEC_EXECUTABLE "mpirun")
    if (NOT MPIEXEC_EXECUTABLE)
        find_program(MPIEXEC_EXECUTABLE "mpiexec")
    endif()
    if (NOT MPIEXEC_NUMPROC_FLAG)
        set(MPIEXEC_NUMPROC_FLAG "-n")
    endif()
endif()

if (MPIEXEC_EXECUTABLE)
    message(STATUS "MPI launcher: ${MPIEXEC_EXECUTABLE}")
else()
    message(WARNING "MPI launcher not found; MPI tests may not run")
endif()

add_executable(csr4mpi_tests
    test_csr_matrix.cpp
    test_csr_builder.cpp
    test_distribution_pattern.cpp
    test_mumps_adapter.cpp
    test_operations.cpp
    test_operations_correctness.cpp
    test_blas_placeholder.cpp
    test_matrix_market.cpp
    test_symmetric.cpp
    test_mpi_symmetric_spmm.cpp
    test_all_scalar_types.cpp
)

target_link_libraries(csr4mpi_tests PRIVATE csr4mpi GTest::gtest_main)
target_compile_definitions(csr4mpi_tests PRIVATE CSR4MPI_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(CSR4MPI_ENABLE_BLAS)
    target_compile_definitions(csr4mpi_tests PRIVATE CSR4MPI_USE_BLAS)
endif()

gtest_discover_tests(csr4mpi_tests)

# Optional MUMPS integration test (requires user-provided MUMPS paths)
option(CSR4MPI_ENABLE_MUMPS "Enable MUMPS integration tests" OFF)
option(CSR4MPI_MUMPS_DEBUG "Enable verbose debug logging in MUMPS test" ON)
option(CSR4MPI_MUMPS_MINIMAL "Use minimal 1x1 matrix for MUMPS debug" OFF)
if(CSR4MPI_ENABLE_MUMPS)
    # Discover MUMPS headers and libraries without hardcoded absolute paths.
    # Users may provide hints via -DMUMPS_ROOT, -DMUMPS_INCLUDE_DIR, -DMUMPS_LIBRARY_DIR or environment variables.
    set(_MUMPS_HINTS)
    if (DEFINED MUMPS_ROOT)
        list(APPEND _MUMPS_HINTS ${MUMPS_ROOT} ${MUMPS_ROOT}/include ${MUMPS_ROOT}/lib)
    endif()
    if (DEFINED ENV{MUMPS_ROOT})
        list(APPEND _MUMPS_HINTS $ENV{MUMPS_ROOT} $ENV{MUMPS_ROOT}/include $ENV{MUMPS_ROOT}/lib)
    endif()

    # Header: dmumps_c.h may be in include or include/mumps; prefer bundled Thirdparty/MUMPS/include
    if (NOT MUMPS_INCLUDE_DIR)
        find_path(MUMPS_INCLUDE_DIR
            NAMES dmumps_c.h
            PATHS "${CMAKE_SOURCE_DIR}/Thirdparty/MUMPS/include"
            PATH_SUFFIXES mumps
            HINTS ${_MUMPS_HINTS}
        )
    endif()

    # Libraries: dmumps, mumps_common; plus BLAS and MPI from system
    # Common library search hints
    set(_LIB_HINTS ${_MUMPS_HINTS} ${CMAKE_PREFIX_PATH} ${CMAKE_LIBRARY_PATH})
    if (DEFINED ENV{MSYSTEM_PREFIX})
        list(APPEND _LIB_HINTS $ENV{MSYSTEM_PREFIX}/lib)
    endif()
    if (WIN32)
        list(APPEND _LIB_HINTS "$ENV{ProgramFiles}/Microsoft MPI/Lib/x64")
    endif()
    # Prefer variant-specific DMUMPS (e.g., dmumps_mpi)
    if (NOT MUMPS_DMUMPS_LIB)
        find_library(MUMPS_DMUMPS_LIB NAMES dmumps_mpi HINTS ${_LIB_HINTS})
    endif()
    if (NOT MUMPS_COMMON_LIB)
        # Prefer MPI variant of common
        find_library(MUMPS_COMMON_LIB NAMES mumps_common_mpi mumps_pord HINTS ${_LIB_HINTS})
    endif()
    # BLAS: prefer OpenBLAS if available; fallback to system blas
    if (NOT BLAS_LIB)
        find_library(BLAS_LIB NAMES openblas blas HINTS ${_MUMPS_HINTS})
    endif()
    # Prefer CMake's ScaLAPACK package if available
    # Ensure MPI targets exist; select a global link target (CXX preferred, fallback to C)
    find_package(MPI QUIET)
    unset(CSR4MPI_MPI_LINK_TARGET)
    if (TARGET MPI::MPI_CXX)
        set(CSR4MPI_MPI_LINK_TARGET MPI::MPI_CXX)
    elseif (TARGET MPI::MPI_C)
        set(CSR4MPI_MPI_LINK_TARGET MPI::MPI_C)
    endif()
    # Locate ScaLAPACK by library name to avoid imported target issues
    if (NOT SCALAPACK_LIB)
        find_library(SCALAPACK_LIB NAMES scalapack HINTS ${_LIB_HINTS})
    endif()
    # METIS/ParMETIS
    if (NOT METIS_LIB)
        find_library(METIS_LIB NAMES metis HINTS ${_LIB_HINTS})
    endif()
    if (NOT PARMETIS_LIB)
        find_library(PARMETIS_LIB NAMES parmetis HINTS ${_LIB_HINTS})
    endif()
    # SCOTCH (and error lib)
    if (NOT SCOTCH_LIB)
        find_library(SCOTCH_LIB NAMES scotch HINTS ${_LIB_HINTS})
    endif()
    if (NOT SCOTCHERR_LIB)
        find_library(SCOTCHERR_LIB NAMES scotcherr HINTS ${_LIB_HINTS})
    endif()
    # Parallel SCOTCH (ptscotch) if available — required by some MUMPS builds
    if (NOT PTSCOTCH_LIB)
        find_library(PTSCOTCH_LIB NAMES ptscotch HINTS ${_LIB_HINTS})
    endif()
    if (NOT PTSCOTCHERR_LIB)
        find_library(PTSCOTCHERR_LIB NAMES ptscotcherr HINTS ${_LIB_HINTS})
    endif()
    # SCOTCH MUMPS bridge libraries (esmumps / parallel ptesmumps)
    if (NOT ESMUMPS_LIB)
        find_library(ESMUMPS_LIB NAMES esmumps HINTS ${_LIB_HINTS})
    endif()
    if (NOT PTESMUMPS_LIB)
        find_library(PTESMUMPS_LIB NAMES ptesmumps HINTS ${_LIB_HINTS})
    endif()
    # MPI lib: rely on compiler's MPI; don't hardcode name — runtime uses MPIEXEC
    # Allow user to pass extra libs via -DMUMPS_EXTRA_LIBS
    set(MUMPS_EXTRA_LIBS ${MUMPS_EXTRA_LIBS} gfortran)

    # Build library list in an order that satisfies dependencies:
    # mumps -> esmumps/scotch/parmetis/metis -> scalapack -> blas -> fortran
    set(_MUMPS_LIBS)
    # Core MUMPS libs first
    if (MUMPS_DMUMPS_LIB)
        list(APPEND _MUMPS_LIBS ${MUMPS_DMUMPS_LIB})
    else()
        list(APPEND _MUMPS_LIBS dmumps_mpi)
    endif()
    # Common + PORD ordering library (some builds separate pord)
    if (NOT MUMPS_PORD_LIB)
        find_library(MUMPS_PORD_LIB NAMES mumps_pord HINTS ${_LIB_HINTS})
    endif()
    if (MUMPS_COMMON_LIB)
        list(APPEND _MUMPS_LIBS ${MUMPS_COMMON_LIB})
    else()
        list(APPEND _MUMPS_LIBS mumps_common_mpi)
    endif()
    if (MUMPS_PORD_LIB)
        list(APPEND _MUMPS_LIBS ${MUMPS_PORD_LIB})
    else()
        # Fallback name
        list(APPEND _MUMPS_LIBS mumps_pord)
    endif()
    # SCOTCH bridge and graph libs
    if (ESMUMPS_LIB)
        list(APPEND _MUMPS_LIBS ${ESMUMPS_LIB})
    else()
        list(APPEND _MUMPS_LIBS esmumps)
    endif()
    if (PTSCOTCH_LIB)
        list(APPEND _MUMPS_LIBS ${PTSCOTCH_LIB})
    endif()
    if (PTSCOTCHERR_LIB)
        list(APPEND _MUMPS_LIBS ${PTSCOTCHERR_LIB})
    endif()
    if (SCOTCH_LIB)
        list(APPEND _MUMPS_LIBS ${SCOTCH_LIB})
    else()
        list(APPEND _MUMPS_LIBS scotch)
    endif()
    if (SCOTCHERR_LIB)
        list(APPEND _MUMPS_LIBS ${SCOTCHERR_LIB})
    else()
        list(APPEND _MUMPS_LIBS scotcherr)
    endif()
    # ParMETIS/METIS
    if (PARMETIS_LIB)
        list(APPEND _MUMPS_LIBS ${PARMETIS_LIB})
    else()
        list(APPEND _MUMPS_LIBS parmetis)
    endif()
    if (METIS_LIB)
        list(APPEND _MUMPS_LIBS ${METIS_LIB})
    else()
        list(APPEND _MUMPS_LIBS metis)
    endif()
    # ScaLAPACK (encapsulates BLACS on MSYS2) — link by name
    if (SCALAPACK_LIB)
        list(APPEND _MUMPS_LIBS ${SCALAPACK_LIB})
    else()
        list(APPEND _MUMPS_LIBS scalapack)
    endif()
    # BLAS last
    if (BLAS_LIB)
        list(APPEND _MUMPS_LIBS ${BLAS_LIB})
    endif()
    # Extra libs (e.g., gfortran) last
    if (PTESMUMPS_LIB)
        list(APPEND _MUMPS_LIBS ${PTESMUMPS_LIB})
    endif()
    if (MUMPS_EXTRA_LIBS)
        list(APPEND _MUMPS_LIBS ${MUMPS_EXTRA_LIBS})
    endif()

    if(MUMPS_INCLUDE_DIR AND _MUMPS_LIBS)
        add_executable(csr4mpi_mumps_tests
            test_mpi_mumps_adapter.cpp
        )
        target_include_directories(csr4mpi_mumps_tests PRIVATE ${MUMPS_INCLUDE_DIR})
        # Help linker find import libraries without hardcoding paths
        if (DEFINED ENV{MSYSTEM_PREFIX})
            target_link_directories(csr4mpi_mumps_tests PRIVATE $ENV{MSYSTEM_PREFIX}/lib)
        endif()
        target_link_libraries(csr4mpi_mumps_tests PRIVATE csr4mpi ${_MUMPS_LIBS})
        # Explicitly link MPI for ScaLAPACK configs that require it (use global selection)
        if (DEFINED CSR4MPI_MPI_LINK_TARGET)
            target_link_libraries(csr4mpi_mumps_tests PRIVATE ${CSR4MPI_MPI_LINK_TARGET})
        endif()
        target_compile_definitions(csr4mpi_mumps_tests PRIVATE CSR4MPI_SOURCE_DIR="${CMAKE_SOURCE_DIR}" CSR4MPI_USE_MUMPS)
        if (CSR4MPI_MUMPS_DEBUG)
            target_compile_definitions(csr4mpi_mumps_tests PRIVATE CSR4MPI_MUMPS_DEBUG)
        endif()
        if (CSR4MPI_MUMPS_MINIMAL)
            target_compile_definitions(csr4mpi_mumps_tests PRIVATE CSR4MPI_MUMPS_MINIMAL)
        endif()
        # No ScaLAPACK in this test; rely on local dense fallback
        if(CSR4MPI_ENABLE_BLAS)
            target_compile_definitions(csr4mpi_mumps_tests PRIVATE CSR4MPI_USE_BLAS)
        endif()
        if (MPIEXEC_EXECUTABLE)
            add_test(NAME mpi_mumps_adapter_4proc COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:csr4mpi_mumps_tests>)
            # Ensure MSYS2 runtime DLLs are resolvable on Windows by augmenting PATH for the test
            if (WIN32)
                # Compose candidate bin paths
                set(_CSR4MPI_CANDIDATE_PATHS)
                if (DEFINED ENV{MSYSTEM_PREFIX})
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "$ENV{MSYSTEM_PREFIX}/bin")
                endif()
                if (EXISTS "C:/msys64/mingw64/bin")
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "C:/msys64/mingw64/bin")
                endif()
                if (EXISTS "C:/msys64/bin")
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "C:/msys64/bin")
                endif()
                # Join into PATH env for the test
                if (_CSR4MPI_CANDIDATE_PATHS)
                    string(JOIN ";" _CSR4MPI_MSYS_PATH ${_CSR4MPI_CANDIDATE_PATHS})
                    set_tests_properties(mpi_mumps_adapter_4proc PROPERTIES
                        ENVIRONMENT "PATH=${_CSR4MPI_MSYS_PATH};$ENV{PATH}")
                endif()
            endif()
        else()
            message(WARNING "MPIEXEC not found; registering local MUMPS test fallback")
            add_test(NAME mumps_adapter_local COMMAND $<TARGET_FILE:csr4mpi_mumps_tests>)
            if (WIN32)
                # Also inject PATH for local run if available
                set(_CSR4MPI_CANDIDATE_PATHS)
                if (DEFINED ENV{MSYSTEM_PREFIX})
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "$ENV{MSYSTEM_PREFIX}/bin")
                endif()
                if (EXISTS "C:/msys64/mingw64/bin")
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "C:/msys64/mingw64/bin")
                endif()
                if (EXISTS "C:/msys64/bin")
                    list(APPEND _CSR4MPI_CANDIDATE_PATHS "C:/msys64/bin")
                endif()
                if (_CSR4MPI_CANDIDATE_PATHS)
                    string(JOIN ";" _CSR4MPI_MSYS_PATH ${_CSR4MPI_CANDIDATE_PATHS})
                    set_tests_properties(mumps_adapter_local PROPERTIES
                        ENVIRONMENT "PATH=${_CSR4MPI_MSYS_PATH};$ENV{PATH}")
                endif()
            endif()
        endif()
    else()
        message(WARNING "CSR4MPI_ENABLE_MUMPS=ON but MUMPS headers or libs not found.\n"
                        "MUMPS_INCLUDE_DIR='${MUMPS_INCLUDE_DIR}' dmumps='${MUMPS_DMUMPS_LIB}' mumps_common='${MUMPS_COMMON_LIB}' BLAS='${BLAS_LIB}'.\n"
                        "Provide -DMUMPS_INCLUDE_DIR and -DMUMPS_EXTRA_LIBS (e.g., dmumps;mumps_common;openblas).")
    endif()
endif()

# Provide data directory for MatrixMarket-based tests
set(CSR4MPI_TEST_DATA_DIR ${CMAKE_SOURCE_DIR}/tests/data)
file(MAKE_DIRECTORY ${CSR4MPI_TEST_DATA_DIR})
if (EXISTS ${CSR4MPI_TEST_DATA_DIR})
    # Apply environment to all tests so they can find data
    set_property(GLOBAL PROPERTY CTEST_ENVIRONMENT "CSR4MPI_DATA_DIR=${CSR4MPI_TEST_DATA_DIR}")
endif()