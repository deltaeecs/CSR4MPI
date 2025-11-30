#pragma once

#include <cstdint>

namespace csr4mpi {
enum eValueType {
    eFloat32Real,
    eFloat64Real,
    eFloat32Complex,
    eFloat64Complex
};
enum eSymmStorage {
    eNone,
    eSymmLower,
    eSymmUpper,
    eSymmFull
};
}

#ifndef CSR4MPI_VALUE_TYPE
#define CSR4MPI_VALUE_TYPE 1
#endif

#include <complex>

namespace csr4mpi {
#if CSR4MPI_VALUE_TYPE == 0
using vScalar = float;
#elif CSR4MPI_VALUE_TYPE == 1
using vScalar = double;
#elif CSR4MPI_VALUE_TYPE == 2
using vScalar = std::complex<float>;
#elif CSR4MPI_VALUE_TYPE == 3
using vScalar = std::complex<double>;
#else
#error "Unsupported CSR4MPI_VALUE_TYPE"
#endif

using iIndex = std::int64_t;
using iSize = std::int64_t;
}
