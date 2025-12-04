#pragma once
#include "CSRMatrix.h"
#include "Operations.h"
#include "Global.h"
#include <vector>
#include <iostream>

namespace csr4mpi {

inline bool bBlasEnabled()
{
#ifdef CSR4MPI_USE_BLAS
    return true;
#else
    return false;
#endif
}

template <typename Scalar>
void SpMMBlas(const cCSRMatrix<Scalar>& A, const std::vector<Scalar>& X, iSize nCols, std::vector<Scalar>& Y)
{
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");
#ifdef CSR4MPI_USE_BLAS
    // Placeholder: in future, detect dense conversion threshold and call cblas_Xgemm.
    // For now, just log once per call and fallback.
    std::cerr << "[BLAS placeholder] Falling back to internal SpMM path.\n";
#endif
    SpMM(A, X, nCols, Y);
}

}
