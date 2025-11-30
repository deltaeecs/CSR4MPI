#include "BlasAdapter.h"
#include "Operations.h"
#include <iostream>

namespace csr4mpi {

bool bBlasEnabled()
{
#ifdef CSR4MPI_USE_BLAS
    return true;
#else
    return false;
#endif
}

void SpMMBlas(const cCSRMatrix& A, const std::vector<vScalar>& X, iSize nCols, std::vector<vScalar>& Y)
{
#ifdef CSR4MPI_USE_BLAS
    // Placeholder: in future, detect dense conversion threshold and call cblas_Xgemm.
    // For now, just log once per call and fallback.
    // Avoid performance penalty of i/o in real scenarios; kept minimal here.
    // (Could be guarded by an environment variable.)
    // std::cerr used intentionally to separate from normal output.
    std::cerr << "[BLAS placeholder] Falling back to internal SpMM path.\n";
#endif
    SpMM(A, X, nCols, Y);
}

}
