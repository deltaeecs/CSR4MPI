#pragma once

#include "CSRMatrix.h"
#include <vector>

namespace csr4mpi {
void SpMV(const cCSRMatrix& A, const std::vector<vScalar>& x, std::vector<vScalar>& y);
void SpMV(const cCSRMatrix& A, std::vector<vScalar>& x);
void SpMM(const cCSRMatrix& A, const std::vector<vScalar>& X, iSize nCols, std::vector<vScalar>& Y);
void SpMMInPlace(const cCSRMatrix& A, std::vector<vScalar>& X, iSize nCols);
}
