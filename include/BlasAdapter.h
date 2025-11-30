#pragma once
#include "CSRMatrix.h"
#include "Global.h"
#include <vector>

namespace csr4mpi {
bool bBlasEnabled();
void SpMMBlas(const cCSRMatrix& A, const std::vector<vScalar>& X, iSize nCols, std::vector<vScalar>& Y);
}
