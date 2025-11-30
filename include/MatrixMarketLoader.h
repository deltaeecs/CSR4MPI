#pragma once
#include "Global.h"
#include <string>
#include <vector>

namespace csr4mpi {
bool LoadMatrixMarket(const std::string& sPath,
    std::vector<iIndex>& vRowPtr,
    std::vector<iIndex>& vColInd,
    std::vector<vScalar>& vValues,
    iIndex& iRows,
    iIndex& iCols);
}
