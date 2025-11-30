#include "MumpsAdapter.h"
#include "CSRMatrix.h"
#include "Distribution.h"

namespace csr4mpi {
void cMumpsAdapter::ExportLocalBlock(const cCSRMatrix& cLocal,
    const cRowDistribution& cDistribution,
    std::vector<int>& vIRN,
    std::vector<int>& vJCN,
    std::vector<vScalar>& vA)
{
    const std::vector<iIndex>& vRowPtr = cLocal.vRowPtr();
    const std::vector<iIndex>& vColInd = cLocal.vColInd();
    const std::vector<vScalar>& vValues = cLocal.vValues();

    const iSize iRowBegin = cDistribution.iGlobalRowBegin();
    const iSize iRowEnd = cDistribution.iGlobalRowEnd();
    const iSize iLocalRows = iRowEnd - iRowBegin;
    const iSize iLocalNNZ = static_cast<iSize>(vRowPtr[static_cast<std::size_t>(iLocalRows)]);

    // Minimize reallocations: reserve exact additional capacity for this append.
    vIRN.reserve(vIRN.size() + static_cast<std::size_t>(iLocalNNZ));
    vJCN.reserve(vJCN.size() + static_cast<std::size_t>(iLocalNNZ));
    vA.reserve(vA.size() + static_cast<std::size_t>(iLocalNNZ));

    for (iSize iGlobalRow = iRowBegin; iGlobalRow < iRowEnd; ++iGlobalRow) {
        const iIndex iLocalRow = static_cast<iIndex>(iGlobalRow - iRowBegin);
        const iIndex iStart = vRowPtr[static_cast<std::size_t>(iLocalRow)];
        const iIndex iEnd = vRowPtr[static_cast<std::size_t>(iLocalRow + 1)];

        for (iIndex i = iStart; i < iEnd; ++i) {
            const iIndex iCol = vColInd[static_cast<std::size_t>(i)];
            vIRN.push_back(static_cast<int>(iGlobalRow + 1));
            vJCN.push_back(static_cast<int>(iCol + 1));
            vA.push_back(vValues[static_cast<std::size_t>(i)]);
        }
    }
}

iSize cMumpsAdapter::ExportLocalBlockInto(const cCSRMatrix& cLocal,
    const cRowDistribution& cDistribution,
    int* pIRN,
    int* pJCN,
    vScalar* pA)
{
    const std::vector<iIndex>& vRowPtr = cLocal.vRowPtr();
    const std::vector<iIndex>& vColInd = cLocal.vColInd();
    const std::vector<vScalar>& vValues = cLocal.vValues();

    const iSize iRowBegin = cDistribution.iGlobalRowBegin();
    const iSize iRowEnd = cDistribution.iGlobalRowEnd();
    const iSize iLocalRows = iRowEnd - iRowBegin;
    const iSize iLocalNNZ = static_cast<iSize>(vRowPtr[static_cast<std::size_t>(iLocalRows)]);

    iSize k = 0;
    for (iSize iGlobalRow = iRowBegin; iGlobalRow < iRowEnd; ++iGlobalRow) {
        const iIndex iLocalRow = static_cast<iIndex>(iGlobalRow - iRowBegin);
        const iIndex iStart = vRowPtr[static_cast<std::size_t>(iLocalRow)];
        const iIndex iEnd = vRowPtr[static_cast<std::size_t>(iLocalRow + 1)];
        for (iIndex i = iStart; i < iEnd; ++i, ++k) {
            const iIndex iCol = vColInd[static_cast<std::size_t>(i)];
            pIRN[k] = static_cast<int>(iGlobalRow + 1);
            pJCN[k] = static_cast<int>(iCol + 1);
            pA[k] = vValues[static_cast<std::size_t>(i)];
        }
    }

    return iLocalNNZ;
}
}
