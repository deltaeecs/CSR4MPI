#pragma once

#include "Global.h"
#include <memory>
#include <vector>

namespace csr4mpi {
class cRowDistribution;
}

namespace csr4mpi {
class cCSRMatrix {
public:
    cCSRMatrix();

    cCSRMatrix(iSize iGlobalRowBegin,
        iSize iGlobalRowEnd,
        iSize iGlobalColCount,
        std::vector<iIndex> vRowPtr,
        std::vector<iIndex> vColInd,
        std::vector<vScalar> vValues,
        eSymmStorage eSymm = eNone);

    iSize iGlobalRowBegin() const;
    iSize iGlobalRowEnd() const;

    const std::vector<iIndex>& vRowPtr() const;
    const std::vector<iIndex>& vColInd() const;
    const std::vector<vScalar>& vValues() const;
    iSize iGlobalColCount() const;
    eSymmStorage eSymmetry() const { return m_eSymm; }
    bool bIsSymmetric() const { return m_eSymm != eNone; }

    const std::shared_ptr<cRowDistribution>& pDistribution() const { return m_pDistribution; }
    void AttachDistribution(std::shared_ptr<cRowDistribution> pDist) { m_pDistribution = std::move(pDist); }

private:
    iSize m_iGlobalRowBegin;
    iSize m_iGlobalRowEnd;
    iSize m_iGlobalColCount;

    std::vector<iIndex> m_vRowPtr;
    std::vector<iIndex> m_vColInd;
    std::vector<vScalar> m_vValues;
    std::shared_ptr<cRowDistribution> m_pDistribution;
    eSymmStorage m_eSymm { eNone };
};
}
