#pragma once

#include "Global.h"
#include <vector>

namespace csr4mpi {
class cRowDistribution {
public:
    cRowDistribution();

    cRowDistribution(iSize iGlobalRowCount,
        const std::vector<iSize>& vLocalRowCounts,
        int iRank);

    static cRowDistribution CreateBlockDistribution(iSize iGlobalRowCount,
        int iWorldSize,
        int iRank);

    iSize iGlobalRowBegin() const;
    iSize iGlobalRowEnd() const;
    iSize iGlobalRowCount() const;

    int iOwnerRank(iIndex iGlobalRow) const;

    const std::vector<iSize>& vRowOffsets() const;

private:
    iSize m_iGlobalRowCount;
    iSize m_iGlobalRowBegin;
    iSize m_iGlobalRowEnd;

    std::vector<iSize> m_vRowOffsets;
};
}
