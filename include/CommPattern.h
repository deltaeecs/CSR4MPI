#pragma once

#include "Global.h"
#include <vector>

namespace csr4mpi {
struct cRemoteEntry {
    iIndex m_iGlobalRow;
    iIndex m_iGlobalCol;
    vScalar m_vValue;
};

class cCommPattern {
public:
    cCommPattern();

    void Build(const std::vector<cRemoteEntry>& vEntries,
        const class cRowDistribution& cDistribution,
        int iRank,
        int iWorldSize);

    const std::vector<int>& vTargetRanks() const;
    const std::vector<iIndex>& vSendRows() const;
    const std::vector<iIndex>& vSendCols() const;
    const std::vector<iSize>& vSendOffsets() const;

    const std::vector<iIndex>& vRecvRows() const;
    const std::vector<iIndex>& vRecvCols() const;
    const std::vector<iSize>& vRecvOffsets() const;

private:
    std::vector<int> m_vTargetRanks;
    std::vector<iIndex> m_vSendRows;
    std::vector<iIndex> m_vSendCols;
    std::vector<iSize> m_vSendOffsets;

    std::vector<int> m_vSourceRanks;
    std::vector<iIndex> m_vRecvRows;
    std::vector<iIndex> m_vRecvCols;
    std::vector<iSize> m_vRecvOffsets;
};
}
