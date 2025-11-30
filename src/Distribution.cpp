#include "Distribution.h"

namespace csr4mpi {
cRowDistribution::cRowDistribution()
    : m_iGlobalRowCount(0)
    , m_iGlobalRowBegin(0)
    , m_iGlobalRowEnd(0)
{
}

cRowDistribution::cRowDistribution(iSize iGlobalRowCount,
    const std::vector<iSize>& vLocalRowCounts,
    int iRank)
    : m_iGlobalRowCount(iGlobalRowCount)
    , m_iGlobalRowBegin(0)
    , m_iGlobalRowEnd(0)
    , m_vRowOffsets(vLocalRowCounts.size() + 1, 0)
{
    for (std::size_t i = 0; i < vLocalRowCounts.size(); ++i) {
        m_vRowOffsets[i + 1] = m_vRowOffsets[i] + vLocalRowCounts[i];
    }

    m_iGlobalRowBegin = m_vRowOffsets[static_cast<std::size_t>(iRank)];
    m_iGlobalRowEnd = m_vRowOffsets[static_cast<std::size_t>(iRank) + 1];
}

cRowDistribution cRowDistribution::CreateBlockDistribution(iSize iGlobalRowCount,
    int iWorldSize,
    int iRank)
{
    std::vector<iSize> vLocalRowCounts;
    vLocalRowCounts.resize(static_cast<std::size_t>(iWorldSize));

    iSize iBase = iGlobalRowCount / static_cast<iSize>(iWorldSize);
    iSize iRemainder = iGlobalRowCount % static_cast<iSize>(iWorldSize);

    for (int i = 0; i < iWorldSize; ++i) {
        vLocalRowCounts[static_cast<std::size_t>(i)] = iBase + (i < iRemainder ? 1 : 0);
    }

    return cRowDistribution(iGlobalRowCount, vLocalRowCounts, iRank);
}

iSize cRowDistribution::iGlobalRowBegin() const
{
    return m_iGlobalRowBegin;
}

iSize cRowDistribution::iGlobalRowEnd() const
{
    return m_iGlobalRowEnd;
}

iSize cRowDistribution::iGlobalRowCount() const
{
    return m_iGlobalRowCount;
}

int cRowDistribution::iOwnerRank(iIndex iGlobalRow) const
{
    // Binary search over row offsets.
    int iLeft = 0;
    int iRight = static_cast<int>(m_vRowOffsets.size()) - 1;

    while (iLeft < iRight) {
        int iMid = iLeft + (iRight - iLeft) / 2;
        if (iGlobalRow < static_cast<iIndex>(m_vRowOffsets[static_cast<std::size_t>(iMid)])) {
            iRight = iMid;
        } else if (iGlobalRow >= static_cast<iIndex>(m_vRowOffsets[static_cast<std::size_t>(iMid + 1)])) {
            iLeft = iMid + 1;
        } else {
            return iMid;
        }
    }

    return iLeft;
}

const std::vector<iSize>& cRowDistribution::vRowOffsets() const
{
    return m_vRowOffsets;
}
}
