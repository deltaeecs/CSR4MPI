#include "CommPattern.h"
#include "Distribution.h"
#include <unordered_map>

namespace csr4mpi {
cCommPattern::cCommPattern() = default;

void cCommPattern::Build(const std::vector<cRemoteEntry>& vEntries,
    const cRowDistribution& cDistribution,
    int iRank,
    int iWorldSize)
{
    std::unordered_map<int, std::vector<cRemoteEntry>> mBuckets;
    mBuckets.reserve(static_cast<std::size_t>(iWorldSize));

    for (const cRemoteEntry& cEntry : vEntries) {
        int iOwner = cDistribution.iOwnerRank(cEntry.m_iGlobalRow);
        // Include local and remote entries; local entries will be handled without MPI.
        mBuckets[iOwner].push_back(cEntry);
    }

    m_vTargetRanks.clear();
    m_vSendRows.clear();
    m_vSendCols.clear();
    m_vSendOffsets.clear();

    m_vSourceRanks.clear();
    m_vRecvRows.clear();
    m_vRecvCols.clear();
    m_vRecvOffsets.clear();

    m_vSendOffsets.push_back(0);

    for (const auto& cPair : mBuckets) {
        int iTarget = cPair.first;
        const std::vector<cRemoteEntry>& vBucket = cPair.second;

        m_vTargetRanks.push_back(iTarget);

        for (const cRemoteEntry& cEntry : vBucket) {
            m_vSendRows.push_back(cEntry.m_iGlobalRow);
            m_vSendCols.push_back(cEntry.m_iGlobalCol);
        }

        m_vSendOffsets.push_back(static_cast<iSize>(m_vSendRows.size()));
    }

    // Receive side will be filled during MPI setup inside communication layer.
    // Here we only ensure empty but valid prefix array.
    m_vRecvOffsets.push_back(0);
}

const std::vector<int>& cCommPattern::vTargetRanks() const
{
    return m_vTargetRanks;
}

const std::vector<iIndex>& cCommPattern::vSendRows() const
{
    return m_vSendRows;
}

const std::vector<iIndex>& cCommPattern::vSendCols() const
{
    return m_vSendCols;
}

const std::vector<iSize>& cCommPattern::vSendOffsets() const
{
    return m_vSendOffsets;
}

const std::vector<iIndex>& cCommPattern::vRecvRows() const
{
    return m_vRecvRows;
}

const std::vector<iIndex>& cCommPattern::vRecvCols() const
{
    return m_vRecvCols;
}

const std::vector<iSize>& cCommPattern::vRecvOffsets() const
{
    return m_vRecvOffsets;
}
}
