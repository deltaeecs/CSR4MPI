#pragma once

#include "Global.h"
#include "Distribution.h"
#include <vector>
#include <unordered_map>

namespace csr4mpi {

template <typename Scalar>
struct cRemoteEntry {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");
    iIndex m_iGlobalRow;
    iIndex m_iGlobalCol;
    Scalar m_vValue;
};

// Type aliases for common scalar types
using cRemoteEntryF = cRemoteEntry<float>;
using cRemoteEntryD = cRemoteEntry<double>;
using cRemoteEntryCF = cRemoteEntry<std::complex<float>>;
using cRemoteEntryCD = cRemoteEntry<std::complex<double>>;

template <typename Scalar>
class cCommPattern {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

public:
    using scalar_type = Scalar;

    cCommPattern() = default;

    void Build(const std::vector<cRemoteEntry<Scalar>>& vEntries,
        const cRowDistribution& cDistribution,
        int iRank,
        int iWorldSize)
    {
        std::unordered_map<int, std::vector<cRemoteEntry<Scalar>>> mBuckets;
        mBuckets.reserve(static_cast<std::size_t>(iWorldSize));

        for (const cRemoteEntry<Scalar>& cEntry : vEntries) {
            int iOwner = cDistribution.iOwnerRank(cEntry.m_iGlobalRow);
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
            const std::vector<cRemoteEntry<Scalar>>& vBucket = cPair.second;

            m_vTargetRanks.push_back(iTarget);

            for (const cRemoteEntry<Scalar>& cEntry : vBucket) {
                m_vSendRows.push_back(cEntry.m_iGlobalRow);
                m_vSendCols.push_back(cEntry.m_iGlobalCol);
            }

            m_vSendOffsets.push_back(static_cast<iSize>(m_vSendRows.size()));
        }

        m_vRecvOffsets.push_back(0);
    }

    const std::vector<int>& vTargetRanks() const { return m_vTargetRanks; }
    const std::vector<iIndex>& vSendRows() const { return m_vSendRows; }
    const std::vector<iIndex>& vSendCols() const { return m_vSendCols; }
    const std::vector<iSize>& vSendOffsets() const { return m_vSendOffsets; }

    const std::vector<iIndex>& vRecvRows() const { return m_vRecvRows; }
    const std::vector<iIndex>& vRecvCols() const { return m_vRecvCols; }
    const std::vector<iSize>& vRecvOffsets() const { return m_vRecvOffsets; }

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

// Type aliases for common scalar types
using cCommPatternF = cCommPattern<float>;
using cCommPatternD = cCommPattern<double>;
using cCommPatternCF = cCommPattern<std::complex<float>>;
using cCommPatternCD = cCommPattern<std::complex<double>>;

}
