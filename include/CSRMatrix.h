#pragma once

#include "Global.h"
#include <memory>
#include <vector>

namespace csr4mpi {
class cRowDistribution;

template <typename Scalar>
class cCSRMatrix {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

public:
    using scalar_type = Scalar;

    cCSRMatrix()
        : m_iGlobalRowBegin(0)
        , m_iGlobalRowEnd(0)
        , m_iGlobalColCount(0)
        , m_pDistribution(nullptr)
    {
    }

    cCSRMatrix(iSize iGlobalRowBegin,
        iSize iGlobalRowEnd,
        iSize iGlobalColCount,
        std::vector<iIndex> vRowPtr,
        std::vector<iIndex> vColInd,
        std::vector<Scalar> vValues,
        eSymmStorage eSymm = eNone)
        : m_iGlobalRowBegin(iGlobalRowBegin)
        , m_iGlobalRowEnd(iGlobalRowEnd)
        , m_iGlobalColCount(iGlobalColCount)
        , m_vRowPtr(std::move(vRowPtr))
        , m_vColInd(std::move(vColInd))
        , m_vValues(std::move(vValues))
        , m_pDistribution(nullptr)
        , m_eSymm(eSymm)
    {
    }

    iSize iGlobalRowBegin() const { return m_iGlobalRowBegin; }
    iSize iGlobalRowEnd() const { return m_iGlobalRowEnd; }

    const std::vector<iIndex>& vRowPtr() const { return m_vRowPtr; }
    const std::vector<iIndex>& vColInd() const { return m_vColInd; }
    const std::vector<Scalar>& vValues() const { return m_vValues; }
    std::vector<Scalar>& vValues() { return m_vValues; }
    iSize iGlobalColCount() const { return m_iGlobalColCount; }
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
    std::vector<Scalar> m_vValues;
    std::shared_ptr<cRowDistribution> m_pDistribution;
    eSymmStorage m_eSymm { eNone };
};

// Type aliases for common scalar types
using cCSRMatrixF = cCSRMatrix<float>;
using cCSRMatrixD = cCSRMatrix<double>;
using cCSRMatrixCF = cCSRMatrix<std::complex<float>>;
using cCSRMatrixCD = cCSRMatrix<std::complex<double>>;

}
