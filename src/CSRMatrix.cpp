#include "CSRMatrix.h"

namespace csr4mpi {
cCSRMatrix::cCSRMatrix()
    : m_iGlobalRowBegin(0)
    , m_iGlobalRowEnd(0)
    , m_iGlobalColCount(0)
    , m_pDistribution(nullptr)
{
}

cCSRMatrix::cCSRMatrix(iSize iGlobalRowBegin,
    iSize iGlobalRowEnd,
    iSize iGlobalColCount,
    std::vector<iIndex> vRowPtr,
    std::vector<iIndex> vColInd,
    std::vector<vScalar> vValues,
    eSymmStorage eSymm)
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

iSize cCSRMatrix::iGlobalRowBegin() const
{
    return m_iGlobalRowBegin;
}

iSize cCSRMatrix::iGlobalRowEnd() const
{
    return m_iGlobalRowEnd;
}

const std::vector<iIndex>& cCSRMatrix::vRowPtr() const
{
    return m_vRowPtr;
}

const std::vector<iIndex>& cCSRMatrix::vColInd() const
{
    return m_vColInd;
}

const std::vector<vScalar>& cCSRMatrix::vValues() const
{
    return m_vValues;
}

iSize cCSRMatrix::iGlobalColCount() const
{
    return m_iGlobalColCount;
}
}
