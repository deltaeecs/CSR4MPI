#pragma once

#include "CSRMatrix.h"
#include "Global.h"
#include <algorithm>
#include <vector>

namespace csr4mpi {

template <typename Scalar>
struct cTriplet {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");
    iIndex m_iRow;
    iIndex m_iCol;
    Scalar m_vValue;
    
    cTriplet() : m_iRow(0), m_iCol(0), m_vValue(static_cast<Scalar>(0)) {}
    cTriplet(iIndex iRow, iIndex iCol, Scalar vValue)
        : m_iRow(iRow), m_iCol(iCol), m_vValue(vValue) {}
};

// Type aliases for common scalar types
using cTripletF = cTriplet<float>;
using cTripletD = cTriplet<double>;
using cTripletCF = cTriplet<std::complex<float>>;
using cTripletCD = cTriplet<std::complex<double>>;

template <typename Scalar>
class cCSRMatrixBuilder {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

public:
    using scalar_type = Scalar;

    cCSRMatrixBuilder() = default;

    // Add a single triplet entry
    void AddEntry(iIndex iRow, iIndex iCol, Scalar vValue)
    {
        m_vTriplets.push_back(cTriplet<Scalar>(iRow, iCol, vValue));
    }

    // Add multiple triplet entries
    void AddEntries(const std::vector<cTriplet<Scalar>>& vTriplets)
    {
        m_vTriplets.insert(m_vTriplets.end(), vTriplets.begin(), vTriplets.end());
    }

    // Reserve space for efficiency
    void Reserve(std::size_t size)
    {
        m_vTriplets.reserve(size);
    }

    // Clear all entries
    void Clear()
    {
        m_vTriplets.clear();
    }

    // Build the CSR matrix with automatic deduplication.
    //
    // Sort-based pipeline (issue #7): the previous implementation deduplicated
    // through a global std::unordered_map, which allocated one heap node per
    // unique entry (~48-64B), hashed every triplet, and ended with a global
    // O(n log n) sort — for near-field assemblies with tens of millions of
    // triplets this dominated both build time and peak memory. The pipeline
    // below replaces it with:
    //
    //   Pass 1  count in-range entries per row -> rowPtr prefix sum   O(N)
    //   Pass 2  scatter (col, value) into row-segmented slots         O(N)
    //   Pass 3  per-row std::stable_sort by column + merge of equal-
    //           column runs (accumulate or keep-last)                 O(sum(nnz_r log nnz_r))
    //   Pass 4  final row offsets while emitting the CSR arrays
    //
    // All passes stream sequentially over memory; no hash nodes, no rehash,
    // no global sort. std::stable_sort preserves insertion order within equal
    // columns, so duplicates accumulate in the same order (and to the same
    // bits) as the previous hash-map implementation.
    //
    // iGlobalRowBegin: starting global row index (typically 0 for a standalone local matrix)
    // iGlobalRowEnd: ending global row index (exclusive)
    // iGlobalColCount: total number of columns
    // bAccumulateDuplicates: if true, accumulate duplicate entries; if false, keep last value
    cCSRMatrix<Scalar> Build(iSize iGlobalRowBegin, iSize iGlobalRowEnd, iSize iGlobalColCount, bool bAccumulateDuplicates = true) const
    {
        const iSize iLocalRows = iGlobalRowEnd - iGlobalRowBegin;

        // Pass 1: count in-range entries per row (duplicates included)
        std::vector<iIndex> vRowPtr(static_cast<std::size_t>(iLocalRows + 1), 0);
        for (const auto& triplet : m_vTriplets) {
            // Validate row is within range
            if (triplet.m_iRow < iGlobalRowBegin || triplet.m_iRow >= iGlobalRowEnd) {
                continue; // Skip out-of-range entries
            }
            ++vRowPtr[static_cast<std::size_t>(triplet.m_iRow - iGlobalRowBegin + 1)];
        }

        // Prefix sum -> segment bounds per row
        for (iSize r = 0; r < iLocalRows; ++r) {
            vRowPtr[static_cast<std::size_t>(r + 1)] += vRowPtr[static_cast<std::size_t>(r)];
        }

        const std::size_t nTotal = static_cast<std::size_t>(vRowPtr.back());

        // Pass 2: scatter (col, value) pairs into their row segment,
        // preserving insertion order within each row
        struct cKV {
            iIndex c;
            Scalar v;
        };
        std::vector<cKV> vTmp(nTotal);
        std::vector<iIndex> vCursor(vRowPtr.begin(), vRowPtr.end() - 1);
        for (const auto& triplet : m_vTriplets) {
            if (triplet.m_iRow < iGlobalRowBegin || triplet.m_iRow >= iGlobalRowEnd) {
                continue;
            }
            const std::size_t r = static_cast<std::size_t>(triplet.m_iRow - iGlobalRowBegin);
            vTmp[static_cast<std::size_t>(vCursor[r]++)] = cKV { triplet.m_iCol, triplet.m_vValue };
        }

        // Pass 3 + 4: per-row stable sort by column, merge equal-column runs,
        // and emit the compacted CSR arrays. Slot r of vRowPtr is overwritten
        // with the final offset after its segment [vRowPtr[r], vRowPtr[r+1])
        // has been consumed, so the old prefix sums stay readable while the
        // final ones are written in place.
        std::vector<iIndex> vColInd;
        std::vector<Scalar> vValues;
        vColInd.reserve(nTotal);
        vValues.reserve(nTotal);

        for (iSize r = 0; r < iLocalRows; ++r) {
            const std::size_t s = static_cast<std::size_t>(vRowPtr[static_cast<std::size_t>(r)]);
            const std::size_t e = static_cast<std::size_t>(vRowPtr[static_cast<std::size_t>(r + 1)]);

            std::stable_sort(vTmp.begin() + s, vTmp.begin() + e,
                [](const cKV& a, const cKV& b) { return a.c < b.c; });

            vRowPtr[static_cast<std::size_t>(r)] = static_cast<iIndex>(vColInd.size());

            std::size_t k = s;
            while (k < e) {
                std::size_t m = k + 1;
                while (m < e && vTmp[m].c == vTmp[k].c) {
                    ++m;
                }
                Scalar vAgg = vTmp[k].v;
                for (std::size_t q = k + 1; q < m; ++q) {
                    if (bAccumulateDuplicates) {
                        vAgg = vAgg + vTmp[q].v;
                    } else {
                        vAgg = vTmp[q].v; // Keep last value (stable order -> last inserted)
                    }
                }
                vColInd.push_back(vTmp[k].c);
                vValues.push_back(vAgg);
                k = m;
            }
        }
        vRowPtr[static_cast<std::size_t>(iLocalRows)] = static_cast<iIndex>(vColInd.size());

        return cCSRMatrix<Scalar>(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount,
                                  std::move(vRowPtr), std::move(vColInd), std::move(vValues));
    }

private:
    std::vector<cTriplet<Scalar>> m_vTriplets;
};

// Type aliases for common scalar types
using cCSRMatrixBuilderF = cCSRMatrixBuilder<float>;
using cCSRMatrixBuilderD = cCSRMatrixBuilder<double>;
using cCSRMatrixBuilderCF = cCSRMatrixBuilder<std::complex<float>>;
using cCSRMatrixBuilderCD = cCSRMatrixBuilder<std::complex<double>>;

// Convenience function to build CSR matrix directly from triplets
template <typename Scalar>
cCSRMatrix<Scalar> BuildCSRFromTriplets(
    const std::vector<cTriplet<Scalar>>& vTriplets,
    iSize iGlobalRowBegin,
    iSize iGlobalRowEnd,
    iSize iGlobalColCount,
    bool bAccumulateDuplicates = true)
{
    cCSRMatrixBuilder<Scalar> builder;
    builder.AddEntries(vTriplets);
    return builder.Build(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, bAccumulateDuplicates);
}

// Validation function to check if CSR matrix structure is valid
template <typename Scalar>
bool ValidateCSRMatrix(const cCSRMatrix<Scalar>& matrix, std::string* pErrorMsg = nullptr)
{
    const auto& rowPtr = matrix.vRowPtr();
    const auto& colInd = matrix.vColInd();
    const auto& values = matrix.vValues();
    
    iSize localRows = matrix.iGlobalRowEnd() - matrix.iGlobalRowBegin();
    
    // Check row pointer size
    if (rowPtr.size() != static_cast<std::size_t>(localRows + 1)) {
        if (pErrorMsg) *pErrorMsg = "Row pointer size mismatch";
        return false;
    }
    
    // Check row pointers are non-decreasing
    for (iSize i = 0; i < localRows; ++i) {
        if (rowPtr[static_cast<std::size_t>(i)] > rowPtr[static_cast<std::size_t>(i + 1)]) {
            if (pErrorMsg) *pErrorMsg = "Row pointers are not non-decreasing";
            return false;
        }
    }
    
    // Check first row pointer is 0
    if (rowPtr[0] != 0) {
        if (pErrorMsg) *pErrorMsg = "First row pointer is not 0";
        return false;
    }
    
    // Check last row pointer matches number of non-zeros
    if (rowPtr[static_cast<std::size_t>(localRows)] != static_cast<iIndex>(colInd.size())) {
        if (pErrorMsg) *pErrorMsg = "Last row pointer does not match column index size";
        return false;
    }
    
    // Check column indices and values have same size
    if (colInd.size() != values.size()) {
        if (pErrorMsg) *pErrorMsg = "Column index and value array size mismatch";
        return false;
    }
    
    // Check column indices are sorted within each row and within bounds
    for (iSize r = 0; r < localRows; ++r) {
        iIndex start = rowPtr[static_cast<std::size_t>(r)];
        iIndex end = rowPtr[static_cast<std::size_t>(r + 1)];
        
        for (iIndex k = start; k < end; ++k) {
            iIndex col = colInd[static_cast<std::size_t>(k)];
            
            // Check column is within bounds
            if (col < 0 || col >= matrix.iGlobalColCount()) {
                if (pErrorMsg) *pErrorMsg = "Column index out of bounds";
                return false;
            }
            
            // Check columns are sorted within row
            if (k > start && colInd[static_cast<std::size_t>(k - 1)] >= col) {
                if (pErrorMsg) *pErrorMsg = "Column indices are not sorted within row";
                return false;
            }
        }
    }
    
    return true;
}

}
