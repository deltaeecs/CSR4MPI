#pragma once

#include "CSRMatrix.h"
#include "Global.h"
#include <algorithm>
#include <unordered_map>
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

    // Build the CSR matrix with automatic deduplication
    // iGlobalRowBegin: starting global row index (typically 0 for a standalone local matrix)
    // iGlobalRowEnd: ending global row index (exclusive)
    // iGlobalColCount: total number of columns
    // bAccumulateDuplicates: if true, accumulate duplicate entries; if false, keep last value
    cCSRMatrix<Scalar> Build(iSize iGlobalRowBegin, iSize iGlobalRowEnd, iSize iGlobalColCount, bool bAccumulateDuplicates = true) const
    {
        struct cKey {
            iIndex r;
            iIndex c;
            bool operator==(const cKey& other) const { return r == other.r && c == other.c; }
        };
        struct cKeyHash {
            std::size_t operator()(const cKey& k) const noexcept
            {
                // Use a robust hash combining function to reduce collisions
                // Based on boost::hash_combine algorithm
                std::size_t seed = std::hash<iIndex>()(k.r);
                seed ^= std::hash<iIndex>()(k.c) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        iSize iLocalRows = iGlobalRowEnd - iGlobalRowBegin;
        
        // Deduplicate using hash map
        std::unordered_map<cKey, Scalar, cKeyHash> mEntries;
        mEntries.reserve(m_vTriplets.size());

        for (const auto& triplet : m_vTriplets) {
            // Validate row is within range
            if (triplet.m_iRow < iGlobalRowBegin || triplet.m_iRow >= iGlobalRowEnd) {
                continue; // Skip out-of-range entries
            }
            
            cKey key { triplet.m_iRow, triplet.m_iCol };
            auto it = mEntries.find(key);
            if (it == mEntries.end()) {
                mEntries.emplace(key, triplet.m_vValue);
            } else {
                if (bAccumulateDuplicates) {
                    it->second += triplet.m_vValue;
                } else {
                    it->second = triplet.m_vValue; // Keep last value
                }
            }
        }

        // Convert to sorted triplets
        struct cEntry {
            iIndex r;
            iIndex c;
            Scalar v;
        };
        std::vector<cEntry> vSortedEntries;
        vSortedEntries.reserve(mEntries.size());
        
        for (const auto& p : mEntries) {
            vSortedEntries.push_back({ p.first.r, p.first.c, p.second });
        }

        // Sort by row then column
        std::sort(vSortedEntries.begin(), vSortedEntries.end(), [](const cEntry& a, const cEntry& b) {
            if (a.r != b.r)
                return a.r < b.r;
            return a.c < b.c;
        });

        // Build CSR arrays
        std::vector<iIndex> vRowPtr(static_cast<std::size_t>(iLocalRows + 1), 0);
        std::vector<iIndex> vColInd;
        std::vector<Scalar> vValues;
        
        vColInd.reserve(vSortedEntries.size());
        vValues.reserve(vSortedEntries.size());

        for (const auto& e : vSortedEntries) {
            iIndex iLocalRow = e.r - iGlobalRowBegin;
            vRowPtr[static_cast<std::size_t>(iLocalRow + 1)]++;
            vColInd.push_back(e.c);
            vValues.push_back(e.v);
        }

        // Prefix sum to get row pointers
        for (iSize r = 0; r < iLocalRows; ++r) {
            vRowPtr[static_cast<std::size_t>(r + 1)] += vRowPtr[static_cast<std::size_t>(r)];
        }

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
