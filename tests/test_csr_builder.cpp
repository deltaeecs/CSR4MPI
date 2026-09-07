#include "CSRMatrixBuilder.h"
#include "CSRMatrix.h"
#include "Operations.h"
#include <gtest/gtest.h>
#include <complex>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

using namespace csr4mpi;

// Test with double as the default scalar type
using Scalar = double;

TEST(CSRBuilderTest, BasicConstruction)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Build a simple 3x3 matrix:
    // [1  2  0]
    // [0  3  4]
    // [5  0  6]
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(0, 1, 2.0);
    builder.AddEntry(1, 1, 3.0);
    builder.AddEntry(1, 2, 4.0);
    builder.AddEntry(2, 0, 5.0);
    builder.AddEntry(2, 2, 6.0);
    
    auto matrix = builder.Build(0, 3, 3);
    
    EXPECT_EQ(matrix.iGlobalRowBegin(), 0);
    EXPECT_EQ(matrix.iGlobalRowEnd(), 3);
    EXPECT_EQ(matrix.iGlobalColCount(), 3);
    EXPECT_EQ(matrix.vRowPtr().size(), 4u);
    EXPECT_EQ(matrix.vColInd().size(), 6u);
    EXPECT_EQ(matrix.vValues().size(), 6u);
    
    // Verify structure
    const auto& rowPtr = matrix.vRowPtr();
    const auto& colInd = matrix.vColInd();
    const auto& values = matrix.vValues();
    
    EXPECT_EQ(rowPtr[0], 0);
    EXPECT_EQ(rowPtr[1], 2);
    EXPECT_EQ(rowPtr[2], 4);
    EXPECT_EQ(rowPtr[3], 6);
    
    // Row 0: cols 0,1
    EXPECT_EQ(colInd[0], 0);
    EXPECT_EQ(colInd[1], 1);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 2.0);
    
    // Row 1: cols 1,2
    EXPECT_EQ(colInd[2], 1);
    EXPECT_EQ(colInd[3], 2);
    EXPECT_DOUBLE_EQ(values[2], 3.0);
    EXPECT_DOUBLE_EQ(values[3], 4.0);
    
    // Row 2: cols 0,2
    EXPECT_EQ(colInd[4], 0);
    EXPECT_EQ(colInd[5], 2);
    EXPECT_DOUBLE_EQ(values[4], 5.0);
    EXPECT_DOUBLE_EQ(values[5], 6.0);
}

TEST(CSRBuilderTest, DuplicateAccumulation)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Add duplicates that should accumulate
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(0, 0, 2.0);
    builder.AddEntry(0, 0, 3.0);
    builder.AddEntry(1, 1, 5.0);
    builder.AddEntry(1, 1, 7.0);
    
    auto matrix = builder.Build(0, 2, 2, true); // Accumulate duplicates
    
    const auto& rowPtr = matrix.vRowPtr();
    const auto& colInd = matrix.vColInd();
    const auto& values = matrix.vValues();
    
    EXPECT_EQ(colInd.size(), 2u);
    EXPECT_EQ(values.size(), 2u);
    
    // Row 0: (0,0) should be 1+2+3=6
    EXPECT_EQ(colInd[0], 0);
    EXPECT_DOUBLE_EQ(values[0], 6.0);
    
    // Row 1: (1,1) should be 5+7=12
    EXPECT_EQ(colInd[1], 1);
    EXPECT_DOUBLE_EQ(values[1], 12.0);
}

TEST(CSRBuilderTest, DuplicateLastValue)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Add duplicates, but keep only the last value
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(0, 0, 2.0);
    builder.AddEntry(0, 0, 3.0);
    
    auto matrix = builder.Build(0, 1, 1, false); // Keep last value
    
    const auto& values = matrix.vValues();
    
    EXPECT_EQ(values.size(), 1u);
    EXPECT_DOUBLE_EQ(values[0], 3.0); // Should be the last value
}

TEST(CSRBuilderTest, EmptyMatrix)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Build without adding any entries
    auto matrix = builder.Build(0, 5, 5);
    
    EXPECT_EQ(matrix.iGlobalRowBegin(), 0);
    EXPECT_EQ(matrix.iGlobalRowEnd(), 5);
    EXPECT_EQ(matrix.iGlobalColCount(), 5);
    EXPECT_EQ(matrix.vRowPtr().size(), 6u);
    EXPECT_EQ(matrix.vColInd().size(), 0u);
    EXPECT_EQ(matrix.vValues().size(), 0u);
    
    // All row pointers should be 0
    const auto& rowPtr = matrix.vRowPtr();
    for (std::size_t i = 0; i < rowPtr.size(); ++i) {
        EXPECT_EQ(rowPtr[i], 0);
    }
}

TEST(CSRBuilderTest, SingleElement)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    builder.AddEntry(0, 0, 42.0);
    
    auto matrix = builder.Build(0, 1, 1);
    
    EXPECT_EQ(matrix.vColInd().size(), 1u);
    EXPECT_EQ(matrix.vValues().size(), 1u);
    EXPECT_DOUBLE_EQ(matrix.vValues()[0], 42.0);
}

TEST(CSRBuilderTest, AllDuplicates)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // All entries are duplicates of the same element
    for (int i = 0; i < 100; ++i) {
        builder.AddEntry(2, 3, 1.0);
    }
    
    auto matrix = builder.Build(0, 5, 5);
    
    EXPECT_EQ(matrix.vColInd().size(), 1u);
    EXPECT_EQ(matrix.vValues().size(), 1u);
    EXPECT_DOUBLE_EQ(matrix.vValues()[0], 100.0);
}

TEST(CSRBuilderTest, UnsortedInput)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Add entries in random order
    builder.AddEntry(2, 2, 9.0);
    builder.AddEntry(0, 1, 2.0);
    builder.AddEntry(1, 0, 3.0);
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(2, 0, 7.0);
    builder.AddEntry(1, 2, 6.0);
    builder.AddEntry(1, 1, 5.0);
    builder.AddEntry(2, 1, 8.0);
    
    auto matrix = builder.Build(0, 3, 3);
    
    // Should be sorted by row then column
    const auto& colInd = matrix.vColInd();
    const auto& values = matrix.vValues();
    const auto& rowPtr = matrix.vRowPtr();
    
    // Row 0: cols 0,1
    EXPECT_EQ(colInd[0], 0);
    EXPECT_EQ(colInd[1], 1);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 2.0);
    
    // Row 1: cols 0,1,2
    EXPECT_EQ(colInd[2], 0);
    EXPECT_EQ(colInd[3], 1);
    EXPECT_EQ(colInd[4], 2);
    EXPECT_DOUBLE_EQ(values[2], 3.0);
    EXPECT_DOUBLE_EQ(values[3], 5.0);
    EXPECT_DOUBLE_EQ(values[4], 6.0);
    
    // Row 2: cols 0,1,2
    EXPECT_EQ(colInd[5], 0);
    EXPECT_EQ(colInd[6], 1);
    EXPECT_EQ(colInd[7], 2);
    EXPECT_DOUBLE_EQ(values[5], 7.0);
    EXPECT_DOUBLE_EQ(values[6], 8.0);
    EXPECT_DOUBLE_EQ(values[7], 9.0);
}

TEST(CSRBuilderTest, SparseWithGaps)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Large matrix with only a few entries
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(50, 75, 2.0);
    builder.AddEntry(99, 99, 3.0);
    
    auto matrix = builder.Build(0, 100, 100);
    
    EXPECT_EQ(matrix.iGlobalRowBegin(), 0);
    EXPECT_EQ(matrix.iGlobalRowEnd(), 100);
    EXPECT_EQ(matrix.vRowPtr().size(), 101u);
    EXPECT_EQ(matrix.vColInd().size(), 3u);
    EXPECT_EQ(matrix.vValues().size(), 3u);
}

TEST(CSRBuilderTest, ScatteredDuplicates)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Mix of unique and duplicate entries
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(0, 1, 2.0);
    builder.AddEntry(0, 0, 1.0); // duplicate
    builder.AddEntry(1, 1, 3.0);
    builder.AddEntry(0, 1, 2.0); // duplicate
    builder.AddEntry(1, 1, 3.0); // duplicate
    builder.AddEntry(2, 2, 5.0);
    
    auto matrix = builder.Build(0, 3, 3);
    
    const auto& colInd = matrix.vColInd();
    const auto& values = matrix.vValues();
    
    EXPECT_EQ(colInd.size(), 4u); // 4 unique positions: (0,0), (0,1), (1,1), (2,2)
    
    // (0,0) should be 2.0
    EXPECT_DOUBLE_EQ(values[0], 2.0);
    // (0,1) should be 4.0
    EXPECT_DOUBLE_EQ(values[1], 4.0);
    // (1,1) should be 6.0
    EXPECT_DOUBLE_EQ(values[2], 6.0);
    // (2,2) should be 5.0
    EXPECT_DOUBLE_EQ(values[3], 5.0);
}

TEST(CSRBuilderTest, SpMVWithBuiltMatrix)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Build a matrix and verify SpMV works correctly
    // [2  1]
    // [1  3]
    builder.AddEntry(0, 0, 2.0);
    builder.AddEntry(0, 1, 1.0);
    builder.AddEntry(1, 0, 1.0);
    builder.AddEntry(1, 1, 3.0);
    
    auto matrix = builder.Build(0, 2, 2);
    
    std::vector<Scalar> x { 1.0, 2.0 };
    std::vector<Scalar> y;
    
    SpMV(matrix, x, y);
    
    ASSERT_EQ(y.size(), 2u);
    EXPECT_DOUBLE_EQ(y[0], 4.0);  // 2*1 + 1*2
    EXPECT_DOUBLE_EQ(y[1], 7.0);  // 1*1 + 3*2
}

TEST(CSRBuilderTest, RowRangeSubset)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    // Add entries for a larger matrix but build only a subset
    builder.AddEntry(5, 0, 1.0);
    builder.AddEntry(6, 1, 2.0);
    builder.AddEntry(7, 2, 3.0);
    builder.AddEntry(3, 0, 99.0);  // Out of range
    builder.AddEntry(10, 0, 99.0); // Out of range
    
    auto matrix = builder.Build(5, 8, 3); // Only rows 5-7
    
    EXPECT_EQ(matrix.iGlobalRowBegin(), 5);
    EXPECT_EQ(matrix.iGlobalRowEnd(), 8);
    EXPECT_EQ(matrix.vColInd().size(), 3u);
    EXPECT_EQ(matrix.vValues().size(), 3u);
    
    // Verify only in-range entries were included
    const auto& values = matrix.vValues();
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 2.0);
    EXPECT_DOUBLE_EQ(values[2], 3.0);
}

TEST(CSRBuilderTest, AddEntriesBatch)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    std::vector<cTriplet<Scalar>> triplets;
    triplets.push_back(cTriplet<Scalar>(0, 0, 1.0));
    triplets.push_back(cTriplet<Scalar>(0, 1, 2.0));
    triplets.push_back(cTriplet<Scalar>(1, 1, 3.0));
    
    builder.AddEntries(triplets);
    
    auto matrix = builder.Build(0, 2, 2);
    
    EXPECT_EQ(matrix.vColInd().size(), 3u);
    EXPECT_EQ(matrix.vValues().size(), 3u);
}

TEST(CSRBuilderTest, ClearAndReuse)
{
    cCSRMatrixBuilder<Scalar> builder;
    
    builder.AddEntry(0, 0, 1.0);
    auto matrix1 = builder.Build(0, 1, 1);
    EXPECT_EQ(matrix1.vValues().size(), 1u);
    
    builder.Clear();
    builder.AddEntry(0, 0, 2.0);
    builder.AddEntry(0, 1, 3.0);
    auto matrix2 = builder.Build(0, 1, 2);
    EXPECT_EQ(matrix2.vValues().size(), 2u);
}

// Test with different scalar types
TEST(CSRBuilderTest, FloatScalar)
{
    cCSRMatrixBuilder<float> builder;
    
    builder.AddEntry(0, 0, 1.0f);
    builder.AddEntry(0, 0, 2.0f); // Duplicate
    
    auto matrix = builder.Build(0, 1, 1);
    
    EXPECT_EQ(matrix.vValues().size(), 1u);
    EXPECT_FLOAT_EQ(matrix.vValues()[0], 3.0f);
}

TEST(CSRBuilderTest, ComplexScalar)
{
    using ComplexScalar = std::complex<double>;
    cCSRMatrixBuilder<ComplexScalar> builder;
    
    builder.AddEntry(0, 0, ComplexScalar(1.0, 2.0));
    builder.AddEntry(0, 0, ComplexScalar(3.0, 4.0)); // Duplicate
    
    auto matrix = builder.Build(0, 1, 1);
    
    EXPECT_EQ(matrix.vValues().size(), 1u);
    EXPECT_DOUBLE_EQ(matrix.vValues()[0].real(), 4.0);
    EXPECT_DOUBLE_EQ(matrix.vValues()[0].imag(), 6.0);
}

TEST(CSRBuilderTest, ConvenienceFunctionBuildFromTriplets)
{
    std::vector<cTriplet<Scalar>> triplets;
    triplets.push_back(cTriplet<Scalar>(0, 0, 1.0));
    triplets.push_back(cTriplet<Scalar>(0, 1, 2.0));
    triplets.push_back(cTriplet<Scalar>(1, 1, 3.0));
    
    auto matrix = BuildCSRFromTriplets(triplets, 0, 2, 2);
    
    EXPECT_EQ(matrix.vColInd().size(), 3u);
    EXPECT_EQ(matrix.vValues().size(), 3u);
    
    const auto& values = matrix.vValues();
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 2.0);
    EXPECT_DOUBLE_EQ(values[2], 3.0);
}

TEST(CSRBuilderTest, ValidateCorrectMatrix)
{
    cCSRMatrixBuilder<Scalar> builder;
    builder.AddEntry(0, 0, 1.0);
    builder.AddEntry(0, 1, 2.0);
    builder.AddEntry(1, 1, 3.0);
    
    auto matrix = builder.Build(0, 2, 2);
    
    std::string errorMsg;
    EXPECT_TRUE(ValidateCSRMatrix(matrix, &errorMsg)) << errorMsg;
}

TEST(CSRBuilderTest, ValidateDetectsInvalidRowPtrSize)
{
    // Manually create invalid matrix
    std::vector<iIndex> rowPtr = { 0, 1 }; // Should be size 3 for 2 rows
    std::vector<iIndex> colInd = { 0 };
    std::vector<Scalar> values = { 1.0 };
    
    cCSRMatrix<Scalar> matrix(0, 2, 2, rowPtr, colInd, values);
    
    std::string errorMsg;
    EXPECT_FALSE(ValidateCSRMatrix(matrix, &errorMsg));
    EXPECT_EQ(errorMsg, "Row pointer size mismatch");
}

TEST(CSRBuilderTest, ValidateDetectsUnsortedColumns)
{
    // Manually create matrix with unsorted columns
    std::vector<iIndex> rowPtr = { 0, 2 };
    std::vector<iIndex> colInd = { 1, 0 }; // Unsorted!
    std::vector<Scalar> values = { 1.0, 2.0 };
    
    cCSRMatrix<Scalar> matrix(0, 1, 2, rowPtr, colInd, values);
    
    std::string errorMsg;
    EXPECT_FALSE(ValidateCSRMatrix(matrix, &errorMsg));
    EXPECT_EQ(errorMsg, "Column indices are not sorted within row");
}

TEST(CSRBuilderTest, ValidateDetectsColumnOutOfBounds)
{
    // Manually create matrix with out-of-bounds column
    std::vector<iIndex> rowPtr = { 0, 1 };
    std::vector<iIndex> colInd = { 10 }; // Out of bounds for colCount=2
    std::vector<Scalar> values = { 1.0 };
    
    cCSRMatrix<Scalar> matrix(0, 1, 2, rowPtr, colInd, values);
    
    std::string errorMsg;
    EXPECT_FALSE(ValidateCSRMatrix(matrix, &errorMsg));
    EXPECT_EQ(errorMsg, "Column index out of bounds");
}

TEST(CSRBuilderTest, BuiltMatrixAlwaysValid)
{
    // Any matrix built with the builder should be valid
    cCSRMatrixBuilder<Scalar> builder;

    // Add random entries
    builder.AddEntry(5, 3, 1.0);
    builder.AddEntry(2, 7, 2.0);
    builder.AddEntry(8, 1, 3.0);
    builder.AddEntry(0, 9, 4.0);
    builder.AddEntry(2, 7, 5.0); // duplicate

    auto matrix = builder.Build(0, 10, 10);

    EXPECT_TRUE(ValidateCSRMatrix(matrix));
}

// ---------------------------------------------------------------------------
// Bit-exact equivalence against the previous (hash-map) implementation.
// The reference below is a verbatim copy of the pre-issue-7 Build() pipeline;
// it exists only inside the tests to guarantee the sort-based rewrite produces
// identical structure AND identical floating-point bit patterns.
// ---------------------------------------------------------------------------
namespace refimpl {

template <typename Scalar>
cCSRMatrix<Scalar> BuildReference(const std::vector<cTriplet<Scalar>>& vTriplets,
    iSize iGlobalRowBegin, iSize iGlobalRowEnd, iSize iGlobalColCount, bool bAccumulateDuplicates)
{
    struct cKey {
        iIndex r;
        iIndex c;
        bool operator==(const cKey& other) const { return r == other.r && c == other.c; }
    };
    struct cKeyHash {
        std::size_t operator()(const cKey& k) const noexcept
        {
            std::size_t seed = std::hash<iIndex>()(k.r);
            seed ^= std::hash<iIndex>()(k.c) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    iSize iLocalRows = iGlobalRowEnd - iGlobalRowBegin;

    std::unordered_map<cKey, Scalar, cKeyHash> mEntries;
    mEntries.reserve(vTriplets.size());

    for (const auto& triplet : vTriplets) {
        if (triplet.m_iRow < iGlobalRowBegin || triplet.m_iRow >= iGlobalRowEnd)
            continue;
        cKey key { triplet.m_iRow, triplet.m_iCol };
        auto it = mEntries.find(key);
        if (it == mEntries.end()) {
            mEntries.emplace(key, triplet.m_vValue);
        } else {
            if (bAccumulateDuplicates)
                it->second += triplet.m_vValue;
            else
                it->second = triplet.m_vValue;
        }
    }

    struct cEntry {
        iIndex r;
        iIndex c;
        Scalar v;
    };
    std::vector<cEntry> vSortedEntries;
    vSortedEntries.reserve(mEntries.size());
    for (const auto& p : mEntries)
        vSortedEntries.push_back({ p.first.r, p.first.c, p.second });

    std::sort(vSortedEntries.begin(), vSortedEntries.end(), [](const cEntry& a, const cEntry& b) {
        if (a.r != b.r)
            return a.r < b.r;
        return a.c < b.c;
    });

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
    for (iSize r = 0; r < iLocalRows; ++r) {
        vRowPtr[static_cast<std::size_t>(r + 1)] += vRowPtr[static_cast<std::size_t>(r)];
    }

    return cCSRMatrix<Scalar>(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount,
        std::move(vRowPtr), std::move(vColInd), std::move(vValues));
}

} // namespace refimpl

template <typename Scalar>
static void ExpectBitwiseEqual(const cCSRMatrix<Scalar>& a, const cCSRMatrix<Scalar>& b)
{
    ASSERT_EQ(a.iGlobalRowBegin(), b.iGlobalRowBegin());
    ASSERT_EQ(a.iGlobalRowEnd(), b.iGlobalRowEnd());
    ASSERT_EQ(a.iGlobalColCount(), b.iGlobalColCount());
    ASSERT_EQ(a.vRowPtr(), b.vRowPtr());
    ASSERT_EQ(a.vColInd(), b.vColInd());
    ASSERT_EQ(a.vValues().size(), b.vValues().size());
    // Bitwise comparison: identical accumulation order must give identical bits
    for (std::size_t k = 0; k < a.vValues().size(); ++k) {
        if constexpr (csr4mpi::is_complex_v<Scalar>) {
            EXPECT_EQ(std::memcmp(&a.vValues()[k], &b.vValues()[k], sizeof(Scalar)), 0)
                << "value bits differ at " << k;
        } else {
            EXPECT_EQ(a.vValues()[k], b.vValues()[k]) << "value differs at " << k;
        }
    }
}

// Randomized triplets with duplicates: accumulate mode
TEST(CSRBuilderTest, BitwiseEqualsReferenceAccumulate)
{
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<iIndex> rowDist(0, 199);
    std::uniform_int_distribution<iIndex> colDist(0, 299);
    std::uniform_real_distribution<double> valDist(-10.0, 10.0);

    std::vector<cTriplet<Scalar>> triplets;
    triplets.reserve(20000);
    for (int k = 0; k < 20000; ++k)
        triplets.emplace_back(rowDist(rng), colDist(rng), static_cast<Scalar>(valDist(rng)));

    cCSRMatrixBuilder<Scalar> builder;
    builder.AddEntries(triplets);
    auto got = builder.Build(0, 200, 300, true);
    auto want = refimpl::BuildReference(triplets, 0, 200, 300, true);
    ExpectBitwiseEqual(got, want);
    EXPECT_TRUE(ValidateCSRMatrix(got));
}

// Randomized triplets with duplicates: keep-last mode
TEST(CSRBuilderTest, BitwiseEqualsReferenceKeepLast)
{
    std::mt19937_64 rng(54321);
    std::uniform_int_distribution<iIndex> rowDist(0, 99);
    std::uniform_int_distribution<iIndex> colDist(0, 149);
    std::uniform_real_distribution<double> valDist(-10.0, 10.0);

    std::vector<cTriplet<Scalar>> triplets;
    triplets.reserve(15000);
    for (int k = 0; k < 15000; ++k)
        triplets.emplace_back(rowDist(rng), colDist(rng), static_cast<Scalar>(valDist(rng)));

    cCSRMatrixBuilder<Scalar> builder;
    builder.AddEntries(triplets);
    auto got = builder.Build(0, 100, 150, false);
    auto want = refimpl::BuildReference(triplets, 0, 100, 150, false);
    ExpectBitwiseEqual(got, want);
}

// MLFMA-like near-field pattern: block-dense bands + injected duplicates,
// including out-of-range rows that must be filtered identically
TEST(CSRBuilderTest, BitwiseEqualsReferenceNearFieldPattern)
{
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> valDist(-1.0, 1.0);
    const int nBlocks = 64;
    const int blockSize = 8;
    const int nRows = nBlocks * blockSize;

    auto idx = [](int blk, int i) { return static_cast<iIndex>(blk) * blockSize + i; };
    std::vector<cTriplet<Scalar>> triplets;
    for (int bi = 0; bi < nBlocks; ++bi) {
        for (int bj = std::max(0, bi - 7); bj <= std::min(nBlocks - 1, bi + 7); ++bj) {
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    triplets.emplace_back(idx(bi, i), idx(bj, j), static_cast<Scalar>(valDist(rng)));
        }
    }
    // Re-emit 10% duplicates (exercises accumulation) and add out-of-range rows
    std::uniform_int_distribution<std::size_t> pick(0, triplets.size() - 1);
    const std::size_t nDup = triplets.size() / 10;
    triplets.reserve(triplets.size() + nDup + 100);
    for (std::size_t k = 0; k < nDup; ++k) {
        const auto& t = triplets[pick(rng)];
        triplets.emplace_back(t.m_iRow, t.m_iCol, static_cast<Scalar>(valDist(rng)));
    }
    for (int k = 0; k < 100; ++k)
        triplets.emplace_back(nRows + k, 0, static_cast<Scalar>(valDist(rng))); // out of range

    for (bool accumulate : { true, false }) {
        cCSRMatrixBuilder<Scalar> builder;
        builder.AddEntries(triplets);
        auto got = builder.Build(0, nRows, nRows, accumulate);
        auto want = refimpl::BuildReference(triplets, 0, nRows, nRows, accumulate);
        ExpectBitwiseEqual(got, want);
        EXPECT_TRUE(ValidateCSRMatrix(got));
    }
}

// Empty input and empty row range remain trivially equal
TEST(CSRBuilderTest, BitwiseEqualsReferenceEdgeCases)
{
    // No entries at all
    std::vector<cTriplet<Scalar>> empty;
    auto gotA = cCSRMatrixBuilder<Scalar>().Build(0, 5, 5);
    auto wantA = refimpl::BuildReference(empty, 0, 5, 5, true);
    ExpectBitwiseEqual(gotA, wantA);

    // All entries out of range
    std::vector<cTriplet<Scalar>> outOfRange = {
        cTriplet<Scalar>(100, 0, static_cast<Scalar>(1.0)),
        cTriplet<Scalar>(200, 1, static_cast<Scalar>(2.0)),
    };
    auto gotB = cCSRMatrixBuilder<Scalar>().Build(0, 10, 10);
    auto wantB = refimpl::BuildReference(outOfRange, 0, 10, 10, true);
    ExpectBitwiseEqual(gotB, wantB);
    EXPECT_EQ(gotB.vColInd().size(), 0u);
}

// Complex scalar accumulation order must also match bit-for-bit
TEST(CSRBuilderTest, BitwiseEqualsReferenceComplex)
{
    using CScalar = std::complex<double>;
    std::mt19937_64 rng(2468);
    std::uniform_int_distribution<iIndex> rowDist(0, 49);
    std::uniform_int_distribution<iIndex> colDist(0, 79);
    std::uniform_real_distribution<double> valDist(-1.0, 1.0);

    std::vector<cTriplet<CScalar>> triplets;
    for (int k = 0; k < 8000; ++k)
        triplets.emplace_back(rowDist(rng), colDist(rng), CScalar(valDist(rng), valDist(rng)));

    cCSRMatrixBuilder<CScalar> builder;
    builder.AddEntries(triplets);
    auto got = builder.Build(0, 50, 80, true);
    auto want = refimpl::BuildReference(triplets, 0, 50, 80, true);
    ExpectBitwiseEqual(got, want);
}
