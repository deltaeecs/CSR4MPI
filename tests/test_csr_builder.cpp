#include "CSRMatrixBuilder.h"
#include "CSRMatrix.h"
#include "Operations.h"
#include <gtest/gtest.h>
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
