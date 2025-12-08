#include "CSRComm.h"
#include "CSRMatrix.h"
#include "CommPattern.h"
#include "Distribution.h"
#include <gtest/gtest.h>
#include <vector>

using namespace csr4mpi;

// Test with double as the default scalar type
using Scalar = double;

// Test that the binary search optimization correctly finds elements
TEST(CSRCommOptimizationTest, BinarySearchFindsFirstColumn)
{
    // Matrix with multiple columns per row
    std::vector<iIndex> rowPtr = { 0, 5 };  // One row with 5 elements
    std::vector<iIndex> colInd = { 0, 10, 20, 30, 40 };  // Sorted columns
    std::vector<Scalar> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    cCSRMatrix<Scalar> local(0, 1, 50, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);

    // Assemble to the first column
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 0;
    e1.m_vValue = 10.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 10.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[0], 11.0);  // 1.0 + 10.0
    EXPECT_DOUBLE_EQ(vals[1], 2.0);   // unchanged
}

TEST(CSRCommOptimizationTest, BinarySearchFindsLastColumn)
{
    // Matrix with multiple columns per row
    std::vector<iIndex> rowPtr = { 0, 5 };
    std::vector<iIndex> colInd = { 0, 10, 20, 30, 40 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    cCSRMatrix<Scalar> local(0, 1, 50, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);

    // Assemble to the last column
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 40;
    e1.m_vValue = 20.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 20.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[4], 25.0);  // 5.0 + 20.0
    EXPECT_DOUBLE_EQ(vals[0], 1.0);   // unchanged
}

TEST(CSRCommOptimizationTest, BinarySearchFindsMiddleColumn)
{
    // Matrix with multiple columns per row
    std::vector<iIndex> rowPtr = { 0, 5 };
    std::vector<iIndex> colInd = { 0, 10, 20, 30, 40 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    cCSRMatrix<Scalar> local(0, 1, 50, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);

    // Assemble to the middle column
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 20;
    e1.m_vValue = 30.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 30.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[2], 33.0);  // 3.0 + 30.0
    EXPECT_DOUBLE_EQ(vals[0], 1.0);   // unchanged
    EXPECT_DOUBLE_EQ(vals[4], 5.0);   // unchanged
}

TEST(CSRCommOptimizationTest, BinarySearchMultipleRows)
{
    // Matrix with 3 rows, each with different number of columns
    std::vector<iIndex> rowPtr = { 0, 2, 5, 7 };
    std::vector<iIndex> colInd = { 0, 5, 1, 3, 7, 2, 9 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    cCSRMatrix<Scalar> local(0, 3, 10, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(3, 1, 0);

    // Assemble to different rows
    std::vector<cRemoteEntry<Scalar>> entries;
    
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 5;
    e1.m_vValue = 10.0;
    
    cRemoteEntry<Scalar> e2;
    e2.m_iGlobalRow = 1;
    e2.m_iGlobalCol = 3;
    e2.m_vValue = 20.0;
    
    cRemoteEntry<Scalar> e3;
    e3.m_iGlobalRow = 2;
    e3.m_iGlobalCol = 9;
    e3.m_vValue = 30.0;
    
    entries.push_back(e1);
    entries.push_back(e2);
    entries.push_back(e3);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 10.0, 20.0, 30.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[1], 12.0);  // Row 0, col 5: 2.0 + 10.0
    EXPECT_DOUBLE_EQ(vals[3], 24.0);  // Row 1, col 3: 4.0 + 20.0
    EXPECT_DOUBLE_EQ(vals[6], 37.0);  // Row 2, col 9: 7.0 + 30.0
}

TEST(CSRCommOptimizationTest, LargeRowBinarySearchEfficiency)
{
    // Test with a large row to ensure binary search is used (O(log n) vs O(n))
    const int numCols = 1000;
    std::vector<iIndex> rowPtr = { 0, numCols };
    std::vector<iIndex> colInd;
    std::vector<Scalar> values;
    
    for (int i = 0; i < numCols; ++i) {
        colInd.push_back(i * 10);  // Sparse columns
        values.push_back(static_cast<Scalar>(i + 1));
    }
    
    cCSRMatrix<Scalar> local(0, 1, numCols * 10, rowPtr, colInd, values);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);

    // Add contribution to a column near the end
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 9990;  // Last column
    e1.m_vValue = 999.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 999.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[999], 1999.0);  // 1000.0 + 999.0
}

TEST(CSRCommOptimizationTest, MultipleDuplicateAccumulations)
{
    // Matrix with sorted columns
    std::vector<iIndex> rowPtr = { 0, 3, 6 };
    std::vector<iIndex> colInd = { 0, 5, 10, 1, 6, 11 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    cCSRMatrix<Scalar> local(0, 2, 15, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    // Multiple contributions to the same element
    std::vector<cRemoteEntry<Scalar>> entries;
    for (int i = 0; i < 5; ++i) {
        cRemoteEntry<Scalar> e;
        e.m_iGlobalRow = 0;
        e.m_iGlobalCol = 5;
        e.m_vValue = 10.0;
        entries.push_back(e);
    }

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib(5, 10.0);

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[1], 52.0);  // 2.0 + 5*10.0
}

TEST(CSRCommOptimizationTest, EmptyRow)
{
    // Matrix with an empty row
    std::vector<iIndex> rowPtr = { 0, 0, 3 };  // Row 0 is empty
    std::vector<iIndex> colInd = { 1, 5, 8 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0 };
    cCSRMatrix<Scalar> local(0, 2, 10, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    // Try to assemble to row 1
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 1;
    e1.m_iGlobalCol = 5;
    e1.m_vValue = 100.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 100.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    EXPECT_DOUBLE_EQ(vals[1], 102.0);  // 2.0 + 100.0
}

TEST(CSRCommOptimizationTest, NonExistentColumnIgnored)
{
    // Matrix with specific columns
    std::vector<iIndex> rowPtr = { 0, 3 };
    std::vector<iIndex> colInd = { 0, 5, 10 };
    std::vector<Scalar> values = { 1.0, 2.0, 3.0 };
    cCSRMatrix<Scalar> local(0, 1, 15, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);

    // Try to assemble to a column that doesn't exist in the matrix
    std::vector<cRemoteEntry<Scalar>> entries;
    cRemoteEntry<Scalar> e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 7;  // Not in colInd
    e1.m_vValue = 100.0;
    entries.push_back(e1);

    cCommPattern<Scalar> pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<Scalar> contrib = { 100.0 };

    MPI_Init(nullptr, nullptr);
    cCSRComm<Scalar>::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const auto& vals = local.vValues();
    // Values should remain unchanged
    EXPECT_DOUBLE_EQ(vals[0], 1.0);
    EXPECT_DOUBLE_EQ(vals[1], 2.0);
    EXPECT_DOUBLE_EQ(vals[2], 3.0);
}
