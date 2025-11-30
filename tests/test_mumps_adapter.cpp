#include "CSRMatrix.h"
#include "Distribution.h"
#include "MumpsAdapter.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

TEST(MumpsAdapterTest, ExportLocalBlockSimple)
{
    // 2x3 local block (rows 0..2 global with 0-based in matrix definition).
    std::vector<iIndex> rowPtr = { 0, 2, 3 };
    std::vector<iIndex> colInd = { 0, 2, 1 };
    std::vector<vScalar> values;
    values.push_back(static_cast<vScalar>(10.0));
    values.push_back(static_cast<vScalar>(20.0));
    values.push_back(static_cast<vScalar>(30.0));

    cCSRMatrix local(0, 2, 3, rowPtr, colInd, values);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    std::vector<int> IRN;
    std::vector<int> JCN;
    std::vector<vScalar> A;

    cMumpsAdapter::ExportLocalBlock(local, dist, IRN, JCN, A);

    ASSERT_EQ(IRN.size(), A.size());
    ASSERT_EQ(JCN.size(), A.size());

    // Expect COO entries with 1-based global indices:
    // Row 1: (1,1)=10, (1,3)=20; Row 2: (2,2)=30  (ordering follows CSR traversal)
    EXPECT_EQ(IRN[0], 1);
    EXPECT_EQ(JCN[0], 1);
    EXPECT_EQ(A[0], static_cast<vScalar>(10.0));
    EXPECT_EQ(IRN[1], 1);
    EXPECT_EQ(JCN[1], 3);
    EXPECT_EQ(A[1], static_cast<vScalar>(20.0));
    EXPECT_EQ(IRN[2], 2);
    EXPECT_EQ(JCN[2], 2);
    EXPECT_EQ(A[2], static_cast<vScalar>(30.0));
}

TEST(MumpsAdapterTest, ExportLocalBlockNonZeroOffset)
{
    // Simulate rank 1 of 3 for 6 global rows: rows [2,4) owned locally (2 rows).
    // Local CSR for those two rows (global rows 2 and 3):
    // Row 2: cols 1,3 (values 5,7)   Row 3: col 2 (value 9)
    std::vector<iIndex> vRowPtr { 0, 2, 3 };
    std::vector<iIndex> vColInd { 1, 3, 2 }; // 0-based column indices
    std::vector<vScalar> vValues;
    vValues.push_back(static_cast<vScalar>(5.0));
    vValues.push_back(static_cast<vScalar>(7.0));
    vValues.push_back(static_cast<vScalar>(9.0));
    cCSRMatrix cLocal(2, 4, 6, vRowPtr, vColInd, vValues);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(6, 3, 1); // rank=1 owns [2,4)

    std::vector<int> vIRN;
    std::vector<int> vJCN;
    std::vector<vScalar> vA;
    cMumpsAdapter::ExportLocalBlock(cLocal, dist, vIRN, vJCN, vA);
    ASSERT_EQ(vIRN.size(), 3u);
    ASSERT_EQ(vJCN.size(), 3u);
    ASSERT_EQ(vA.size(), 3u);
    // Expected 1-based rows: 3,3,4 (global rows 2 and 3 -> +1)
    EXPECT_EQ(vIRN[0], 3);
    EXPECT_EQ(vIRN[1], 3);
    EXPECT_EQ(vIRN[2], 4);
    // Expected 1-based cols: (1+1)=2, (3+1)=4, (2+1)=3
    EXPECT_EQ(vJCN[0], 2);
    EXPECT_EQ(vJCN[1], 4);
    EXPECT_EQ(vJCN[2], 3);
    EXPECT_EQ(vA[0], static_cast<vScalar>(5.0));
    EXPECT_EQ(vA[1], static_cast<vScalar>(7.0));
    EXPECT_EQ(vA[2], static_cast<vScalar>(9.0));
}

TEST(MumpsAdapterTest, ExportLocalBlockIntoPreallocated)
{
    std::vector<iIndex> vRowPtr { 0, 1, 3 }; // two rows: first has 1 nnz, second has 2 nnz
    std::vector<iIndex> vColInd { 0, 1, 2 };
    std::vector<vScalar> vValues { static_cast<vScalar>(2.0), static_cast<vScalar>(4.0), static_cast<vScalar>(8.0) };
    cCSRMatrix cLocal(4, 6, 10, vRowPtr, vColInd, vValues); // global rows [4,6)
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(10, 5, 2); // Need rank=2 block starting at row 4? For 10 rows & 5 ranks each gets 2 rows => rank2 owns [4,6)
    // Preallocate
    int iLocalNNZ = static_cast<int>(vRowPtr.back());
    std::vector<int> vIRN(iLocalNNZ);
    std::vector<int> vJCN(iLocalNNZ);
    std::vector<vScalar> vA(iLocalNNZ);
    iSize iWritten = cMumpsAdapter::ExportLocalBlockInto(cLocal, dist, vIRN.data(), vJCN.data(), vA.data());
    ASSERT_EQ(iWritten, vRowPtr.back());
    // Expected rows: global 4,5 -> 1-based 5,5,6
    EXPECT_EQ(vIRN[0], 5);
    EXPECT_EQ(vIRN[1], 6); // wait ordering: first row only 1 entry, second row two entries -> second row entries indices 1 and 2
    // Re-evaluate ordering: First local row (global 4) has one nnz at col0 => (5,1)
    // Second local row (global 5) has two nnz at col1,col2 => (6,2) (6,3)
    EXPECT_EQ(vIRN[0], 5);
    EXPECT_EQ(vJCN[0], 1);
    EXPECT_EQ(vIRN[1], 6);
    EXPECT_EQ(vJCN[1], 2);
    EXPECT_EQ(vIRN[2], 6);
    EXPECT_EQ(vJCN[2], 3);
    EXPECT_EQ(vA[0], static_cast<vScalar>(2.0));
    EXPECT_EQ(vA[1], static_cast<vScalar>(4.0));
    EXPECT_EQ(vA[2], static_cast<vScalar>(8.0));
}

TEST(MumpsAdapterTest, ExportLocalBlockAppendPreservesExisting)
{
    std::vector<iIndex> vRowPtr { 0, 2 };
    std::vector<iIndex> vColInd { 0, 2 };
    std::vector<vScalar> vValues { static_cast<vScalar>(1.0), static_cast<vScalar>(3.0) };
    cCSRMatrix cLocal(0, 1, 5, vRowPtr, vColInd, vValues);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);
    std::vector<int> vIRN { -999 };
    std::vector<int> vJCN { -888 };
    std::vector<vScalar> vA { static_cast<vScalar>(-1.0) };
    cMumpsAdapter::ExportLocalBlock(cLocal, dist, vIRN, vJCN, vA);
    ASSERT_EQ(vIRN.size(), 1u + 2u);
    // Existing sentinel preserved
    EXPECT_EQ(vIRN[0], -999);
    EXPECT_EQ(vJCN[0], -888);
    EXPECT_EQ(vA[0], static_cast<vScalar>(-1.0));
    // Appended entries (row 0 -> 1-based 1)
    EXPECT_EQ(vIRN[1], 1);
    EXPECT_EQ(vJCN[1], 1);
    EXPECT_EQ(vA[1], static_cast<vScalar>(1.0));
    EXPECT_EQ(vIRN[2], 1);
    EXPECT_EQ(vJCN[2], 3);
    EXPECT_EQ(vA[2], static_cast<vScalar>(3.0));
}

TEST(MumpsAdapterTest, ExportLocalBlockEmpty)
{
    std::vector<iIndex> vRowPtr { 0 }; // zero rows
    std::vector<iIndex> vColInd;
    std::vector<vScalar> vValues;
    cCSRMatrix cLocal(0, 0, 4, vRowPtr, vColInd, vValues);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(0, 1, 0);
    std::vector<int> vIRN;
    std::vector<int> vJCN;
    std::vector<vScalar> vA;
    cMumpsAdapter::ExportLocalBlock(cLocal, dist, vIRN, vJCN, vA);
    EXPECT_TRUE(vIRN.empty());
    EXPECT_TRUE(vJCN.empty());
    EXPECT_TRUE(vA.empty());
}
