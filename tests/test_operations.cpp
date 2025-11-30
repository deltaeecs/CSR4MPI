#include "CSRMatrix.h"
#include "Operations.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

// Helper to build a simple 3x3 local CSR matrix with rows [0,3)
static cCSRMatrix BuildSimple3x3()
{
    iSize iGlobalRowBegin = 0;
    iSize iGlobalRowEnd = 3;
    iSize iGlobalColCount = 3;
    // Row 0: cols 0,2 -> values 1,2
    // Row 1: col 1 -> value 3
    // Row 2: cols 0,1,2 -> values 4,5,6
    std::vector<iIndex> vRowPtr = { 0, 2, 3, 6 };
    std::vector<iIndex> vColInd = { 0, 2, 1, 0, 1, 2 };
    std::vector<vScalar> vValues = { 1, 2, 3, 4, 5, 6 };
    return cCSRMatrix(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, vRowPtr, vColInd, vValues);
}

TEST(TestOperations, SpMVBasic)
{
    auto A = BuildSimple3x3();
    std::vector<vScalar> x = { 10, 20, 30 };
    std::vector<vScalar> y;
    SpMV(A, x, y);
    ASSERT_EQ(y.size(), 3u);
    // Row0: 1*10 + 2*30 = 70
    // Row1: 3*20 = 60
    // Row2: 4*10 +5*20 +6*30 = 4*10 +5*20 +180 = 40+100+180 = 320
    EXPECT_EQ(y[0], static_cast<vScalar>(70));
    EXPECT_EQ(y[1], static_cast<vScalar>(60));
    EXPECT_EQ(y[2], static_cast<vScalar>(320));
}

TEST(TestOperations, SpMVInPlace)
{
    auto A = BuildSimple3x3();
    std::vector<vScalar> x = { 10, 20, 30 };
    SpMV(A, x);
    ASSERT_EQ(x.size(), 3u);
    EXPECT_EQ(x[0], static_cast<vScalar>(70));
    EXPECT_EQ(x[1], static_cast<vScalar>(60));
    EXPECT_EQ(x[2], static_cast<vScalar>(320));
}

TEST(TestOperations, SpMMBasic)
{
    auto A = BuildSimple3x3();
    // X: 3x2 column-major -> columns x0={10,20,30}, x1={1,2,3}
    // Stored as [10,20,30, 1,2,3]
    std::vector<vScalar> X = { 10, 20, 30, 1, 2, 3 };
    std::vector<vScalar> Y;
    SpMM(A, X, 2, Y);
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    ASSERT_EQ(Y.size(), static_cast<size_t>(localRows * 2));
    // Column 0 result equals SpMV with x0
    EXPECT_EQ(Y[0], static_cast<vScalar>(70));
    EXPECT_EQ(Y[1], static_cast<vScalar>(60));
    EXPECT_EQ(Y[2], static_cast<vScalar>(320));
    // Column 1: multiply with x1={1,2,3}
    // Row0: 1*1 + 2*3 = 7
    // Row1: 3*2 = 6
    // Row2: 4*1 +5*2 +6*3 = 4 +10 +18 = 32
    EXPECT_EQ(Y[3], static_cast<vScalar>(7));
    EXPECT_EQ(Y[4], static_cast<vScalar>(6));
    EXPECT_EQ(Y[5], static_cast<vScalar>(32));
}

TEST(TestOperations, SpMMInPlace)
{
    auto A = BuildSimple3x3();
    std::vector<vScalar> X = { 10, 20, 30, 1, 2, 3 };
    SpMMInPlace(A, X, 2);
    EXPECT_EQ(X[0], static_cast<vScalar>(70));
    EXPECT_EQ(X[1], static_cast<vScalar>(60));
    EXPECT_EQ(X[2], static_cast<vScalar>(320));
    EXPECT_EQ(X[3], static_cast<vScalar>(7));
    EXPECT_EQ(X[4], static_cast<vScalar>(6));
    EXPECT_EQ(X[5], static_cast<vScalar>(32));
}

TEST(TestOperations, SpMVEmptyRow)
{
    // Matrix 2x3 with second row empty
    iSize iGlobalRowBegin = 0, iGlobalRowEnd = 2, iGlobalColCount = 3;
    std::vector<iIndex> vRowPtr = { 0, 2, 2 };
    std::vector<iIndex> vColInd = { 0, 2 };
    std::vector<vScalar> vValues = { 5, 6 };
    cCSRMatrix A(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, vRowPtr, vColInd, vValues);
    std::vector<vScalar> x = { 1, 2, 3 };
    std::vector<vScalar> y;
    SpMV(A, x, y);
    ASSERT_EQ(y.size(), 2u);
    EXPECT_EQ(y[0], static_cast<vScalar>(5 * 1 + 6 * 3));
    EXPECT_EQ(y[1], static_cast<vScalar>(0));
}

TEST(TestOperations, SpMMEmptyRow)
{
    iSize iGlobalRowBegin = 0, iGlobalRowEnd = 2, iGlobalColCount = 3;
    std::vector<iIndex> vRowPtr = { 0, 2, 2 };
    std::vector<iIndex> vColInd = { 0, 2 };
    std::vector<vScalar> vValues = { 5, 6 };
    cCSRMatrix A(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, vRowPtr, vColInd, vValues);
    // 3x2 dense
    std::vector<vScalar> X = { 1, 2, 3, 4, 5, 6 };
    std::vector<vScalar> Y;
    SpMM(A, X, 2, Y);
    ASSERT_EQ(Y.size(), 4u);
    // Column0: row0 5*1 +6*3=23; row1 0
    EXPECT_EQ(Y[0], static_cast<vScalar>(23));
    EXPECT_EQ(Y[1], static_cast<vScalar>(0));
    // Column1: row0 5*4 +6*6 = 20+36=56; row1 0
    EXPECT_EQ(Y[2], static_cast<vScalar>(56));
    EXPECT_EQ(Y[3], static_cast<vScalar>(0));
}
