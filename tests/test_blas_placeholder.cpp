#include "BlasAdapter.h"
#include "CSRMatrix.h"
#include "Operations.h"
#include <gtest/gtest.h>

using namespace csr4mpi;
using Scalar = double;

TEST(TestBlasPlaceholder, FlagReflectsOption)
{
#ifdef CSR4MPI_USE_BLAS
    EXPECT_TRUE(bBlasEnabled());
#else
    EXPECT_FALSE(bBlasEnabled());
#endif
}

TEST(TestBlasPlaceholder, SpMMBlasMatchesSpMM)
{
    // Small matrix 3x4
    iSize rBegin = 0, rEnd = 3, cCount = 4;
    std::vector<iIndex> rowPtr = { 0, 2, 3, 5 };
    std::vector<iIndex> colInd = { 0, 2, 1, 1, 3 };
    std::vector<Scalar> values = { 1, 2, 3, 4, 5 };
    cCSRMatrix<Scalar> A(rBegin, rEnd, cCount, rowPtr, colInd, values);
    iSize nCols = 2;
    // Dense RHS (4x2): columns {1,2,3,4} and {5,6,7,8}
    std::vector<Scalar> X = { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<Scalar> Y1, Y2;
    SpMM(A, X, nCols, Y1);
    SpMMBlas(A, X, nCols, Y2);
    ASSERT_EQ(Y1.size(), Y2.size());
    for (size_t i = 0; i < Y1.size(); ++i) {
        EXPECT_EQ(Y1[i], Y2[i]);
    }
}
