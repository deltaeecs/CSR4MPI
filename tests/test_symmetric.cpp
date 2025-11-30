#include "CSRMatrix.h"
#include "MatrixMarketLoader.h"
#include "Operations.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

static cCSRMatrix LoadSymLower(const std::string& rel)
{
    std::string path = std::string(CSR4MPI_SOURCE_DIR) + rel;
    std::vector<iIndex> rp, ci;
    std::vector<vScalar> vv;
    iIndex rows = 0, cols = 0;
    bool ok = LoadMatrixMarket(path, rp, ci, vv, rows, cols);
    EXPECT_TRUE(ok);
    if (!ok)
        throw std::runtime_error("LoadMatrixMarket failed for sym file");
    // 判定为下三角对称（除对角外 r>=c）
    eSymmStorage eSymm = eSymmLower;
    for (size_t r = 0; r + 1 < rp.size(); ++r) {
        for (iIndex k = rp[r]; k < rp[r + 1]; ++k) {
            if (ci[(size_t)k] > (iIndex)r) {
                eSymm = eSymmFull;
                break;
            }
        }
        if (eSymm == eSymmFull)
            break;
    }
    return cCSRMatrix(0, rows, cols, rp, ci, vv, eSymm);
}

TEST(SymmetricTest, SpMVLowerTriangleExpansion)
{
    auto A = LoadSymLower("/tests/matrices/sym3_lower.mtx");
    ASSERT_TRUE(A.bIsSymmetric());
    std::vector<vScalar> x { static_cast<vScalar>(1), static_cast<vScalar>(2), static_cast<vScalar>(3) };
    std::vector<vScalar> y;
    SpMV(A, x, y);
    // Full matrix equivalent:
    // [4 1 0; 1 3 2; 0 2 5] * [1 2 3] = [4*1+1*2=6, 1*1+3*2+2*3=1+6+6=13, 2*2+5*3=4+15=19]
    ASSERT_EQ(y.size(), (size_t)3);
    EXPECT_EQ(y[0], static_cast<vScalar>(6));
    EXPECT_EQ(y[1], static_cast<vScalar>(13));
    EXPECT_EQ(y[2], static_cast<vScalar>(19));
}

TEST(SymmetricTest, SpMMLowerTriangleExpansion)
{
    auto A = LoadSymLower("/tests/matrices/sym3_lower.mtx");
    iSize nCols = 2;
    // X shape 3 x 2, column-major; columns: [1 2 3]^T and [2 0 1]^T
    std::vector<vScalar> X { static_cast<vScalar>(1), static_cast<vScalar>(2), static_cast<vScalar>(3),
        static_cast<vScalar>(2), static_cast<vScalar>(0), static_cast<vScalar>(1) };
    std::vector<vScalar> Y;
    SpMM(A, X, nCols, Y);
    // First column: same as SpMV with x:[1 2 3] => [6,13,19]
    // Second column: A * [2 0 1] = [4*2 +1*0=8, 1*2+3*0+2*1=2+0+2=4, 2*0+5*1=5]
    ASSERT_EQ(Y.size(), (size_t)(3 * 2));
    EXPECT_EQ(Y[0], static_cast<vScalar>(6));
    EXPECT_EQ(Y[1], static_cast<vScalar>(13));
    EXPECT_EQ(Y[2], static_cast<vScalar>(19));
    EXPECT_EQ(Y[3], static_cast<vScalar>(8));
    EXPECT_EQ(Y[4], static_cast<vScalar>(4));
    EXPECT_EQ(Y[5], static_cast<vScalar>(5));
}
