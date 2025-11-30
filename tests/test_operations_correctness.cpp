#include "CSRMatrix.h"
#include "Operations.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

// Build a 4x5 sparse matrix with pattern:
// Row0: 1 0 2 0 0
// Row1: 0 0 0 3 0
// Row2: 4 5 0 0 6
// Row3: 0 0 7 0 8
static cCSRMatrix BuildMatrix4x5()
{
    iSize iGlobalRowBegin = 0;
    iSize iGlobalRowEnd = 4;
    iSize iGlobalColCount = 5;
    std::vector<iIndex> vRowPtr = { 0, 2, 3, 6, 8 };
    std::vector<iIndex> vColInd = { 0, 2, 3, 0, 1, 4, 2, 4 };
    std::vector<vScalar> vValues = { 1, 2, 3, 4, 5, 6, 7, 8 };
    return cCSRMatrix(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, vRowPtr, vColInd, vValues);
}

TEST(TestOperationsCorrectness, SpMVMatchesDense)
{
    auto A = BuildMatrix4x5();
    std::vector<vScalar> x = { 10, 11, 12, 13, 14 };
    // Dense reference computation
    iSize rows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    iSize cols = A.iGlobalColCount();
    // Reconstruct dense matrix row-major for reference
    // Using the same integer values ensures exact arithmetic for real types
    std::vector<vScalar> dense(rows * cols, static_cast<vScalar>(0));
    // Fill from pattern
    dense[0 * cols + 0] = static_cast<vScalar>(1);
    dense[0 * cols + 2] = static_cast<vScalar>(2);
    dense[1 * cols + 3] = static_cast<vScalar>(3);
    dense[2 * cols + 0] = static_cast<vScalar>(4);
    dense[2 * cols + 1] = static_cast<vScalar>(5);
    dense[2 * cols + 4] = static_cast<vScalar>(6);
    dense[3 * cols + 2] = static_cast<vScalar>(7);
    dense[3 * cols + 4] = static_cast<vScalar>(8);
    std::vector<vScalar> y_ref(rows, static_cast<vScalar>(0));
    for (iSize r = 0; r < rows; ++r) {
        vScalar acc = static_cast<vScalar>(0);
        for (iSize c = 0; c < cols; ++c) {
            acc += dense[r * cols + c] * x[static_cast<size_t>(c)];
        }
        y_ref[static_cast<size_t>(r)] = acc;
    }
    std::vector<vScalar> y_spmv;
    SpMV(A, x, y_spmv);
    ASSERT_EQ(y_spmv.size(), y_ref.size());
    for (size_t i = 0; i < y_ref.size(); ++i) {
        EXPECT_EQ(y_spmv[i], y_ref[i]);
    }
}

TEST(TestOperationsCorrectness, SpMMMatchesDense)
{
    auto A = BuildMatrix4x5();
    iSize rows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    iSize cols = A.iGlobalColCount();
    iSize nCols = 3; // three dense RHS columns
    // Columns (each length 5):
    // col0: 1 2 3 4 5
    // col1: 6 7 8 9 10
    // col2: 11 12 13 14 15
    std::vector<vScalar> X(cols * nCols, static_cast<vScalar>(0));
    for (iSize c = 0; c < cols; ++c) {
        X[static_cast<size_t>(c + 0 * cols)] = static_cast<vScalar>(1 + c);
        X[static_cast<size_t>(c + 1 * cols)] = static_cast<vScalar>(6 + c);
        X[static_cast<size_t>(c + 2 * cols)] = static_cast<vScalar>(11 + c);
    }
    // Dense matrix reconstruction same as previous test
    std::vector<vScalar> dense(rows * cols, static_cast<vScalar>(0));
    dense[0 * cols + 0] = static_cast<vScalar>(1);
    dense[0 * cols + 2] = static_cast<vScalar>(2);
    dense[1 * cols + 3] = static_cast<vScalar>(3);
    dense[2 * cols + 0] = static_cast<vScalar>(4);
    dense[2 * cols + 1] = static_cast<vScalar>(5);
    dense[2 * cols + 4] = static_cast<vScalar>(6);
    dense[3 * cols + 2] = static_cast<vScalar>(7);
    dense[3 * cols + 4] = static_cast<vScalar>(8);
    // Reference multiplication Y_ref (column-major rows x nCols)
    std::vector<vScalar> Y_ref(rows * nCols, static_cast<vScalar>(0));
    for (iSize j = 0; j < nCols; ++j) {
        for (iSize r = 0; r < rows; ++r) {
            vScalar acc = static_cast<vScalar>(0);
            for (iSize c = 0; c < cols; ++c) {
                acc += dense[r * cols + c] * X[static_cast<size_t>(c + j * cols)];
            }
            Y_ref[static_cast<size_t>(r + j * rows)] = acc;
        }
    }
    std::vector<vScalar> Y_spmm;
    SpMM(A, X, nCols, Y_spmm);
    ASSERT_EQ(Y_spmm.size(), Y_ref.size());
    for (size_t idx = 0; idx < Y_ref.size(); ++idx) {
        EXPECT_EQ(Y_spmm[idx], Y_ref[idx]);
    }
}
