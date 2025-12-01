#include "CSRMatrix.h"
#include "MatrixMarketLoader.h"
#include "Operations.h"
#include <gtest/gtest.h>
#include <string>

using namespace csr4mpi;

static cCSRMatrix BuildFromMM(const std::string& path)
{
    std::vector<iIndex> rowPtr, colInd;
    std::vector<vScalar> values;
    iIndex rows = 0, cols = 0;
    bool ok = LoadMatrixMarket(path, rowPtr, colInd, values, rows, cols);
    if (!ok)
        throw std::runtime_error("Failed to load MatrixMarket file: " + path);
    return cCSRMatrix(0, rows, cols, rowPtr, colInd, values);
}

TEST(MatrixMarketTest, Lap5SpMVOnes)
{
    std::string path = std::string(CSR4MPI_SOURCE_DIR) + "/tests/data/lap5.mtx";
    auto A = BuildFromMM(path);
    std::vector<vScalar> x(static_cast<size_t>(A.iGlobalColCount()), static_cast<vScalar>(1));
    std::vector<vScalar> y;
    SpMV(A, x, y);
    ASSERT_EQ(y.size(), static_cast<size_t>(5));
    EXPECT_EQ(y[0], static_cast<vScalar>(1));
    EXPECT_EQ(y[1], static_cast<vScalar>(0));
    EXPECT_EQ(y[2], static_cast<vScalar>(0));
    EXPECT_EQ(y[3], static_cast<vScalar>(0));
    EXPECT_EQ(y[4], static_cast<vScalar>(1));
}

TEST(MatrixMarketTest, Small4DuplicateAccumulation)
{
    std::string path = std::string(CSR4MPI_SOURCE_DIR) + "/tests/data/small4.mtx";
    auto A = BuildFromMM(path);
    // Check duplicated (3,4) aggregated to 14
    const auto& rowPtr = A.vRowPtr();
    const auto& colInd = A.vColInd();
    const auto& values = A.vValues();
    // Find row 2 (0-based index 2 -> global row 3) entries
    iIndex r = 2;
    iIndex start = rowPtr[static_cast<size_t>(r)];
    iIndex end = rowPtr[static_cast<size_t>(r + 1)];
    bool found = false;
    vScalar val14 = static_cast<vScalar>(0);
    for (iIndex k = start; k < end; ++k) {
        if (colInd[static_cast<size_t>(k)] == 3) {
            found = true;
            val14 = values[static_cast<size_t>(k)];
            break;
        }
    }
    ASSERT_TRUE(found);
    EXPECT_EQ(val14, static_cast<vScalar>(14));

    std::vector<vScalar> x { static_cast<vScalar>(1), static_cast<vScalar>(2), static_cast<vScalar>(3), static_cast<vScalar>(4) };
    std::vector<vScalar> y;
    SpMV(A, x, y);
    ASSERT_EQ(y.size(), static_cast<size_t>(4));
    EXPECT_EQ(y[0], static_cast<vScalar>(19)); // 10*1 + 3*3
    EXPECT_EQ(y[1], static_cast<vScalar>(10)); // 5*2
    EXPECT_EQ(y[2], static_cast<vScalar>(2 + 14 * 4)); // 2*1 +14*4=58
    EXPECT_EQ(y[3], static_cast<vScalar>(4)); // 1*4
}

TEST(MatrixMarketTest, Dup3AccumulationAndSpMV)
{
    std::string path = std::string(CSR4MPI_SOURCE_DIR) + "/tests/data/dup3.mtx";
    auto A = BuildFromMM(path);
    // Row0 col0 should be 3 (1+2)
    const auto& rowPtr = A.vRowPtr();
    const auto& colInd = A.vColInd();
    const auto& values = A.vValues();
    // locate (0,0)
    iIndex start = rowPtr[0];
    iIndex end = rowPtr[1];
    vScalar v00 = static_cast<vScalar>(0);
    bool f = false;
    for (iIndex k = start; k < end; ++k) {
        if (colInd[static_cast<size_t>(k)] == 0) {
            v00 = values[static_cast<size_t>(k)];
            f = true;
            break;
        }
    }
    ASSERT_TRUE(f);
    EXPECT_EQ(v00, static_cast<vScalar>(3));
    std::vector<vScalar> x { static_cast<vScalar>(1), static_cast<vScalar>(1), static_cast<vScalar>(1) };
    std::vector<vScalar> y;
    SpMV(A, x, y);
    ASSERT_EQ(y.size(), static_cast<size_t>(3));
    EXPECT_EQ(y[0], static_cast<vScalar>(7));
    EXPECT_EQ(y[1], static_cast<vScalar>(5));
    EXPECT_EQ(y[2], static_cast<vScalar>(13));
}
