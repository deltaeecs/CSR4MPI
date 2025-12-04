/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-11-30 05:19:25
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-11-30 15:54:23
 * @FilePath: \CSR4MPI\tests\test_mpi_symmetric_spmm.cpp
 * @Description:
 *
 * Copyright (c) 2025 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved.
 */
#include "CSRMatrix.h"
#include "Distribution.h"
#include "Operations.h"
#include <gtest/gtest.h>
#include <mpi.h>

using namespace csr4mpi;
using Scalar = double;

// 分布式对称（下三角存储）矩阵 4x4：
// Lower stored entries per row (owned rows only):
// Row0: (0,0)=2
// Row1: (1,0)=1 (1,1)=3
// Row2: (2,1)=4 (2,2)=5
// Row3: (3,2)=6 (3,3)=7
// Full symmetric expansion -> additional upper entries: (0,1)=1,(1,2)=4,(2,3)=6
// Full matrix:
// [2 1 0 0]
// [1 3 4 0]
// [0 4 5 6]
// [0 0 6 7]
// Dense X (col-major) 4x2: col0 = [1,1,1,1], col1 = [2,3,4,5]
// Expected Y columns:
// col0: [3,8,15,13]
// col1: [7,27,62,59]

TEST(MpiSymmetricSpMMTest, DistributedLowerExpansionGatherMM)
{
    int initFlag = 0;
    MPI_Initialized(&initFlag);
    if (!initFlag)
        MPI_Init(nullptr, nullptr);
    int worldSize = 1, rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize != 2) {
        int finFlagEarly = 0;
        MPI_Finalized(&finFlagEarly);
        if (!finFlagEarly && !initFlag)
            MPI_Finalize();
        GTEST_SKIP() << "Requires 2 MPI ranks";
    }
    // Block row distribution 4 rows across 2 ranks => rows per rank =2
    auto dist = std::make_shared<cRowDistribution>(cRowDistribution::CreateBlockDistribution(4, worldSize, rank));
    iSize iGlobalRowBegin = dist->iGlobalRowBegin();
    iSize iGlobalRowEnd = dist->iGlobalRowEnd();
    iSize localRows = iGlobalRowEnd - iGlobalRowBegin;
    // Build CSR for owned rows only (lower triangle entries on those rows)
    std::vector<iIndex> vRowPtr;
    std::vector<iIndex> vColInd;
    std::vector<Scalar> vValues;
    vRowPtr.push_back(0);
    for (iIndex gRow = iGlobalRowBegin; gRow < iGlobalRowEnd; ++gRow) {
        if (gRow == 0) {
            vColInd.push_back(0);
            vValues.push_back(static_cast<Scalar>(2));
        } else if (gRow == 1) {
            vColInd.push_back(0);
            vValues.push_back(static_cast<Scalar>(1));
            vColInd.push_back(1);
            vValues.push_back(static_cast<Scalar>(3));
        } else if (gRow == 2) {
            vColInd.push_back(1);
            vValues.push_back(static_cast<Scalar>(4));
            vColInd.push_back(2);
            vValues.push_back(static_cast<Scalar>(5));
        } else if (gRow == 3) {
            vColInd.push_back(2);
            vValues.push_back(static_cast<Scalar>(6));
            vColInd.push_back(3);
            vValues.push_back(static_cast<Scalar>(7));
        }
        vRowPtr.push_back(static_cast<iIndex>(vColInd.size()));
    }
    cCSRMatrix<Scalar> A(iGlobalRowBegin, iGlobalRowEnd, 4, vRowPtr, vColInd, vValues, eSymmLower);
    A.AttachDistribution(dist);
    // Dense X 4x2 col-major
    std::vector<Scalar> X = { static_cast<Scalar>(1), static_cast<Scalar>(1), static_cast<Scalar>(1), static_cast<Scalar>(1),
        static_cast<Scalar>(2), static_cast<Scalar>(3), static_cast<Scalar>(4), static_cast<Scalar>(5) };
    std::vector<Scalar> Y;
    SpMM(A, X, 2, Y);
    ASSERT_EQ(Y.size(), static_cast<size_t>(localRows * 2));
    // Expected slices
    if (rank == 0) {
        EXPECT_EQ(Y[0], static_cast<Scalar>(3));
        EXPECT_EQ(Y[1], static_cast<Scalar>(8));
        EXPECT_EQ(Y[2], static_cast<Scalar>(7));
        EXPECT_EQ(Y[3], static_cast<Scalar>(27));
    } else {
        EXPECT_EQ(Y[0], static_cast<Scalar>(15));
        EXPECT_EQ(Y[1], static_cast<Scalar>(13));
        EXPECT_EQ(Y[2], static_cast<Scalar>(62));
        EXPECT_EQ(Y[3], static_cast<Scalar>(59));
    }
    int finFlag = 0;
    MPI_Finalized(&finFlag);
    if (!finFlag && !initFlag)
        MPI_Finalize();
}

TEST(MpiSymmetricSpMMTest, DistributedLowerExpansionGatherMM4Proc)
{
    int initFlag = 0;
    MPI_Initialized(&initFlag);
    if (!initFlag)
        MPI_Init(nullptr, nullptr);
    int worldSize = 1, rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize != 4) {
        int finFlagEarly = 0;
        MPI_Finalized(&finFlagEarly);
        if (!finFlagEarly && !initFlag)
            MPI_Finalize();
        GTEST_SKIP() << "Requires 4 MPI ranks";
    }
    // Block row distribution 4 rows across 4 ranks => 1 row/rank
    auto dist = std::make_shared<cRowDistribution>(cRowDistribution::CreateBlockDistribution(4, worldSize, rank));
    iSize iGlobalRowBegin = dist->iGlobalRowBegin();
    iSize iGlobalRowEnd = dist->iGlobalRowEnd();
    iSize localRows = iGlobalRowEnd - iGlobalRowBegin;
    std::vector<iIndex> vRowPtr;
    std::vector<iIndex> vColInd;
    std::vector<Scalar> vValues;
    vRowPtr.push_back(0);
    for (iIndex gRow = iGlobalRowBegin; gRow < iGlobalRowEnd; ++gRow) {
        if (gRow == 0) {
            vColInd.push_back(0);
            vValues.push_back(static_cast<Scalar>(2));
        } else if (gRow == 1) {
            vColInd.push_back(0);
            vValues.push_back(static_cast<Scalar>(1));
            vColInd.push_back(1);
            vValues.push_back(static_cast<Scalar>(3));
        } else if (gRow == 2) {
            vColInd.push_back(1);
            vValues.push_back(static_cast<Scalar>(4));
            vColInd.push_back(2);
            vValues.push_back(static_cast<Scalar>(5));
        } else if (gRow == 3) {
            vColInd.push_back(2);
            vValues.push_back(static_cast<Scalar>(6));
            vColInd.push_back(3);
            vValues.push_back(static_cast<Scalar>(7));
        }
        vRowPtr.push_back(static_cast<iIndex>(vColInd.size()));
    }
    cCSRMatrix<Scalar> A(iGlobalRowBegin, iGlobalRowEnd, 4, vRowPtr, vColInd, vValues, eSymmLower);
    A.AttachDistribution(dist);
    // Dense X 4x2 col-major
    std::vector<Scalar> X = { static_cast<Scalar>(1), static_cast<Scalar>(1), static_cast<Scalar>(1), static_cast<Scalar>(1),
        static_cast<Scalar>(2), static_cast<Scalar>(3), static_cast<Scalar>(4), static_cast<Scalar>(5) };
    std::vector<Scalar> Y;
    SpMM(A, X, 2, Y);
    ASSERT_EQ(Y.size(), static_cast<size_t>(localRows * 2));
    // Expected per-rank slice
    switch (rank) {
    case 0:
        EXPECT_EQ(Y[0], static_cast<Scalar>(3));
        EXPECT_EQ(Y[1], static_cast<Scalar>(7));
        break;
    case 1:
        EXPECT_EQ(Y[0], static_cast<Scalar>(8));
        EXPECT_EQ(Y[1], static_cast<Scalar>(27));
        break;
    case 2:
        EXPECT_EQ(Y[0], static_cast<Scalar>(15));
        EXPECT_EQ(Y[1], static_cast<Scalar>(62));
        break;
    case 3:
        EXPECT_EQ(Y[0], static_cast<Scalar>(13));
        EXPECT_EQ(Y[1], static_cast<Scalar>(59));
        break;
    }
    int finFlag = 0;
    MPI_Finalized(&finFlag);
    if (!finFlag && !initFlag)
        MPI_Finalize();
}
