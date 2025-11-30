#include "CSRComm.h"
#include "CSRMatrix.h"
#include "CommPattern.h"
#include "Distribution.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

TEST(RowDistributionTest, BlockEvenSplit)
{
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(8, 4, 2);
    EXPECT_EQ(dist.iGlobalRowBegin(), 4);
    EXPECT_EQ(dist.iGlobalRowEnd(), 6);
    EXPECT_EQ(dist.iGlobalRowCount(), 8);
    // Owner checks
    EXPECT_EQ(dist.iOwnerRank(0), 0);
    EXPECT_EQ(dist.iOwnerRank(1), 0);
    EXPECT_EQ(dist.iOwnerRank(2), 1);
    EXPECT_EQ(dist.iOwnerRank(3), 1);
    EXPECT_EQ(dist.iOwnerRank(4), 2);
    EXPECT_EQ(dist.iOwnerRank(5), 2);
    EXPECT_EQ(dist.iOwnerRank(6), 3);
    EXPECT_EQ(dist.iOwnerRank(7), 3);
}

TEST(RowDistributionTest, BlockUnevenSplit)
{
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(10, 4, 3); // rows: 3,3,2,2
    EXPECT_EQ(dist.iGlobalRowBegin(), 8);
    EXPECT_EQ(dist.iGlobalRowEnd(), 10);
    EXPECT_EQ(dist.iOwnerRank(0), 0);
    EXPECT_EQ(dist.iOwnerRank(2), 0);
    EXPECT_EQ(dist.iOwnerRank(3), 1);
    EXPECT_EQ(dist.iOwnerRank(5), 1);
    EXPECT_EQ(dist.iOwnerRank(6), 2);
    EXPECT_EQ(dist.iOwnerRank(7), 2);
    EXPECT_EQ(dist.iOwnerRank(8), 3);
    EXPECT_EQ(dist.iOwnerRank(9), 3);
}

TEST(CommPatternTest, DuplicateEntriesAccumulate)
{
    // Local 2x2 matrix with both rows owned.
    std::vector<iIndex> rowPtr = { 0, 2, 4 };
    std::vector<iIndex> colInd = { 0, 1, 0, 1 };
    std::vector<vScalar> values = { 1.0, 2.0, 3.0, 4.0 };
    cCSRMatrix local(0, 2, 2, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    std::vector<cRemoteEntry> entries;
    cRemoteEntry e1;
    e1.m_iGlobalRow = 0;
    e1.m_iGlobalCol = 0;
    e1.m_vValue = static_cast<vScalar>(5.0);
    cRemoteEntry e2;
    e2.m_iGlobalRow = 0;
    e2.m_iGlobalCol = 0;
    e2.m_vValue = static_cast<vScalar>(6.0);
    cRemoteEntry e3;
    e3.m_iGlobalRow = 1;
    e3.m_iGlobalCol = 1;
    e3.m_vValue = static_cast<vScalar>(7.0);
    entries.push_back(e1);
    entries.push_back(e2);
    entries.push_back(e3);

    cCommPattern pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<vScalar> contrib;
    contrib.push_back(static_cast<vScalar>(5.0));
    contrib.push_back(static_cast<vScalar>(6.0));
    contrib.push_back(static_cast<vScalar>(7.0));

    MPI_Init(nullptr, nullptr);
    cCSRComm::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const std::vector<vScalar>& newVals = local.vValues();
    // Original (row0,col0)=1 plus 5 + 6 => 12; (row1,col1)=4 plus 7 => 11
    EXPECT_EQ(newVals[0], static_cast<vScalar>(12.0));
    EXPECT_EQ(newVals[3], static_cast<vScalar>(11.0));
}

TEST(CommPatternTest, EmptyPatternNoChange)
{
    std::vector<iIndex> rowPtr = { 0, 1 };
    std::vector<iIndex> colInd = { 0 };
    std::vector<vScalar> values = { 2.5 };
    cCSRMatrix local(0, 1, 1, rowPtr, colInd, values);

    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(1, 1, 0);
    std::vector<cRemoteEntry> entries; // empty

    cCommPattern pattern;
    pattern.Build(entries, dist, 0, 1);

    std::vector<vScalar> contrib; // empty
    MPI_Init(nullptr, nullptr);
    cCSRComm::Assemble(local, pattern, contrib, MPI_COMM_WORLD);
    MPI_Finalize();

    EXPECT_EQ(local.vValues()[0], static_cast<vScalar>(2.5));
}
