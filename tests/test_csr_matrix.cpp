#include "CSRComm.h"
#include "CSRMatrix.h"
#include "CommPattern.h"
#include "Distribution.h"
#include <gtest/gtest.h>

using namespace csr4mpi;

TEST(CSRMatrixTest, BasicConstruction)
{
    std::vector<iIndex> vRowPtr;
    vRowPtr.push_back(0);
    vRowPtr.push_back(2);
    vRowPtr.push_back(3);

    std::vector<iIndex> vColInd;
    vColInd.push_back(0);
    vColInd.push_back(2);
    vColInd.push_back(1);

    std::vector<vScalar> vValues;
    vValues.push_back(static_cast<vScalar>(1.0));
    vValues.push_back(static_cast<vScalar>(2.0));
    vValues.push_back(static_cast<vScalar>(3.0));

    cCSRMatrix cLocal(0, 2, 3, vRowPtr, vColInd, vValues);

    EXPECT_EQ(cLocal.iGlobalRowBegin(), 0);
    EXPECT_EQ(cLocal.iGlobalRowEnd(), 2);
    EXPECT_EQ(cLocal.vRowPtr().size(), static_cast<std::size_t>(3));
    EXPECT_EQ(cLocal.vColInd().size(), static_cast<std::size_t>(3));
    EXPECT_EQ(cLocal.vValues().size(), static_cast<std::size_t>(3));
}

TEST(CSRCommTest, LocalAssembleNoMPI)
{
    std::vector<iIndex> vRowPtr;
    vRowPtr.push_back(0);
    vRowPtr.push_back(1);
    vRowPtr.push_back(2);

    std::vector<iIndex> vColInd;
    vColInd.push_back(0);
    vColInd.push_back(1);

    std::vector<vScalar> vValues;
    vValues.push_back(static_cast<vScalar>(1.0));
    vValues.push_back(static_cast<vScalar>(2.0));

    cCSRMatrix cLocal(0, 2, 2, vRowPtr, vColInd, vValues);

    cRowDistribution cDist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    std::vector<cRemoteEntry> vEntries;
    cRemoteEntry cEntry0;
    cEntry0.m_iGlobalRow = 0;
    cEntry0.m_iGlobalCol = 0;
    cEntry0.m_vValue = static_cast<vScalar>(3.0);
    vEntries.push_back(cEntry0);

    cRemoteEntry cEntry1;
    cEntry1.m_iGlobalRow = 1;
    cEntry1.m_iGlobalCol = 1;
    cEntry1.m_vValue = static_cast<vScalar>(4.0);
    vEntries.push_back(cEntry1);

    cCommPattern cPattern;
    cPattern.Build(vEntries, cDist, 0, 1);

    std::vector<vScalar> vContrib;
    vContrib.push_back(static_cast<vScalar>(3.0));
    vContrib.push_back(static_cast<vScalar>(4.0));

    MPI_Init(nullptr, nullptr);
    cCSRComm::Assemble(cLocal, cPattern, vContrib, MPI_COMM_WORLD);
    MPI_Finalize();

    const std::vector<vScalar>& vNewValues = cLocal.vValues();

    EXPECT_EQ(vNewValues[0], static_cast<vScalar>(4.0));
    EXPECT_EQ(vNewValues[1], static_cast<vScalar>(6.0));
}
