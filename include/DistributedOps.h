#pragma once
#include "CSRMatrix.h"
#include "Distribution.h"
#include "Operations.h"
#include <mpi.h>
#include <vector>
#include <numeric>
#include <stdexcept>

namespace csr4mpi {

template <typename Scalar>
void DistributedSpMV(const cCSRMatrix<Scalar>& A, const cRowDistribution& dist,
    const std::vector<Scalar>& xLocalOrGlobal,
    std::vector<Scalar>& y,
    bool bReplicatedX, MPI_Comm comm)
{
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    iSize globalCols = A.iGlobalColCount();
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();

    std::vector<Scalar> xGlobal;
    if (bReplicatedX) {
        if (xLocalOrGlobal.size() != static_cast<size_t>(globalCols)) {
            throw std::runtime_error("DistributedSpMV: replicated x size mismatch");
        }
        xGlobal = xLocalOrGlobal;
    } else {
        iSize ownedBegin = dist.iGlobalRowBegin();
        iSize ownedEnd = dist.iGlobalRowEnd();
        iSize ownedCount = ownedEnd - ownedBegin;
        if (xLocalOrGlobal.size() != static_cast<size_t>(ownedCount)) {
            throw std::runtime_error("DistributedSpMV: local x slice size mismatch");
        }
        std::vector<int> recvCounts(worldSize);
        std::vector<int> displs(worldSize);
        if (worldSize == 1) {
            recvCounts[0] = static_cast<int>(ownedCount);
        } else {
            throw std::runtime_error("DistributedSpMV: recvCounts inference for multi-rank not implemented (need distribution metadata)");
        }
        displs[0] = 0;
        for (int r = 1; r < worldSize; ++r)
            displs[r] = displs[r - 1] + recvCounts[r - 1];
        xGlobal.resize(static_cast<size_t>(globalCols));
        MPI_Datatype tCreatedType = MPI_DATATYPE_NULL;
        MPI_Datatype dt = detail::GetMPIDatatype<Scalar>(&tCreatedType);
        MPI_Allgatherv(xLocalOrGlobal.data(), recvCounts[worldRank], dt,
            xGlobal.data(), recvCounts.data(), displs.data(), dt, comm);
        if (tCreatedType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tCreatedType);
        }
    }

    SpMV(A, xGlobal, y);
}

}
