#include "DistributedOps.h"
#include "Operations.h"
#include <mpi.h>
#include <numeric>
#include <stdexcept>

namespace csr4mpi {
void DistributedSpMV(const cCSRMatrix& A, const cRowDistribution& dist,
    const std::vector<vScalar>& xLocalOrGlobal,
    std::vector<vScalar>& y,
    bool bReplicatedX, MPI_Comm comm)
{
    int worldSize = 1, worldRank = 0;
    MPI_Comm_size(comm, &worldSize);
    MPI_Comm_rank(comm, &worldRank);

    iSize globalCols = A.iGlobalColCount();
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();

    std::vector<vScalar> xGlobal;
    if (bReplicatedX) {
        if (xLocalOrGlobal.size() != static_cast<size_t>(globalCols)) {
            throw std::runtime_error("DistributedSpMV: replicated x size mismatch");
        }
        // Use directly
        xGlobal = xLocalOrGlobal; // copy to allow unified code path
    } else {
        // Assume column distribution mirrors row distribution (square or compatible).
        // Each rank provides its owned segment; gather all.
        iSize ownedBegin = dist.iGlobalRowBegin(); // local process row begin
        iSize ownedEnd = dist.iGlobalRowEnd();
        iSize ownedCount = ownedEnd - ownedBegin;
        if (xLocalOrGlobal.size() != static_cast<size_t>(ownedCount)) {
            throw std::runtime_error("DistributedSpMV: local x slice size mismatch");
        }
        std::vector<int> recvCounts(worldSize);
        std::vector<int> displs(worldSize);
        // Reconstruct counts by assuming uniform block distribution from constructor parameters not stored.
        // Fallback: infer per-rank sizes from ownedCount when size==1; when size>1 require external provision (not implemented yet).
        if (worldSize == 1) {
            recvCounts[0] = static_cast<int>(ownedCount);
        } else {
            throw std::runtime_error("DistributedSpMV: recvCounts inference for multi-rank not implemented (need distribution metadata)");
        }
        displs[0] = 0;
        for (int r = 1; r < worldSize; ++r)
            displs[r] = displs[r - 1] + recvCounts[r - 1];
        xGlobal.resize(static_cast<size_t>(globalCols));
        // 选择 MPI_Datatype（仅支持实数 float/double；复杂与其它需扩展）
        MPI_Datatype dt;
        if constexpr (std::is_same_v<vScalar, float>)
            dt = MPI_FLOAT;
        else if constexpr (std::is_same_v<vScalar, double>)
            dt = MPI_DOUBLE;
        else {
            throw std::runtime_error("DistributedSpMV: MPI gather not yet implemented for complex types");
        }
        MPI_Allgatherv(xLocalOrGlobal.data(), recvCounts[worldRank], dt,
            xGlobal.data(), recvCounts.data(), displs.data(), dt, comm);
    }

    SpMV(A, xGlobal, y);
}
}
