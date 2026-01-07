#pragma once

#include "CSRMatrix.h"
#include <mpi.h>
#include <vector>

namespace csr4mpi {

template <typename Scalar>
class cPardisoAdapter {
public:
    // Gathers a distributed CSR matrix to Rank 0.
    // If rank != 0, returns a valid but empty/dummy matrix (or nullptr equivalent).
    // If rank == 0, returns a fully assembled Global CSR Matrix.
    static bool GatherToRoot(const cCSRMatrix<Scalar>& localA,
        std::vector<iIndex>& globalRowPtr,
        std::vector<iIndex>& globalColInd,
        std::vector<Scalar>& globalValues,
        iSize& outGlobalRows)
    {
        int rank = 0, size = 1;
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
        }

        if (!initialized || size == 1) {
            // Serial case - just copy
            globalRowPtr = localA.vRowPtr();
            globalColInd = localA.vColInd();
            globalValues = localA.vValues();
            outGlobalRows = localA.iGlobalRowEnd() - localA.iGlobalRowBegin();
            return true;
        }

        // Parallel Case: Gather
        iSize localRows = localA.iGlobalRowEnd() - localA.iGlobalRowBegin();

        // 1. Gather Row Counts
        std::vector<int> recv_counts(size); // Assuming size fits in int (MPI limit)
        int localRowsInt = (int)localRows;
        MPI_Allgather(&localRowsInt, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        iSize totalRows = 0;
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = (int)totalRows;
            totalRows += recv_counts[i];
        }
        outGlobalRows = totalRows;

        // 2. Gather Row NNZ counts (to build RowPtr)
        std::vector<int> localRowNNZ(localRows);
        const auto& lRowPtr = localA.vRowPtr();
        for (iSize i = 0; i < localRows; ++i) {
            localRowNNZ[i] = (int)(lRowPtr[i + 1] - lRowPtr[i]);
        }

        std::vector<int> globalRowNNZ;
        if (rank == 0)
            globalRowNNZ.resize(totalRows);

        MPI_Gatherv(localRowNNZ.data(), localRowsInt, MPI_INT,
            rank == 0 ? globalRowNNZ.data() : nullptr, recv_counts.data(), displs.data(), MPI_INT,
            0, MPI_COMM_WORLD);

        // 3. Root builds Global RowPtr
        iSize totalNNZ = 0;
        if (rank == 0) {
            globalRowPtr.resize(totalRows + 1);
            globalRowPtr[0] = 0;
            for (iSize i = 0; i < totalRows; ++i) {
                globalRowPtr[i + 1] = globalRowPtr[i] + globalRowNNZ[i];
            }
            totalNNZ = globalRowPtr[totalRows];
            globalColInd.resize(totalNNZ);
            globalValues.resize(totalNNZ);
        }

        // 4. Gather Col Indices and Values
        // Need NNZ counts/displs per rank
        std::vector<int> nnz_counts(size);
        int localNNZ = (int)localA.vValues().size();
        MPI_Gather(&localNNZ, 1, MPI_INT, nnz_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> nnz_displs(size);
        if (rank == 0) {
            int d = 0;
            for (int i = 0; i < size; ++i) {
                nnz_displs[i] = d;
                d += nnz_counts[i];
            }
        }

        // We assume iIndex matches MPI_LONG_LONG. If not, needs conversion.
        // Assuming iIndex = global indices = long long.
        MPI_Gatherv(localA.vColInd().data(), localNNZ, MPI_LONG_LONG,
            rank == 0 ? globalColInd.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_LONG_LONG,
            0, MPI_COMM_WORLD);

        MPI_Gatherv(localA.vValues().data(), localNNZ, MPI_DOUBLE, // Assuming Scalar=double
            rank == 0 ? globalValues.data() : nullptr, nnz_counts.data(), nnz_displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        return true;
    }
};

}
