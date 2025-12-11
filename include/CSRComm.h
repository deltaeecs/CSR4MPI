#pragma once

#include "CSRMatrix.h"
#include "CommPattern.h"
#include "Global.h"
#include <algorithm>
#include <mpi.h>
#include <vector>

namespace csr4mpi {

template <typename Scalar>
class cCSRComm {
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

public:
    static void Assemble(cCSRMatrix<Scalar>& cLocal,
        const cCommPattern<Scalar>& cPattern,
        const std::vector<Scalar>& vValues,
        MPI_Comm cComm)
    {
        const std::vector<iIndex>& vSendRows = cPattern.vSendRows();
        const std::vector<iIndex>& vSendCols = cPattern.vSendCols();
        const std::vector<iSize>& vSendOffsets = cPattern.vSendOffsets();
        const std::vector<int>& vTargets = cPattern.vTargetRanks();

        const std::vector<iIndex>& vRowPtr = cLocal.vRowPtr();
        const std::vector<iIndex>& vColInd = cLocal.vColInd();
        std::vector<Scalar>& vLocalVal = cLocal.vValues();

        (void)vSendOffsets;
        (void)vTargets;

        int iRank = 0;
        int iWorldSize = 1;
        MPI_Comm_rank(cComm, &iRank);
        MPI_Comm_size(cComm, &iWorldSize);

        // Prepare send counts per target.
        std::vector<int> vSendCounts(iWorldSize, 0);
        std::vector<int> vRecvCounts(iWorldSize, 0);

        for (std::size_t i = 0; i < vTargets.size(); ++i) {
            int iTarget = vTargets[i];
            iSize iBegin = vSendOffsets[i];
            iSize iEnd = vSendOffsets[i + 1];
            vSendCounts[static_cast<std::size_t>(iTarget)] += static_cast<int>(iEnd - iBegin);
        }

        // Exchange counts.
        MPI_Alltoall(vSendCounts.data(), 1, MPI_INT,
            vRecvCounts.data(), 1, MPI_INT,
            cComm);

        // DEBUG: Print Send Counts
        for (int t = 0; t < iWorldSize; ++t) {
            if (vSendCounts[t] > 0) {
                printf("[Rank %d] Sending %d entries to Rank %d\n", iRank, vSendCounts[t], t);
            }
        }
        fflush(stdout);

        int iTotalSend = 0;
        int iTotalRecv = 0;
        for (int i = 0; i < iWorldSize; ++i) {
            iTotalSend += vSendCounts[static_cast<std::size_t>(i)];
            iTotalRecv += vRecvCounts[static_cast<std::size_t>(i)];
        }

        // Use separate arrays for Rows, Cols, Vals to avoid struct padding/MPI type issues
        std::vector<iIndex> vSendRowsBuf(static_cast<std::size_t>(iTotalSend));
        std::vector<iIndex> vSendColsBuf(static_cast<std::size_t>(iTotalSend));
        std::vector<Scalar> vSendValsBuf(static_cast<std::size_t>(iTotalSend));

        std::vector<iIndex> vRecvRowsBuf(static_cast<std::size_t>(iTotalRecv));
        std::vector<iIndex> vRecvColsBuf(static_cast<std::size_t>(iTotalRecv));
        std::vector<Scalar> vRecvValsBuf(static_cast<std::size_t>(iTotalRecv));

        // Fill send buffers in rank-major order.
        std::vector<int> vOffsets(iWorldSize, 0);
        for (int i = 1; i < iWorldSize; ++i) {
            vOffsets[static_cast<std::size_t>(i)] = vOffsets[static_cast<std::size_t>(i - 1)] + vSendCounts[static_cast<std::size_t>(i - 1)];
        }

        for (std::size_t iBucket = 0; iBucket < vTargets.size(); ++iBucket) {
            int iTarget = vTargets[iBucket];
            iSize iBegin = vSendOffsets[iBucket];
            iSize iEnd = vSendOffsets[iBucket + 1];

            for (iSize i = iBegin; i < iEnd; ++i) {
                int iPos = vOffsets[static_cast<std::size_t>(iTarget)]++;
                vSendRowsBuf[static_cast<std::size_t>(iPos)] = vSendRows[static_cast<std::size_t>(i)];
                vSendColsBuf[static_cast<std::size_t>(iPos)] = vSendCols[static_cast<std::size_t>(i)];
                vSendValsBuf[static_cast<std::size_t>(iPos)] = vValues[static_cast<std::size_t>(i)];
            }
        }

        // Build displacement arrays for Alltoallv.
        std::vector<int> vSendDispls(iWorldSize, 0);
        std::vector<int> vRecvDispls(iWorldSize, 0);

        for (int i = 1; i < iWorldSize; ++i) {
            vSendDispls[static_cast<std::size_t>(i)] = vSendDispls[static_cast<std::size_t>(i - 1)] + vSendCounts[static_cast<std::size_t>(i - 1)];
            vRecvDispls[static_cast<std::size_t>(i)] = vRecvDispls[static_cast<std::size_t>(i - 1)] + vRecvCounts[static_cast<std::size_t>(i - 1)];
        }

        MPI_Datatype tScalarType = mpi_helper::GetMPIDatatype<Scalar>(nullptr);

        // Exchange Rows
        MPI_Alltoallv(vSendRowsBuf.data(), vSendCounts.data(), vSendDispls.data(), MPI_LONG_LONG,
            vRecvRowsBuf.data(), vRecvCounts.data(), vRecvDispls.data(), MPI_LONG_LONG, cComm);

        // Exchange Cols
        MPI_Alltoallv(vSendColsBuf.data(), vSendCounts.data(), vSendDispls.data(), MPI_LONG_LONG,
            vRecvColsBuf.data(), vRecvCounts.data(), vRecvDispls.data(), MPI_LONG_LONG, cComm);

        // Exchange Vals
        MPI_Alltoallv(vSendValsBuf.data(), vSendCounts.data(), vSendDispls.data(), tScalarType,
            vRecvValsBuf.data(), vRecvCounts.data(), vRecvDispls.data(), tScalarType, cComm);

        // Accumulate received contributions into local CSR matrix.
        // Use binary search for better performance (O(log n) instead of O(n) per lookup)
        for (size_t k = 0; k < vRecvRowsBuf.size(); ++k) {
            iIndex iGlobalRow = vRecvRowsBuf[k];
            iIndex iGlobalCol = vRecvColsBuf[k];
            Scalar val = vRecvValsBuf[k];

            iIndex iLocalRow = static_cast<iIndex>(iGlobalRow - cLocal.iGlobalRowBegin());

            if (iLocalRow < 0 || iLocalRow >= static_cast<iIndex>(vRowPtr.size() - 1)) {
                printf("[ERROR] CSRComm::Assemble: Received row %lld which is not local! LocalRange=[%lld, %lld)\n",
                    (long long)iGlobalRow, (long long)cLocal.iGlobalRowBegin(), (long long)cLocal.iGlobalRowEnd());
                fflush(stdout);
                continue; // Skip invalid rows to avoid crash
            }

            iIndex iStart = vRowPtr[static_cast<std::size_t>(iLocalRow)];
            iIndex iEnd = vRowPtr[static_cast<std::size_t>(iLocalRow + 1)];

            // Binary search within the row's column indices
            auto itBegin = vColInd.begin() + iStart;
            auto itEnd = vColInd.begin() + iEnd;
            auto it = std::lower_bound(itBegin, itEnd, iGlobalCol);

            if (it != itEnd && *it == iGlobalCol) {
                std::size_t idx = static_cast<std::size_t>(it - vColInd.begin());
                vLocalVal[idx] += val;
            } else {
                // printf("[WARN] CSRComm::Assemble: Received entry (%lld, %lld) but column not found in local structure.\n", (long long)iGlobalRow, (long long)iGlobalCol);
            }
        }
    }
};

// Type aliases for common scalar types
using cCSRCommF = cCSRComm<float>;
using cCSRCommD = cCSRComm<double>;
using cCSRCommCF = cCSRComm<std::complex<float>>;
using cCSRCommCD = cCSRComm<std::complex<double>>;

}
