#pragma once

#include "Global.h"
#include "CSRMatrix.h"
#include "CommPattern.h"
#include <mpi.h>
#include <vector>

namespace csr4mpi {

namespace detail {
    // Helper to get MPI datatype for a scalar type
    template <typename Scalar>
    inline MPI_Datatype GetMPIDatatype(MPI_Datatype* pCreatedType = nullptr) {
        if constexpr (std::is_same_v<Scalar, float>) {
            return MPI_FLOAT;
        } else if constexpr (std::is_same_v<Scalar, double>) {
            return MPI_DOUBLE;
        } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
#ifdef MPI_C_FLOAT_COMPLEX
            return MPI_C_FLOAT_COMPLEX;
#else
            MPI_Datatype dt;
            MPI_Type_contiguous(2, MPI_FLOAT, &dt);
            MPI_Type_commit(&dt);
            if (pCreatedType) *pCreatedType = dt;
            return dt;
#endif
        } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
#ifdef MPI_C_DOUBLE_COMPLEX
            return MPI_C_DOUBLE_COMPLEX;
#else
            MPI_Datatype dt;
            MPI_Type_contiguous(2, MPI_DOUBLE, &dt);
            MPI_Type_commit(&dt);
            if (pCreatedType) *pCreatedType = dt;
            return dt;
#endif
        }
    }
}

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

        int iTotalSend = 0;
        int iTotalRecv = 0;
        for (int i = 0; i < iWorldSize; ++i) {
            iTotalSend += vSendCounts[static_cast<std::size_t>(i)];
            iTotalRecv += vRecvCounts[static_cast<std::size_t>(i)];
        }

        struct cTriplet {
            iIndex m_iRow;
            iIndex m_iCol;
            Scalar m_vVal;
        };

        std::vector<cTriplet> vSendBuf(static_cast<std::size_t>(iTotalSend));
        std::vector<cTriplet> vRecvBuf(static_cast<std::size_t>(iTotalRecv));

        // Fill send buffer in rank-major order.
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
                vSendBuf[static_cast<std::size_t>(iPos)].m_iRow = vSendRows[static_cast<std::size_t>(i)];
                vSendBuf[static_cast<std::size_t>(iPos)].m_iCol = vSendCols[static_cast<std::size_t>(i)];
                vSendBuf[static_cast<std::size_t>(iPos)].m_vVal = vValues[static_cast<std::size_t>(i)];
            }
        }

        // Build displacement arrays for Alltoallv.
        std::vector<int> vSendDispls(iWorldSize, 0);
        std::vector<int> vRecvDispls(iWorldSize, 0);

        for (int i = 1; i < iWorldSize; ++i) {
            vSendDispls[static_cast<std::size_t>(i)] = vSendDispls[static_cast<std::size_t>(i - 1)] + vSendCounts[static_cast<std::size_t>(i - 1)];
            vRecvDispls[static_cast<std::size_t>(i)] = vRecvDispls[static_cast<std::size_t>(i - 1)] + vRecvCounts[static_cast<std::size_t>(i - 1)];
        }

        // Define MPI datatype for cTriplet with correct scalar handling.
        MPI_Datatype tTripletType;
        MPI_Datatype tCreatedScalarType = MPI_DATATYPE_NULL;
        MPI_Datatype tScalarType = detail::GetMPIDatatype<Scalar>(&tCreatedScalarType);

        {
            cTriplet cDummy;
            int iBlockLengths[3] = { 1, 1, 1 };
            MPI_Aint vDispls[3];
            MPI_Aint iBase;

            MPI_Get_address(&cDummy, &iBase);
            MPI_Get_address(&cDummy.m_iRow, &vDispls[0]);
            MPI_Get_address(&cDummy.m_iCol, &vDispls[1]);
            MPI_Get_address(&cDummy.m_vVal, &vDispls[2]);

            vDispls[0] -= iBase;
            vDispls[1] -= iBase;
            vDispls[2] -= iBase;

            MPI_Datatype vTypes[3] = { MPI_LONG_LONG, MPI_LONG_LONG, tScalarType };
            MPI_Type_create_struct(3, iBlockLengths, vDispls, vTypes, &tTripletType);
            MPI_Type_commit(&tTripletType);
        }

        MPI_Alltoallv(vSendBuf.data(),
            vSendCounts.data(),
            vSendDispls.data(),
            tTripletType,
            vRecvBuf.data(),
            vRecvCounts.data(),
            vRecvDispls.data(),
            tTripletType,
            cComm);

        MPI_Type_free(&tTripletType);
        if (tCreatedScalarType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tCreatedScalarType);
        }

        // Accumulate received contributions into local CSR matrix.
        for (const cTriplet& cT : vRecvBuf) {
            iIndex iGlobalRow = cT.m_iRow;
            iIndex iGlobalCol = cT.m_iCol;

            iIndex iLocalRow = static_cast<iIndex>(iGlobalRow - cLocal.iGlobalRowBegin());
            iIndex iStart = vRowPtr[static_cast<std::size_t>(iLocalRow)];
            iIndex iEnd = vRowPtr[static_cast<std::size_t>(iLocalRow + 1)];

            for (iIndex i = iStart; i < iEnd; ++i) {
                if (vColInd[static_cast<std::size_t>(i)] == iGlobalCol) {
                    vLocalVal[static_cast<std::size_t>(i)] += cT.m_vVal;
                    break;
                }
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
