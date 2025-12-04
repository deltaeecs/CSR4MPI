#pragma once

#include "CSRMatrix.h"
#include "Distribution.h"
#include <mpi.h>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#ifdef CSR4MPI_USE_OPENMP
#include <omp.h>
#endif

namespace csr4mpi {

namespace detail {
    // Helper to get MPI datatype for a scalar type (defined in CSRComm.h too, but we redefine for independence)
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
void SpMV(const cCSRMatrix<Scalar>& A, const std::vector<Scalar>& x, std::vector<Scalar>& y)
{
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    const auto& rowPtr = A.vRowPtr();
    const auto& colInd = A.vColInd();
    const auto& values = A.vValues();
    eSymmStorage eSymm = A.eSymmetry();
    auto pDist = A.pDistribution();
    int rank = 0, worldSize = 1;
    if (pDist) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    }
    // 构造全局 x (必要时聚合)
    std::vector<Scalar> xGlobalBuf;
    const std::vector<Scalar>* pXFull = &x;
    if (x.size() != static_cast<size_t>(A.iGlobalColCount())) {
        if (!pDist)
            throw std::runtime_error("SpMV: input vector length mismatch and no distribution");
        if (worldSize < 2)
            throw std::runtime_error("SpMV: distribution mismatch in single-rank build");
        const auto& offsets = pDist->vRowOffsets();
        if (offsets.size() != static_cast<size_t>(worldSize + 1))
            throw std::runtime_error("SpMV: offsets size mismatch");
        if (A.iGlobalColCount() != pDist->iGlobalRowCount())
            throw std::runtime_error("SpMV: non-square distributed matrix unsupported");
        iSize expected = offsets[static_cast<size_t>(rank + 1)] - offsets[static_cast<size_t>(rank)];
        if (x.size() != static_cast<size_t>(expected))
            throw std::runtime_error("SpMV: local slice size mismatch");
        xGlobalBuf.resize(static_cast<size_t>(A.iGlobalColCount()));
        std::vector<int> counts(worldSize), displs(worldSize);
        for (int r = 0; r < worldSize; ++r)
            counts[r] = static_cast<int>(offsets[static_cast<size_t>(r + 1)] - offsets[static_cast<size_t>(r)]);
        displs[0] = 0;
        for (int r = 1; r < worldSize; ++r)
            displs[r] = displs[r - 1] + counts[r - 1];
        MPI_Datatype tCreatedType = MPI_DATATYPE_NULL;
        MPI_Datatype dt = detail::GetMPIDatatype<Scalar>(&tCreatedType);
        MPI_Allgatherv(x.data(), counts[rank], dt, xGlobalBuf.data(), counts.data(), displs.data(), dt, MPI_COMM_WORLD);
        if (tCreatedType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tCreatedType);
        }
        pXFull = &xGlobalBuf;
    }
    const std::vector<Scalar>& xFull = *pXFull;
    y.assign(static_cast<size_t>(localRows), static_cast<Scalar>(0));
    iSize rowBase = A.iGlobalRowBegin();
    bool symTriangular = (eSymm == eSymmLower || eSymm == eSymmUpper);
    if (!symTriangular || eSymm == eSymmFull) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<size_t>(r)];
            iIndex end = rowPtr[static_cast<size_t>(r + 1)];
            Scalar sum = static_cast<Scalar>(0);
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<size_t>(k)];
                sum += values[static_cast<size_t>(k)] * xFull[static_cast<size_t>(c)];
            }
            y[static_cast<size_t>(r)] = sum;
        }
        return;
    }
    // Triangular symmetric expansion
    bool hasRemote = (pDist && worldSize > 1);
    std::vector<iIndex> sendRows;
    std::vector<Scalar> sendVals;
    std::vector<int> sendCounts(hasRemote ? worldSize : 1, 0);
    // 主循环：并行仅在无远程通信时启用
    if (!hasRemote) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<size_t>(r)];
            iIndex end = rowPtr[static_cast<size_t>(r + 1)];
            Scalar diagAcc = static_cast<Scalar>(0);
            iIndex gRow = r + rowBase;
            struct cUpd {
                iIndex idx;
                Scalar val;
            };
            std::vector<cUpd> vAdds;
            vAdds.reserve(static_cast<size_t>(end - start));
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<size_t>(k)];
                Scalar a = values[static_cast<size_t>(k)];
                if (c == gRow) {
                    diagAcc += a * xFull[static_cast<size_t>(c)];
                    continue;
                }
                bool lowerCond = (gRow >= c);
                if ((eSymm == eSymmLower && lowerCond) || (eSymm == eSymmUpper && !lowerCond)) {
                    diagAcc += a * xFull[static_cast<size_t>(c)];
                    iIndex maybeLocal = c - rowBase;
                    if (maybeLocal >= 0 && maybeLocal < localRows)
                        vAdds.push_back({ maybeLocal, a * xFull[static_cast<size_t>(gRow)] });
                }
            }
            if (!vAdds.empty()) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp critical
#endif
                for (const auto& u : vAdds)
                    y[static_cast<size_t>(u.idx)] += u.val;
            }
            y[static_cast<size_t>(r)] += diagAcc;
        }
        return;
    }
    // 有远程通信时，顺序执行收集远程对称展开贡献
    for (iSize r = 0; r < localRows; ++r) {
        iIndex start = rowPtr[static_cast<size_t>(r)];
        iIndex end = rowPtr[static_cast<size_t>(r + 1)];
        Scalar diagAcc = static_cast<Scalar>(0);
        iIndex gRow = r + rowBase;
        for (iIndex k = start; k < end; ++k) {
            iIndex c = colInd[static_cast<size_t>(k)];
            Scalar a = values[static_cast<size_t>(k)];
            if (c == gRow) {
                diagAcc += a * xFull[static_cast<size_t>(c)];
                continue;
            }
            bool lowerCond = (gRow >= c);
            if ((eSymm == eSymmLower && lowerCond) || (eSymm == eSymmUpper && !lowerCond)) {
                diagAcc += a * xFull[static_cast<size_t>(c)];
                int owner = pDist->iOwnerRank(c);
                if (owner == rank) {
                    iIndex maybeLocal = c - rowBase;
                    if (maybeLocal >= 0 && maybeLocal < localRows)
                        y[static_cast<size_t>(maybeLocal)] += a * xFull[static_cast<size_t>(gRow)];
                } else {
                    sendRows.push_back(c);
                    sendVals.push_back(a * xFull[static_cast<size_t>(gRow)]);
                }
            }
        }
        y[static_cast<size_t>(r)] += diagAcc;
    }
    if (worldSize > 1) {
        // 统计发送计数（允许 0）
        for (size_t i = 0; i < sendRows.size(); ++i) {
            int tgt = pDist->iOwnerRank(sendRows[i]);
            sendCounts[tgt]++;
        }
        std::vector<int> recvCounts(worldSize, 0);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int totalSend = 0, totalRecv = 0;
        for (int i = 0; i < worldSize; ++i) {
            totalSend += sendCounts[i];
            totalRecv += recvCounts[i];
        }
        struct cPair {
            iIndex row;
            Scalar val;
        };
        std::vector<cPair> sendBuf(static_cast<size_t>(totalSend));
        std::vector<cPair> recvBuf(static_cast<size_t>(totalRecv));
        std::vector<int> sendDispls(worldSize, 0), recvDispls(worldSize, 0), offsets(worldSize, 0);
        for (int i = 1; i < worldSize; ++i) {
            sendDispls[i] = sendDispls[i - 1] + sendCounts[i - 1];
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
        }
        for (size_t i = 0; i < sendRows.size(); ++i) {
            int tgt = pDist->iOwnerRank(sendRows[i]);
            int pos = sendDispls[tgt] + offsets[tgt]++;
            sendBuf[static_cast<size_t>(pos)].row = sendRows[i];
            sendBuf[static_cast<size_t>(pos)].val = sendVals[i];
        }
        MPI_Datatype tPairType;
        MPI_Datatype tCreatedScalarType = MPI_DATATYPE_NULL;
        {
            cPair dummy;
            int bl[2] = { 1, 1 };
            MPI_Aint disp[2];
            MPI_Aint base;
            MPI_Get_address(&dummy, &base);
            MPI_Get_address(&dummy.row, &disp[0]);
            MPI_Get_address(&dummy.val, &disp[1]);
            disp[0] -= base;
            disp[1] -= base;
            MPI_Datatype types[2];
            types[0] = MPI_LONG_LONG;
            types[1] = detail::GetMPIDatatype<Scalar>(&tCreatedScalarType);
            MPI_Type_create_struct(2, bl, disp, types, &tPairType);
            MPI_Type_commit(&tPairType);
        }
        MPI_Alltoallv(sendBuf.data(), sendCounts.data(), sendDispls.data(), tPairType,
            recvBuf.data(), recvCounts.data(), recvDispls.data(), tPairType,
            MPI_COMM_WORLD);
        MPI_Type_free(&tPairType);
        if (tCreatedScalarType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tCreatedScalarType);
        }
        iSize rowBase2 = A.iGlobalRowBegin();
        for (const cPair& pr : recvBuf) {
            iIndex local = pr.row - rowBase2;
            if (local >= 0 && local < localRows)
                y[static_cast<size_t>(local)] += pr.val;
        }
    }
}

template <typename Scalar>
void SpMV(const cCSRMatrix<Scalar>& A, std::vector<Scalar>& x)
{
    std::vector<Scalar> y;
    SpMV(A, x, y);
    x.swap(y);
}

template <typename Scalar>
void SpMM(const cCSRMatrix<Scalar>& A, const std::vector<Scalar>& X, iSize nCols, std::vector<Scalar>& Y)
{
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

    iSize globalCols = A.iGlobalColCount();
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    if (X.size() != static_cast<std::size_t>(globalCols * nCols)) {
        throw std::runtime_error("SpMM: input dense matrix size mismatch");
    }
    Y.assign(static_cast<std::size_t>(localRows * nCols), static_cast<Scalar>(0));
    const auto& rowPtr = A.vRowPtr();
    const auto& colInd = A.vColInd();
    const auto& values = A.vValues();
    eSymmStorage eSymm = A.eSymmetry();
    iSize rowBase = A.iGlobalRowBegin();
    auto pDist = A.pDistribution();
    int rank = 0, worldSize = 1;
    if (pDist) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    }
    if (eSymm == eNone || eSymm == eSymmFull) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<std::size_t>(r)];
            iIndex end = rowPtr[static_cast<std::size_t>(r + 1)];
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<std::size_t>(k)];
                Scalar a = values[static_cast<std::size_t>(k)];
                const Scalar* xColBase = &X[static_cast<std::size_t>(c)];
                Scalar* yRowBase = &Y[static_cast<std::size_t>(r)];
                for (iSize j = 0; j < nCols; ++j) {
                    yRowBase[static_cast<std::size_t>(j * localRows)] += a * xColBase[static_cast<std::size_t>(j * globalCols)];
                }
            }
        }
    } else {
        bool isLowerStored = (eSymm == eSymmLower);
        bool isUpperStored = (eSymm == eSymmUpper);
        bool hasRemote = (pDist && worldSize > 1);
        // 如果无远程通信：批量本地更新，按行收集再一次性写入
        if (!hasRemote) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (iSize r = 0; r < localRows; ++r) {
                iIndex start = rowPtr[static_cast<std::size_t>(r)];
                iIndex end = rowPtr[static_cast<std::size_t>(r + 1)];
                iIndex gRow = r + rowBase;
                struct cUpd {
                    iIndex idx;
                    std::vector<Scalar> vals;
                };
                std::vector<cUpd> vAdds;
                for (iIndex k = start; k < end; ++k) {
                    iIndex c = colInd[static_cast<std::size_t>(k)];
                    Scalar a = values[static_cast<std::size_t>(k)];
                    bool lowerCond = (gRow >= c);
                    if (c == gRow) {
                        const Scalar* xColBase = &X[static_cast<std::size_t>(c)];
                        Scalar* yRowBase = &Y[static_cast<std::size_t>(r)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase[static_cast<std::size_t>(j * localRows)] += a * xColBase[static_cast<std::size_t>(j * globalCols)];
                        continue;
                    }
                    if ((isLowerStored && lowerCond) || (isUpperStored && !lowerCond)) {
                        const Scalar* xColBase_c = &X[static_cast<std::size_t>(c)];
                        Scalar* yRowBase_r = &Y[static_cast<std::size_t>(r)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase_r[static_cast<std::size_t>(j * localRows)] += a * xColBase_c[static_cast<std::size_t>(j * globalCols)];
                        iIndex maybeLocal = c - rowBase;
                        if (maybeLocal >= 0 && maybeLocal < localRows) {
                            const Scalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
                            cUpd upd;
                            upd.idx = maybeLocal;
                            upd.vals.resize(static_cast<size_t>(nCols));
                            for (iSize j = 0; j < nCols; ++j)
                                upd.vals[static_cast<size_t>(j)] = a * xColBase_r[static_cast<std::size_t>(j * globalCols)];
                            vAdds.push_back(std::move(upd));
                        }
                    }
                }
                if (!vAdds.empty()) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp critical
#endif
                    for (const auto& u : vAdds) {
                        Scalar* yRowBase_c = &Y[static_cast<std::size_t>(u.idx)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase_c[static_cast<std::size_t>(j * localRows)] += u.vals[static_cast<size_t>(j)];
                    }
                }
            }
            return;
        }
        // 远程路径：行块聚合，针对每个远程镜像行收集长度为 nCols 的向量，减少元数据
        std::unordered_map<iIndex, std::vector<Scalar>> mRemoteBlocks;
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<std::size_t>(r)];
            iIndex end = rowPtr[static_cast<std::size_t>(r + 1)];
            iIndex gRow = r + rowBase;
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<std::size_t>(k)];
                Scalar a = values[static_cast<std::size_t>(k)];
                bool lowerCond = (gRow >= c);
                if (c == gRow) {
                    const Scalar* xColBase = &X[static_cast<std::size_t>(c)];
                    Scalar* yRowBase = &Y[static_cast<std::size_t>(r)];
                    for (iSize j = 0; j < nCols; ++j)
                        yRowBase[static_cast<std::size_t>(j * localRows)] += a * xColBase[static_cast<std::size_t>(j * globalCols)];
                    continue;
                }
                if ((isLowerStored && lowerCond) || (isUpperStored && !lowerCond)) {
                    const Scalar* xColBase_c = &X[static_cast<std::size_t>(c)];
                    Scalar* yRowBase_r = &Y[static_cast<std::size_t>(r)];
                    for (iSize j = 0; j < nCols; ++j)
                        yRowBase_r[static_cast<std::size_t>(j * localRows)] += a * xColBase_c[static_cast<std::size_t>(j * globalCols)];
                    int owner = pDist->iOwnerRank(c);
                    if (owner == rank) {
                        iIndex maybeLocal = c - rowBase;
                        if (maybeLocal >= 0 && maybeLocal < localRows) {
                            const Scalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
                            Scalar* yRowBase_c = &Y[static_cast<std::size_t>(maybeLocal)];
                            for (iSize j = 0; j < nCols; ++j)
                                yRowBase_c[static_cast<std::size_t>(j * localRows)] += a * xColBase_r[static_cast<std::size_t>(j * globalCols)];
                        }
                    } else {
                        const Scalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
                        auto& vecRef = mRemoteBlocks[c];
                        if (vecRef.empty())
                            vecRef.assign(static_cast<size_t>(nCols), static_cast<Scalar>(0));
                        for (iSize j = 0; j < nCols; ++j)
                            vecRef[static_cast<size_t>(j)] += a * xColBase_r[static_cast<std::size_t>(j * globalCols)];
                    }
                }
            }
        }
        // 统计远程行块数量
        std::vector<int> sendRowCounts(worldSize, 0);
        for (const auto& kv : mRemoteBlocks) {
            int tgt = pDist->iOwnerRank(kv.first);
            sendRowCounts[tgt]++;
        }
        std::vector<int> recvCounts(worldSize, 0);
        MPI_Alltoall(sendRowCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        int totalSend = 0, totalRecv = 0;
        for (int i = 0; i < worldSize; ++i) {
            totalSend += sendRowCounts[i];
            totalRecv += recvCounts[i];
        }
        std::vector<iIndex> sendRowsList;
        sendRowsList.reserve(static_cast<size_t>(totalSend));
        std::vector<Scalar> sendValsList;
        sendValsList.reserve(static_cast<size_t>(totalSend) * static_cast<size_t>(nCols));
        std::vector<std::vector<iIndex>> perRankRows(worldSize);
        std::vector<std::vector<Scalar>> perRankVals(worldSize);
        for (auto& kv : mRemoteBlocks) {
            int tgt = pDist->iOwnerRank(kv.first);
            perRankRows[tgt].push_back(kv.first);
            auto& vec = kv.second;
            perRankVals[tgt].insert(perRankVals[tgt].end(), vec.begin(), vec.end());
        }
        for (int rnk = 0; rnk < worldSize; ++rnk) {
            sendRowsList.insert(sendRowsList.end(), perRankRows[rnk].begin(), perRankRows[rnk].end());
            sendValsList.insert(sendValsList.end(), perRankVals[rnk].begin(), perRankVals[rnk].end());
        }
        std::vector<int> sendRowDispls(worldSize, 0), recvRowDispls(worldSize, 0);
        for (int i = 1; i < worldSize; ++i) {
            sendRowDispls[i] = sendRowDispls[i - 1] + sendRowCounts[i - 1];
            recvRowDispls[i] = recvRowDispls[i - 1] + recvCounts[i - 1];
        }
        std::vector<iIndex> recvRowsList(static_cast<size_t>(totalRecv));
        MPI_Alltoallv(sendRowsList.data(), sendRowCounts.data(), sendRowDispls.data(), MPI_LONG_LONG,
            recvRowsList.data(), recvCounts.data(), recvRowDispls.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
        std::vector<int> sendValCounts(worldSize, 0), recvValCounts(worldSize, 0), sendValDispls(worldSize, 0), recvValDispls(worldSize, 0);
        for (int i = 0; i < worldSize; ++i) {
            sendValCounts[i] = sendRowCounts[i] * static_cast<int>(nCols);
            recvValCounts[i] = recvCounts[i] * static_cast<int>(nCols);
        }
        for (int i = 1; i < worldSize; ++i) {
            sendValDispls[i] = sendValDispls[i - 1] + sendValCounts[i - 1];
            recvValDispls[i] = recvValDispls[i - 1] + recvValCounts[i - 1];
        }
        std::vector<Scalar> recvValsList(static_cast<size_t>(totalRecv) * static_cast<size_t>(nCols));
        MPI_Datatype tCreatedScalarType = MPI_DATATYPE_NULL;
        MPI_Datatype scalarType = detail::GetMPIDatatype<Scalar>(&tCreatedScalarType);
        MPI_Alltoallv(sendValsList.data(), sendValCounts.data(), sendValDispls.data(), scalarType,
            recvValsList.data(), recvValCounts.data(), recvValDispls.data(), scalarType,
            MPI_COMM_WORLD);
        for (size_t blk = 0; blk < recvRowsList.size(); ++blk) {
            iIndex local = recvRowsList[blk] - rowBase;
            if (local >= 0 && local < localRows) {
                Scalar* yRowBase = &Y[static_cast<size_t>(local)];
                const Scalar* vals = &recvValsList[blk * static_cast<size_t>(nCols)];
                for (iSize j = 0; j < nCols; ++j) {
                    yRowBase[static_cast<size_t>(j * localRows)] += vals[static_cast<size_t>(j)];
                }
            }
        }
        if (tCreatedScalarType != MPI_DATATYPE_NULL) {
            MPI_Type_free(&tCreatedScalarType);
        }
    }
}

template <typename Scalar>
void SpMMInPlace(const cCSRMatrix<Scalar>& A, std::vector<Scalar>& X, iSize nCols)
{
    std::vector<Scalar> Y;
    SpMM(A, X, nCols, Y);
    X.swap(Y);
}

}
