/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-11-30 04:30:36
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-11-30 14:13:39
 * @FilePath: \CSR4MPI\src\Operations.cpp
 * @Description:
 *
 * Copyright (c) 2025 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved.
 */
#include "Operations.h"
#include "Distribution.h"
#include <mpi.h>
#include <stdexcept>
#ifdef CSR4MPI_USE_OPENMP
#include <omp.h>
#endif
#include <unordered_map>

namespace csr4mpi {
void SpMV(const cCSRMatrix& A, const std::vector<vScalar>& x, std::vector<vScalar>& y)
{
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
    std::vector<vScalar> xGlobalBuf;
    const std::vector<vScalar>* pXFull = &x;
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
        MPI_Datatype dt;
        if constexpr (std::is_same_v<vScalar, float>)
            dt = MPI_FLOAT;
        else if constexpr (std::is_same_v<vScalar, double>)
            dt = MPI_DOUBLE;
        else if constexpr (std::is_same_v<vScalar, std::complex<float>>) {
#ifdef MPI_C_FLOAT_COMPLEX
            dt = MPI_C_FLOAT_COMPLEX;
#else
            MPI_Type_contiguous(2, MPI_FLOAT, &dt);
            MPI_Type_commit(&dt);
#endif
        } else if constexpr (std::is_same_v<vScalar, std::complex<double>>) {
#ifdef MPI_C_DOUBLE_COMPLEX
            dt = MPI_C_DOUBLE_COMPLEX;
#else
            MPI_Type_contiguous(2, MPI_DOUBLE, &dt);
            MPI_Type_commit(&dt);
#endif
        } else {
            throw std::runtime_error("SpMV: unsupported scalar type");
        }
        MPI_Allgatherv(x.data(), counts[rank], dt, xGlobalBuf.data(), counts.data(), displs.data(), dt, MPI_COMM_WORLD);
#if !defined(MPI_C_FLOAT_COMPLEX)
        if constexpr (std::is_same_v<vScalar, std::complex<float>>) {
            if (dt != MPI_C_FLOAT_COMPLEX)
                MPI_Type_free(&dt);
        }
#endif
#if !defined(MPI_C_DOUBLE_COMPLEX)
        if constexpr (std::is_same_v<vScalar, std::complex<double>>) {
            if (dt != MPI_C_DOUBLE_COMPLEX)
                MPI_Type_free(&dt);
        }
#endif
        pXFull = &xGlobalBuf;
    }
    const std::vector<vScalar>& xFull = *pXFull;
    y.assign(static_cast<size_t>(localRows), static_cast<vScalar>(0));
    iSize rowBase = A.iGlobalRowBegin();
    bool symTriangular = (eSymm == eSymmLower || eSymm == eSymmUpper);
    if (!symTriangular || eSymm == eSymmFull) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<size_t>(r)];
            iIndex end = rowPtr[static_cast<size_t>(r + 1)];
            vScalar sum = static_cast<vScalar>(0);
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
    std::vector<vScalar> sendVals;
    std::vector<int> sendCounts(hasRemote ? worldSize : 1, 0);
    // 主循环：并行仅在无远程通信时启用
    if (!hasRemote) {
#ifdef CSR4MPI_USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<size_t>(r)];
            iIndex end = rowPtr[static_cast<size_t>(r + 1)];
            vScalar diagAcc = static_cast<vScalar>(0);
            iIndex gRow = r + rowBase;
            struct cUpd {
                iIndex idx;
                vScalar val;
            };
            std::vector<cUpd> vAdds;
            vAdds.reserve(static_cast<size_t>(end - start));
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<size_t>(k)];
                vScalar a = values[static_cast<size_t>(k)];
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
        vScalar diagAcc = static_cast<vScalar>(0);
        iIndex gRow = r + rowBase;
        for (iIndex k = start; k < end; ++k) {
            iIndex c = colInd[static_cast<size_t>(k)];
            vScalar a = values[static_cast<size_t>(k)];
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
            vScalar val;
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
            if constexpr (std::is_same_v<vScalar, float>)
                types[1] = MPI_FLOAT;
            else if constexpr (std::is_same_v<vScalar, double>)
                types[1] = MPI_DOUBLE;
            else if constexpr (std::is_same_v<vScalar, std::complex<float>>) {
#ifdef MPI_C_FLOAT_COMPLEX
                types[1] = MPI_C_FLOAT_COMPLEX;
#else
                MPI_Type_contiguous(2, MPI_FLOAT, &types[1]);
                MPI_Type_commit(&types[1]);
#endif
            } else if constexpr (std::is_same_v<vScalar, std::complex<double>>) {
#ifdef MPI_C_DOUBLE_COMPLEX
                types[1] = MPI_C_DOUBLE_COMPLEX;
#else
                MPI_Type_contiguous(2, MPI_DOUBLE, &types[1]);
                MPI_Type_commit(&types[1]);
#endif
            } else {
                throw std::runtime_error("SpMV symmetric remote: unsupported scalar type");
            }
            MPI_Type_create_struct(2, bl, disp, types, &tPairType);
            MPI_Type_commit(&tPairType);
        }
        MPI_Alltoallv(sendBuf.data(), sendCounts.data(), sendDispls.data(), tPairType,
            recvBuf.data(), recvCounts.data(), recvDispls.data(), tPairType,
            MPI_COMM_WORLD);
        MPI_Type_free(&tPairType);
        iSize rowBase2 = A.iGlobalRowBegin();
        for (const cPair& pr : recvBuf) {
            iIndex local = pr.row - rowBase2;
            if (local >= 0 && local < localRows)
                y[static_cast<size_t>(local)] += pr.val;
        }
    }
}

void SpMV(const cCSRMatrix& A, std::vector<vScalar>& x)
{
    std::vector<vScalar> y;
    SpMV(A, x, y);
    x.swap(y); // x now holds local result
}

void SpMM(const cCSRMatrix& A, const std::vector<vScalar>& X, iSize nCols, std::vector<vScalar>& Y)
{
    iSize globalCols = A.iGlobalColCount();
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    if (X.size() != static_cast<std::size_t>(globalCols * nCols)) {
        throw std::runtime_error("SpMM: input dense matrix size mismatch");
    }
    Y.assign(static_cast<std::size_t>(localRows * nCols), static_cast<vScalar>(0));
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
                vScalar a = values[static_cast<std::size_t>(k)];
                const vScalar* xColBase = &X[static_cast<std::size_t>(c)];
                vScalar* yRowBase = &Y[static_cast<std::size_t>(r)];
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
                    std::vector<vScalar> vals;
                };
                std::vector<cUpd> vAdds;
                for (iIndex k = start; k < end; ++k) {
                    iIndex c = colInd[static_cast<std::size_t>(k)];
                    vScalar a = values[static_cast<std::size_t>(k)];
                    bool lowerCond = (gRow >= c);
                    if (c == gRow) {
                        const vScalar* xColBase = &X[static_cast<std::size_t>(c)];
                        vScalar* yRowBase = &Y[static_cast<std::size_t>(r)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase[static_cast<std::size_t>(j * localRows)] += a * xColBase[static_cast<std::size_t>(j * globalCols)];
                        continue;
                    }
                    if ((isLowerStored && lowerCond) || (isUpperStored && !lowerCond)) {
                        const vScalar* xColBase_c = &X[static_cast<std::size_t>(c)];
                        vScalar* yRowBase_r = &Y[static_cast<std::size_t>(r)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase_r[static_cast<std::size_t>(j * localRows)] += a * xColBase_c[static_cast<std::size_t>(j * globalCols)];
                        iIndex maybeLocal = c - rowBase;
                        if (maybeLocal >= 0 && maybeLocal < localRows) {
                            const vScalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
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
                        vScalar* yRowBase_c = &Y[static_cast<std::size_t>(u.idx)];
                        for (iSize j = 0; j < nCols; ++j)
                            yRowBase_c[static_cast<std::size_t>(j * localRows)] += u.vals[static_cast<size_t>(j)];
                    }
                }
            }
            return;
        }
        // 远程路径：行块聚合，针对每个远程镜像行收集长度为 nCols 的向量，减少元数据
        std::unordered_map<iIndex, std::vector<vScalar>> mRemoteBlocks; // row -> contributions[nCols]
        for (iSize r = 0; r < localRows; ++r) {
            iIndex start = rowPtr[static_cast<std::size_t>(r)];
            iIndex end = rowPtr[static_cast<std::size_t>(r + 1)];
            iIndex gRow = r + rowBase;
            for (iIndex k = start; k < end; ++k) {
                iIndex c = colInd[static_cast<std::size_t>(k)];
                vScalar a = values[static_cast<std::size_t>(k)];
                bool lowerCond = (gRow >= c);
                if (c == gRow) {
                    const vScalar* xColBase = &X[static_cast<std::size_t>(c)];
                    vScalar* yRowBase = &Y[static_cast<std::size_t>(r)];
                    for (iSize j = 0; j < nCols; ++j)
                        yRowBase[static_cast<std::size_t>(j * localRows)] += a * xColBase[static_cast<std::size_t>(j * globalCols)];
                    continue;
                }
                if ((isLowerStored && lowerCond) || (isUpperStored && !lowerCond)) {
                    // y_r += a * x_c
                    const vScalar* xColBase_c = &X[static_cast<std::size_t>(c)];
                    vScalar* yRowBase_r = &Y[static_cast<std::size_t>(r)];
                    for (iSize j = 0; j < nCols; ++j)
                        yRowBase_r[static_cast<std::size_t>(j * localRows)] += a * xColBase_c[static_cast<std::size_t>(j * globalCols)];
                    int owner = pDist->iOwnerRank(c);
                    if (owner == rank) {
                        iIndex maybeLocal = c - rowBase;
                        if (maybeLocal >= 0 && maybeLocal < localRows) {
                            const vScalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
                            vScalar* yRowBase_c = &Y[static_cast<std::size_t>(maybeLocal)];
                            for (iSize j = 0; j < nCols; ++j)
                                yRowBase_c[static_cast<std::size_t>(j * localRows)] += a * xColBase_r[static_cast<std::size_t>(j * globalCols)];
                        }
                    } else {
                        const vScalar* xColBase_r = &X[static_cast<std::size_t>(gRow)];
                        auto& vecRef = mRemoteBlocks[c];
                        if (vecRef.empty())
                            vecRef.assign(static_cast<size_t>(nCols), static_cast<vScalar>(0));
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
        // 打包行索引与块值两个向量；行数量与块数量相同
        std::vector<iIndex> sendRowsList;
        sendRowsList.reserve(static_cast<size_t>(totalSend));
        std::vector<vScalar> sendValsList;
        sendValsList.reserve(static_cast<size_t>(totalSend) * static_cast<size_t>(nCols));
        // rank 分桶（保持顺序，不需稳定，仅需与计数一致）
        std::vector<int> bucketInserted(worldSize, 0);
        // 临时：按目标 rank 聚合，再按 rank 复制到线性数组
        std::vector<std::vector<iIndex>> perRankRows(worldSize);
        std::vector<std::vector<vScalar>> perRankVals(worldSize);
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
        // 行索引 Alltoallv
        std::vector<int> sendRowDispls(worldSize, 0), recvRowDispls(worldSize, 0);
        for (int i = 1; i < worldSize; ++i) {
            sendRowDispls[i] = sendRowDispls[i - 1] + sendRowCounts[i - 1];
            recvRowDispls[i] = recvRowDispls[i - 1] + recvCounts[i - 1];
        }
        std::vector<iIndex> recvRowsList(static_cast<size_t>(totalRecv));
        MPI_Alltoallv(sendRowsList.data(), sendRowCounts.data(), sendRowDispls.data(), MPI_LONG_LONG,
            recvRowsList.data(), recvCounts.data(), recvRowDispls.data(), MPI_LONG_LONG, MPI_COMM_WORLD);
        // 块值 Alltoallv：每个块有 nCols 标量
        std::vector<int> sendValCounts(worldSize, 0), recvValCounts(worldSize, 0), sendValDispls(worldSize, 0), recvValDispls(worldSize, 0);
        for (int i = 0; i < worldSize; ++i) {
            sendValCounts[i] = sendRowCounts[i] * static_cast<int>(nCols);
            recvValCounts[i] = recvCounts[i] * static_cast<int>(nCols);
        }
        for (int i = 1; i < worldSize; ++i) {
            sendValDispls[i] = sendValDispls[i - 1] + sendValCounts[i - 1];
            recvValDispls[i] = recvValDispls[i - 1] + recvValCounts[i - 1];
        }
        std::vector<vScalar> recvValsList(static_cast<size_t>(totalRecv) * static_cast<size_t>(nCols));
        MPI_Datatype scalarType;
        if constexpr (std::is_same_v<vScalar, float>)
            scalarType = MPI_FLOAT;
        else if constexpr (std::is_same_v<vScalar, double>)
            scalarType = MPI_DOUBLE;
        else if constexpr (std::is_same_v<vScalar, std::complex<float>>) {
#ifdef MPI_C_FLOAT_COMPLEX
            scalarType = MPI_C_FLOAT_COMPLEX;
#else
            MPI_Type_contiguous(2, MPI_FLOAT, &scalarType);
            MPI_Type_commit(&scalarType);
#endif
        } else if constexpr (std::is_same_v<vScalar, std::complex<double>>) {
#ifdef MPI_C_DOUBLE_COMPLEX
            scalarType = MPI_C_DOUBLE_COMPLEX;
#else
            MPI_Type_contiguous(2, MPI_DOUBLE, &scalarType);
            MPI_Type_commit(&scalarType);
#endif
        } else {
            throw std::runtime_error("SpMM symmetric remote block: unsupported scalar type");
        }
        MPI_Alltoallv(sendValsList.data(), sendValCounts.data(), sendValDispls.data(), scalarType,
            recvValsList.data(), recvValCounts.data(), recvValDispls.data(), scalarType,
            MPI_COMM_WORLD);
        // 应用接收块
        for (size_t blk = 0; blk < recvRowsList.size(); ++blk) {
            iIndex local = recvRowsList[blk] - rowBase;
            if (local >= 0 && local < localRows) {
                vScalar* yRowBase = &Y[static_cast<size_t>(local)];
                const vScalar* vals = &recvValsList[blk * static_cast<size_t>(nCols)];
                for (iSize j = 0; j < nCols; ++j) {
                    yRowBase[static_cast<size_t>(j * localRows)] += vals[static_cast<size_t>(j)];
                }
            }
        }
        // 释放临时派生类型（若创建了复数 contiguous 类型）
#if !defined(MPI_C_FLOAT_COMPLEX)
        if constexpr (std::is_same_v<vScalar, std::complex<float>>) {
            MPI_Type_free(&scalarType);
        }
#endif
#if !defined(MPI_C_DOUBLE_COMPLEX)
        if constexpr (std::is_same_v<vScalar, std::complex<double>>) {
            MPI_Type_free(&scalarType);
        }
#endif
    }
}

void SpMMInPlace(const cCSRMatrix& A, std::vector<vScalar>& X, iSize nCols)
{
    std::vector<vScalar> Y;
    SpMM(A, X, nCols, Y);
    X.swap(Y); // X now holds localRows x nCols result
}
}
