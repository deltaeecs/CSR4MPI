#pragma once
#include "Global.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

namespace csr4mpi {

template <typename Scalar>
bool LoadMatrixMarket(const std::string& sPath,
    std::vector<iIndex>& vRowPtr,
    std::vector<iIndex>& vColInd,
    std::vector<Scalar>& vValues,
    iIndex& iRows,
    iIndex& iCols)
{
    static_assert(is_supported_scalar_v<Scalar>, "Scalar must be float, double, std::complex<float>, or std::complex<double>");

    struct cKey {
        iIndex r;
        iIndex c;
        bool operator==(const cKey& other) const { return r == other.r && c == other.c; }
    };
    struct cKeyHash {
        std::size_t operator()(const cKey& k) const noexcept
        {
            return std::hash<iIndex>()((k.r << 32) ^ k.c);
        }
    };

    std::ifstream fin(sPath);
    if (!fin.is_open()) {
        return false;
    }
    std::string line;
    // Header
    if (!std::getline(fin, line))
        return false;
    if (line.rfind("%%MatrixMarket", 0) != 0)
        return false;
    // Expect tokens
    bool bSymmetric = false;
    bool bGeneral = false;
    {
        std::istringstream iss(line);
        std::string mm, matrix, format, datatype, storage;
        iss >> mm >> matrix >> format >> datatype >> storage;
        if (matrix != "matrix" || format != "coordinate" || datatype != "real") {
            return false;
        }
        if (storage == "general")
            bGeneral = true;
        else if (storage == "symmetric")
            bSymmetric = true;
        else
            return false;
    }
    // Skip comments
    while (std::getline(fin, line)) {
        if (line.empty())
            continue;
        if (line[0] == '%')
            continue;
        // First non-comment is size line
        std::istringstream sz(line);
        iIndex nnz;
        sz >> iRows >> iCols >> nnz;
        if (sz.fail())
            return false;
        std::unordered_map<cKey, Scalar, cKeyHash> mAcc;
        mAcc.reserve(static_cast<size_t>(nnz));
        for (iIndex k = 0; k < nnz; ++k) {
            if (!std::getline(fin, line))
                return false;
            if (line.empty()) {
                --k;
                continue;
            }
            if (line[0] == '%') {
                --k;
                continue;
            }
            std::istringstream es(line);
            iIndex ir, ic;
            double val;
            es >> ir >> ic >> val;
            if (es.fail())
                return false;
            // 1-based to 0-based
            cKey key { ir - 1, ic - 1 };
            auto it = mAcc.find(key);
            Scalar v = static_cast<Scalar>(val);
            if (it == mAcc.end())
                mAcc.emplace(key, v);
            else
                it->second += v;
            if (bSymmetric && ir != ic) {
                // 如果输入给的是仅单边三角（常见），不自动插入另一侧，在乘法阶段展开。
            }
        }
        // Move to vector and sort
        struct cEntry {
            iIndex r;
            iIndex c;
            Scalar v;
        };
        std::vector<cEntry> entries;
        entries.reserve(mAcc.size());
        for (const auto& p : mAcc) {
            entries.push_back({ p.first.r, p.first.c, p.second });
        }
        std::sort(entries.begin(), entries.end(), [](const cEntry& a, const cEntry& b) {
            if (a.r != b.r)
                return a.r < b.r;
            return a.c < b.c;
        });
        vRowPtr.assign(static_cast<size_t>(iRows + 1), 0);
        vColInd.clear();
        vValues.clear();
        for (const auto& e : entries) {
            vRowPtr[static_cast<size_t>(e.r + 1)]++;
            vColInd.push_back(e.c);
            vValues.push_back(e.v);
        }
        // prefix sum
        for (iIndex r = 0; r < iRows; ++r) {
            vRowPtr[static_cast<size_t>(r + 1)] += vRowPtr[static_cast<size_t>(r)];
        }
        return true;
    }
    return false;
}

}
