// Benchmark for cCSRMatrixBuilder::Build() on an MLFMA-like near-field
// assembly pattern (issue #7): leaf boxes of blockSize interacting with a
// (2*nbr+1)^3 stencil neighborhood as dense blocks, plus a configurable
// fraction of re-emitted duplicate triplets.
#include "CSRMatrixBuilder.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace csr4mpi;
using Scalar = double;

static double NowSec()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static std::vector<cTriplet<Scalar>> MakeNearFieldTriplets(int nBlocks, int blockSize, int nbr,
    double duplicateFraction, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> val(-1.0, 1.0);
    auto idx = [blockSize](int blk, int i) { return static_cast<iIndex>(blk) * blockSize + i; };

    std::vector<cTriplet<Scalar>> v;
    for (int bi = 0; bi < nBlocks; ++bi) {
        for (int bj = std::max(0, bi - nbr); bj <= std::min(nBlocks - 1, bi + nbr); ++bj) {
            for (int i = 0; i < blockSize; ++i)
                for (int j = 0; j < blockSize; ++j)
                    v.emplace_back(idx(bi, i), idx(bj, j), static_cast<Scalar>(val(rng)));
        }
    }
    const std::size_t nDup = static_cast<std::size_t>(static_cast<double>(v.size()) * duplicateFraction);
    std::uniform_int_distribution<std::size_t> pick(0, v.size() - 1);
    v.reserve(v.size() + nDup);
    for (std::size_t k = 0; k < nDup; ++k) {
        const auto& t = v[pick(rng)];
        v.emplace_back(t.m_iRow, t.m_iCol, static_cast<Scalar>(val(rng)));
    }
    return v;
}

int main(int argc, char** argv)
{
    int nBlocks = 1024;
    int blockSize = 16;
    int nbr = 13;
    double dupFrac = 0.10;
    int repeats = 3;
    if (argc > 1) nBlocks = std::atoi(argv[1]);
    if (argc > 2) blockSize = std::atoi(argv[2]);
    if (argc > 3) nbr = std::atoi(argv[3]);
    if (argc > 4) dupFrac = std::atof(argv[4]);
    if (argc > 5) repeats = std::atoi(argv[5]);

    const iSize n = static_cast<iSize>(nBlocks) * blockSize;
    std::printf("generating near-field triplets: %d blocks x %d, stencil +/-%d, %.0f%% dups ...\n",
        nBlocks, blockSize, nbr, dupFrac * 100.0);
    const auto triplets = MakeNearFieldTriplets(nBlocks, blockSize, nbr, dupFrac, 2024);
    std::printf("triplets: %zu\n", triplets.size());

    cCSRMatrixBuilder<Scalar> builder;
    builder.Reserve(triplets.size());
    builder.AddEntries(triplets);

    double best = 1e300;
    std::size_t nnz = 0;
    for (int r = 0; r < repeats; ++r) {
        double t0 = NowSec();
        auto m = builder.Build(0, n, n, true);
        double t1 = NowSec();
        nnz = m.vColInd().size();
        best = std::min(best, t1 - t0);
        if (!ValidateCSRMatrix(m)) {
            std::printf("ERROR: built matrix failed validation\n");
            return 1;
        }
    }
    std::printf("Build(): nnz=%zu  best=%.3fs  (%.1fM triplets/s)\n",
        nnz, best, static_cast<double>(triplets.size()) / best / 1e6);
    return 0;
}
