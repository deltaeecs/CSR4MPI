#include "CSRMatrix.h"
#include "Distribution.h"
#include "MatrixMarketLoader.h"
#include "Operations.h"
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

using namespace csr4mpi;

static cCSRMatrix ExtractLocal(const cCSRMatrix& full, const cRowDistribution& dist)
{
    iIndex rBeg = dist.iGlobalRowBegin();
    iIndex rEnd = dist.iGlobalRowEnd();
    iSize localRows = rEnd - rBeg;
    const auto& FRP = full.vRowPtr();
    const auto& FC = full.vColInd();
    const auto& FV = full.vValues();
    std::vector<iIndex> lRP(localRows + 1, 0);
    std::vector<iIndex> lC;
    std::vector<vScalar> lV;
    for (iSize lr = 0; lr < localRows; ++lr) {
        iIndex g = rBeg + lr;
        for (iIndex k = FRP[(size_t)g]; k < FRP[(size_t)g + 1]; ++k) {
            lC.push_back(FC[(size_t)k]);
            lV.push_back(FV[(size_t)k]);
        }
        lRP[(size_t)lr + 1] = (iIndex)lC.size();
    }
    cCSRMatrix local(rBeg, rEnd, full.iGlobalColCount(), lRP, lC, lV, full.eSymmetry());
    local.AttachDistribution(std::make_shared<cRowDistribution>(dist));
    return local;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    std::string matrixFile = (argc > 1 ? argv[1] : "bcsstk13.mtx");
    std::string path = std::string(CSR4MPI_SOURCE_DIR) + "/tests/matrices/" + matrixFile;
    std::vector<iIndex> gRP, gCI;
    std::vector<vScalar> gV;
    iIndex gRows = 0, gCols = 0;
    if (!LoadMatrixMarket(path, gRP, gCI, gV, gRows, gCols)) {
        if (rank == 0)
            std::cout << "Matrix not found: " << path << "\n";
        MPI_Finalize();
        return 0;
    }
    auto dist = cRowDistribution::CreateBlockDistribution(gRows, worldSize, rank);
    cCSRMatrix local = ExtractLocal(cCSRMatrix(0, gRows, gCols, gRP, gCI, gV), dist);
    // Build local slice of x (random deterministic)
    const auto& offs = dist.vRowOffsets();
    iIndex cBeg = offs[(size_t)rank];
    iIndex cEnd = offs[(size_t)rank + 1];
    std::vector<vScalar> xLocal;
    xLocal.reserve(static_cast<size_t>(cEnd - cBeg));
    for (iIndex g = cBeg; g < cEnd; ++g)
        xLocal.push_back(static_cast<vScalar>(1));
    std::vector<vScalar> yLocal;
    int iters = (argc > 2 ? std::stoi(argv[2]) : 10);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
        SpMV(local, xLocal, yLocal);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double localTime = (t1 - t0) / iters;
    double maxTime = 0, minTime = 0, avgTime = 0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime, &avgTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        avgTime /= worldSize;
        const double nnz = static_cast<double>(gCI.size());
        const double gflops = (2.0 * nnz) / avgTime / 1e9; // 1 mul + 1 add per nz
        std::cout << "SpMV benchmark (" << matrixFile << ", iters=" << iters << "): avg=" << avgTime
                  << " s max=" << maxTime << " s min=" << minTime << " s nnz=" << (long long)gCI.size()
                  << " GFLOPs=" << gflops << "\n";
    }
    MPI_Finalize();
    return 0;
}
