#include "CSRMatrix.h"
#include "Operations.h"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace csr4mpi;
using Scalar = double;

static cCSRMatrix<Scalar> BuildRandomUniform(iSize rows, iSize cols, iSize nnzPerRow, unsigned seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<iSize> colDist(0, cols - 1);
    std::uniform_real_distribution<double> valDist(0.0, 1.0);
    std::vector<iIndex> vRowPtr(rows + 1, 0);
    std::vector<iIndex> vColInd;
    vColInd.reserve(rows * nnzPerRow);
    std::vector<Scalar> vValues;
    vValues.reserve(rows * nnzPerRow);
    for (iSize r = 0; r < rows; ++r) {
        std::vector<iSize> colsChosen;
        colsChosen.reserve(nnzPerRow);
        // simple unique selection via retry (fine for small nnzPerRow)
        while (static_cast<iSize>(colsChosen.size()) < nnzPerRow) {
            iSize c = colDist(gen);
            bool dup = false;
            for (auto cc : colsChosen) {
                if (cc == c) {
                    dup = true;
                    break;
                }
            }
            if (!dup)
                colsChosen.push_back(c);
        }
        for (auto c : colsChosen) {
            vColInd.push_back(static_cast<iIndex>(c));
            double rv = valDist(gen);
            vValues.push_back(static_cast<Scalar>(rv));
        }
        vRowPtr[static_cast<size_t>(r + 1)] = static_cast<iIndex>(vColInd.size());
    }
    return cCSRMatrix<Scalar>(0, rows, cols, vRowPtr, vColInd, vValues);
}

int main(int argc, char** argv)
{
    iSize rows = 2000;
    iSize cols = 2000;
    iSize nnzPerRow = 10;
    iSize spmmCols = 8;
    int repeats = 10;
    if (argc > 1)
        rows = std::stoll(argv[1]);
    if (argc > 2)
        cols = std::stoll(argv[2]);
    if (argc > 3)
        nnzPerRow = std::stoll(argv[3]);
    if (argc > 4)
        spmmCols = std::stoll(argv[4]);
    if (argc > 5)
        repeats = std::stoi(argv[5]);

    auto A = BuildRandomUniform(rows, cols, nnzPerRow, 42);
    std::vector<Scalar> x(static_cast<size_t>(cols));
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = static_cast<Scalar>(1);

    // SpMV benchmark
    std::vector<Scalar> y;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
        SpMV(A, x, y);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dtSpMV = t1 - t0;

    // SpMM benchmark
    std::vector<Scalar> X(static_cast<size_t>(cols * spmmCols));
    for (size_t i = 0; i < X.size(); ++i)
        X[i] = static_cast<Scalar>(1);
    std::vector<Scalar> Y;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeats; ++r) {
        SpMM(A, X, spmmCols, Y);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dtSpMM = t3 - t2;

    double nnz = static_cast<double>(A.vValues().size());
    double spmvGFLOP = (nnz * 2.0 * repeats) / (dtSpMV.count() * 1e9);
    double spmmGFLOP = (nnz * 2.0 * static_cast<double>(spmmCols) * repeats) / (dtSpMM.count() * 1e9);

    std::cout << "CSR4MPI micro benchmark (rows=" << rows << ", cols=" << cols
              << ", nnz/row=" << nnzPerRow << ", repeats=" << repeats << ")\n";
    std::cout << "SpMV total time: " << dtSpMV.count() << " s, approx GFLOP/s: " << spmvGFLOP << "\n";
    std::cout << "SpMM (" << spmmCols << " cols) total time: " << dtSpMM.count() << " s, approx GFLOP/s: " << spmmGFLOP << "\n";
    return 0;
}
