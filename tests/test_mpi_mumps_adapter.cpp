// Clean minimal MUMPS distributed test harness.
#include <mpi.h>
extern "C" {
#include <dmumps_c.h>
}
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int iRank = 0, iSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &iRank);
    MPI_Comm_size(MPI_COMM_WORLD, &iSize);
#ifndef CSR4MPI_USE_MUMPS
    if (iRank == 0)
        fprintf(stderr, "MUMPS not compiled; rebuild with CSR4MPI_ENABLE_MUMPS=ON.\n");
    MPI_Finalize();
    return 0;
#else
    const int iN = (getenv("CSR4MPI_MUMPS_N") ? atoi(getenv("CSR4MPI_MUMPS_N")) : 50);
    // Simple block distribution like PETSc style
    int iRowsBase = 0;
    for (int r = 0; r < iRank; ++r) {
        iRowsBase += iN / iSize + ((iN % iSize) > r ? 1 : 0);
    }
    int iLocalRows = iN / iSize + ((iN % iSize) > iRank ? 1 : 0);
    int iRowBeg = iRowsBase;
    int iRowEnd = iRowsBase + iLocalRows;
    std::vector<int> vIRN;
    std::vector<int> vJCN;
    std::vector<double> vA;
    vIRN.reserve((size_t)iLocalRows * 3);
    vJCN.reserve((size_t)iLocalRows * 3);
    vA.reserve((size_t)iLocalRows * 3);
    for (int g = iRowBeg; g < iRowEnd; ++g) {
        // diagonal
        vIRN.push_back(g + 1);
        vJCN.push_back(g + 1);
        vA.push_back(4.0);
        if (g > 0) {
            vIRN.push_back(g + 1);
            vJCN.push_back(g);
            vA.push_back(1.0);
        }
        if (g + 1 < iN) {
            vIRN.push_back(g + 1);
            vJCN.push_back(g + 2);
            vA.push_back(1.0);
        }
    }
    if (iRank == 0)
        fprintf(stderr, "[MUMPS] ranks=%d N=%d local_nz_rank0=%zu\n", iSize, iN, vA.size());
    DMUMPS_STRUC_C sId;
    std::memset(&sId, 0, sizeof(sId));
    sId.par = 1;
    sId.comm_fortran = MPI_Comm_c2f(MPI_COMM_WORLD);
    sId.job = -1;
    dmumps_c(&sId);
    sId.n = iN;
    sId.sym = 0; // general
    sId.icntl[0] = 0;
    sId.icntl[1] = 0;
    sId.icntl[2] = 0;
    sId.icntl[3] = 0; // silence
    sId.icntl[4] = 0; // assembled
    sId.icntl[17] = 3; // distributed entries
    sId.nz_loc = (MUMPS_INT)vIRN.size();
    sId.irn_loc = vIRN.data();
    sId.jcn_loc = vJCN.data();
#if CSR4MPI_VALUE_TYPE == 1
    sId.a_loc = vA.data();
#else
    if (iRank == 0)
        fprintf(stderr, "This test requires double (CSR4MPI_VALUE_TYPE=1).\n");
    sId.job = -2;
    dmumps_c(&sId);
    MPI_Finalize();
    return 0;
#endif
    // Analyze
    sId.job = 1;
    dmumps_c(&sId);
    if (iRank == 0)
        fprintf(stderr, "[MUMPS] analyze info=%d %d\n", sId.info[0], sId.info[1]);
    if (sId.info[0] != 0) {
        MPI_Abort(MPI_COMM_WORLD, sId.info[0]);
    }
    // Factor
    sId.job = 2;
    dmumps_c(&sId);
    if (iRank == 0)
        fprintf(stderr, "[MUMPS] factor info=%d %d\n", sId.info[0], sId.info[1]);
    if (sId.info[0] != 0) {
        MPI_Abort(MPI_COMM_WORLD, sId.info[0]);
    }
    // RHS & solve
    std::vector<double> vRhs;
    if (iRank == 0) {
        vRhs.resize(iN);
        for (int i = 0; i < iN; ++i)
            vRhs[i] = (double)(i + 1);
        sId.nrhs = 1;
        sId.lrhs = iN;
        sId.rhs = vRhs.data();
    } else {
        sId.nrhs = 1;
        sId.lrhs = 0;
        sId.rhs = nullptr;
    }
    sId.job = 3;
    dmumps_c(&sId);
    if (iRank == 0)
        fprintf(stderr, "[MUMPS] solve info=%d %d\n", sId.info[0], sId.info[1]);
    if (sId.info[0] != 0) {
        MPI_Abort(MPI_COMM_WORLD, sId.info[0]);
    }
    if (iRank == 0) {
        // Quick residual check
        std::vector<double> vSol = vRhs; // overwritten by solution
        double dErr = 0.0, dNorm = 0.0;
        for (int r = 0; r < iN; ++r) {
            double Ax = 4.0 * vSol[r];
            if (r > 0)
                Ax += 1.0 * vSol[r - 1];
            if (r + 1 < iN)
                Ax += 1.0 * vSol[r + 1];
            double b = (double)(r + 1);
            double d = Ax - b;
            dErr += d * d;
            dNorm += b * b;
        }
        fprintf(stderr, "[MUMPS] rel_res=%e\n", std::sqrt(dErr / (dNorm + 1e-16)));
    }
    sId.job = -2;
    dmumps_c(&sId);
    MPI_Finalize();
    return 0;
#endif
}
