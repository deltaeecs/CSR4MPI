#pragma once

#include "Global.h"
#include <vector>

namespace csr4mpi {
class cCSRMatrix;
class cRowDistribution;

class cMumpsAdapter {
public:
    // Export local owned CSR block to MUMPS COO arrays (1-based indices).
    // Minimizes reallocations by reserving exact capacity before appending.
    static void ExportLocalBlock(const cCSRMatrix& cLocal,
        const cRowDistribution& cDistribution,
        std::vector<int>& vIRN,
        std::vector<int>& vJCN,
        std::vector<vScalar>& vA);

    // Export into pre-allocated buffers provided by the caller to avoid extra allocations.
    // Returns the number of nonzeros written. Buffers must have capacity >= local nnz.
    static iSize ExportLocalBlockInto(const cCSRMatrix& cLocal,
        const cRowDistribution& cDistribution,
        int* pIRN,
        int* pJCN,
        vScalar* pA);
};
}
