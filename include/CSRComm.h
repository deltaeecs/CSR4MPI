#pragma once

#include "Global.h"
#include <mpi.h>
#include <vector>

namespace csr4mpi {
class cCSRMatrix;
class cCommPattern;

class cCSRComm {
public:
    static void Assemble(cCSRMatrix& cLocal,
        const cCommPattern& cPattern,
        const std::vector<vScalar>& vValues,
        MPI_Comm cComm);
};
}
