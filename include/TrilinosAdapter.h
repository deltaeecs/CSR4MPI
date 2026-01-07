#pragma once

#include "CSRMatrix.h"
#include <mpi.h>
#include <vector>

#ifdef DCNOVA_USE_TRILINOS
#include <Epetra_ConfigDefs.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Teuchos_RCP.hpp>
#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#else
#include <Epetra_SerialComm.h>
#endif
#endif

namespace csr4mpi {

template <typename Scalar>
class cTrilinosAdapter {
public:
#ifdef DCNOVA_USE_TRILINOS
    static Teuchos::RCP<Epetra_CrsMatrix> CreateMapAndMatrix(const cCSRMatrix<Scalar>& A, const Epetra_Comm& comm)
    {
        // 1. Create Map
        // Epetra uses `int` for global IDs (unless 64-bit map is enabled, which is non-standard in older Epetra).
        // We assume 32-bit for now.
        int numGlobalElements = (int)A.iGlobalRowEnd(); // Assuming square
        // Actually A.iGlobalRowEnd() is just the end of local block? No, usually total if partitioned 1D end-to-end?
        // Wait, A.iGlobalRowEnd() in DcNova PhysicalModel is `(iTotal * (iRank + 1)) / iSize`.
        // So `iGlobalRowEnd` of the last rank gives total rows.

        // Better: We know local rows.
        int numMyElements = (int)(A.iGlobalRowEnd() - A.iGlobalRowBegin());
        long long globalStart = A.iGlobalRowBegin();

        // Construct Map
        // Epetra_Map(GlobalNumElements, NumMyElements, IndexBase, Comm)
        // If GlobalNumElements is -1, it computes sum of NumMyElements.
        Epetra_Map Map(-1, numMyElements, 0, comm);

        // 2. Create Matrix
        // We use View if possible, but RowPtr structure is different (Epetra stores entries per row separately usually? OR compressed).
        // Epetra_CrsMatrix has a specific constructor for CRS inputs?
        // Epetra_CrsMatrix (Copy, RowMap, ColMap, ...)
        // There is no direct "pointer adoption" constructor for standard CSR arrays in Epetra_CrsMatrix without copying into its internal structure.
        // So we must Insert.

        Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy, Map, 0));

        const auto& rowPtr = A.vRowPtr();
        const auto& colInd = A.vColInd();
        const auto& values = A.vValues();

        std::vector<int> col_indices_buffer;
        col_indices_buffer.reserve(100);

        for (int i = 0; i < numMyElements; ++i) {
            int GlobalRow = (int)(globalStart + i);
            iSize start = rowPtr[i];
            iSize end = rowPtr[i + 1];
            int numEntries = (int)(end - start);

            if (numEntries > 0) {
                if (col_indices_buffer.size() < (size_t)numEntries)
                    col_indices_buffer.resize(numEntries);

                // Convert indices to int
                for (int k = 0; k < numEntries; ++k) {
                    col_indices_buffer[k] = (int)colInd[start + k];
                }

                // Values
                // Epetra Recieves double*
                const double* val_ptr = (const double*)&values[start];

                matrix->InsertGlobalValues(GlobalRow, numEntries, val_ptr, col_indices_buffer.data());
            }
        }

        matrix->FillComplete();
        return matrix;
    }
#endif
};

}
