#pragma once

#include "CSRMatrix.h"
#include <vector>

#ifdef DCNOVA_USE_PETSC
#include "petscmat.h"
#include "petscvec.h"
#endif

namespace csr4mpi {

template <typename Scalar>
class cPetscAdapter {
public:
#ifdef DCNOVA_USE_PETSC
    // Zero-copy (if possible) creation of PETSc Matrix from CSR4MPI Matrix
    static bool CreateMat(const cCSRMatrix<Scalar>& A, Mat* pMat, MPI_Comm comm)
    {
        PetscInt m = (PetscInt)(A.iGlobalRowEnd() - A.iGlobalRowBegin());
        PetscInt n = (PetscInt)A.iGlobalColCount();

        // PETSc expects PetscInt. If iIndex (int64_t) differs from PetscInt, we must copy.
        // Also PETSc MatCreateMPIAIJWithArrays takes non-const pointers for safety (though it doesn't modify if we promise).

        // We assume PETSc is using 32-bit ints by default, but check at compile time.
        // If they match, we use reinterpret_cast.

        // Note: Included petscsys.h defines PetscInt properly.
        // We rely on the compiler resolving PetscInt from the included headers.

        // Check types
        if constexpr (sizeof(PetscInt) == sizeof(iIndex)) {
            // Zero-copy path for indices
            // Warning: const_cast is unsafe if PETSc modifies them, but for assembly it usually reads.
            PetscInt* i_ptr = reinterpret_cast<PetscInt*>(const_cast<iIndex*>(A.vRowPtr().data()));
            PetscInt* j_ptr = reinterpret_cast<PetscInt*>(const_cast<iIndex*>(A.vColInd().data()));
            PetscScalar* a_ptr = reinterpret_cast<PetscScalar*>(const_cast<Scalar*>(A.vValues().data()));

            MatCreateMPIAIJWithArrays(comm, m, PETSC_DETERMINE, PETSC_DETERMINE, n, i_ptr, j_ptr, a_ptr, pMat);
        } else {
            // Copy path (Narrowing or Widening)
            // We cannot use WithArrays because we need to allocate new arrays compatible with PetscInt.
            // But WithArrays takes ownership or views? It views. So we need to manage the lifetime of the converted arrays.
            // This Adapter doesn't manage lifetime. So strict Zero-Copy is failed here.
            // Fallback: Use standard SetValues which copies internally.

            // Optimized Insert:
            MatCreate(comm, pMat);
            MatSetSizes(*pMat, m, m, PETSC_DETERMINE, n);

            MatSetFromOptions(*pMat);
            // Disable New Nonzero Allocation Error (Override options)
            MatSetOption(*pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

            MatMPIAIJSetPreallocation(*pMat, 0, NULL, 0, NULL); // Heuristic
            MatSeqAIJSetPreallocation(*pMat, 0, NULL);
            MatSetUp(*pMat);

            const auto& rowPtr = A.vRowPtr();
            const auto& colInd = A.vColInd();
            const auto& values = A.vValues();
            iSize globalRowStart = A.iGlobalRowBegin();

            // Reuse a buffer for columns
            std::vector<PetscInt> cols_buffer;
            cols_buffer.reserve(100);

            for (iSize i = 0; i < m; ++i) {
                PetscInt globalRow = (PetscInt)(globalRowStart + i);
                iSize start = rowPtr[i];
                iSize end = rowPtr[i + 1];
                PetscInt ncols = (PetscInt)(end - start);

                if (ncols > 0) {
                    if (cols_buffer.size() < (size_t)ncols)
                        cols_buffer.resize(ncols);

                    for (iSize k = 0; k < ncols; ++k) {
                        cols_buffer[k] = (PetscInt)colInd[start + k];
                    }
                    // Values can be passed directly if Scalar == PetscScalar (double usually)
                    const Scalar* v_ptr = &values[start];
                    MatSetValues(*pMat, 1, &globalRow, ncols, cols_buffer.data(), (const PetscScalar*)v_ptr, INSERT_VALUES);
                }
            }
            MatAssemblyBegin(*pMat, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(*pMat, MAT_FINAL_ASSEMBLY);
        }
        return true;
    }
#endif
};

}
