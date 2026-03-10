#pragma once

#include "CSRMatrix.h"
#include <vector>

#ifdef DCNOVA_USE_PETSC
#include "petscmat.h"
#include "petscvec.h"
#endif

// For diagnostic logging
#ifdef DCNOVA_USE_PETSC
#include "core/Logger.h"
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

        // DIAGNOSTIC: Log matrix dimensions
        DC_INFO("PetscAdapter::CreateMat() - Local rows={}, Global cols={}", m, n);
        DC_INFO("PetscAdapter::CreateMat() - sizeof(PetscInt)={}, sizeof(iIndex)={}", sizeof(PetscInt), sizeof(iIndex));
        DC_INFO("PetscAdapter::CreateMat() - vRowPtr.size()={}, vColInd.size()={}, vValues.size()={}",
            A.vRowPtr().size(), A.vColInd().size(), A.vValues().size());

        // PETSc expects PetscInt. If iIndex (int64_t) differs from PetscInt, we must copy.
        // Also PETSc MatCreateMPIAIJWithArrays takes non-const pointers for safety (though it doesn't modify if we promise).

        // We assume PETSc is using 32-bit ints by default, but check at compile time.
        // If they match, we use reinterpret_cast.

        // Note: Included petscsys.h defines PetscInt properly.
        // We rely on the compiler resolving PetscInt from the included headers.

        // Check types
        if constexpr (sizeof(PetscInt) == sizeof(iIndex)) {
            // Zero-copy path for indices
            DC_INFO("PetscAdapter::CreateMat() - Using ZERO-COPY path (PetscInt == iIndex)");
            // Warning: const_cast is unsafe if PETSc modifies them, but for assembly it usually reads.
            PetscInt* i_ptr = reinterpret_cast<PetscInt*>(const_cast<iIndex*>(A.vRowPtr().data()));
            PetscInt* j_ptr = reinterpret_cast<PetscInt*>(const_cast<iIndex*>(A.vColInd().data()));
            PetscScalar* a_ptr = reinterpret_cast<PetscScalar*>(const_cast<Scalar*>(A.vValues().data()));

            // Validate last rowPtr == nnz
            iIndex iLastRowPtr = A.vRowPtr()[A.vRowPtr().size() - 1];
            DC_INFO("PetscAdapter::CreateMat() - Last vRowPtr[{}]={}, Expected NNZ={}",
                A.vRowPtr().size() - 1, iLastRowPtr, A.vColInd().size());
            if (iLastRowPtr != (iIndex)A.vColInd().size()) {
                DC_ERROR("PetscAdapter::CreateMat() - CRITICAL: Last rowPtr != ColInd.size()! {} != {}",
                    iLastRowPtr, A.vColInd().size());
            }

            // FIXED: MatCreateMPIAIJWithArrays signature is (comm, m, n, M, N, i, j, a, mat)
            //   m = local rows, n = local cols, M = global rows, N = global cols
            //   For square matrices: n should equal m (local), N = global cols
            DC_INFO("PetscAdapter::CreateMat() - Calling MatCreateMPIAIJWithArrays(comm, m={}, n={}, M=AUTO, N={}, ...)", m, m, n);
            MatCreateMPIAIJWithArrays(comm, m, m, PETSC_DETERMINE, n, i_ptr, j_ptr, a_ptr, pMat);

            DC_INFO("PetscAdapter::CreateMat() - MatCreateMPIAIJWithArrays returned, now calling MatAssembly...");
            // CRITICAL FIX: MatCreateMPIAIJWithArrays requires Assembly before use
            MatAssemblyBegin(*pMat, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(*pMat, MAT_FINAL_ASSEMBLY);
            DC_INFO("PetscAdapter::CreateMat() - MatAssembly complete.");
        } else {
            DC_INFO("PetscAdapter::CreateMat() - Using COPY path (PetscInt != iIndex)");
            // Copy path (Narrowing or Widening)
            // Fix: Calculate exact preallocation to avoid MAT_NEW_NONZERO_ALLOCATION_ERR

            MatCreate(comm, pMat);
            MatSetSizes(*pMat, m, m, PETSC_DETERMINE, n);
            MatSetFromOptions(*pMat);

            // Calculate Preallocation
            const auto& rowPtr = A.vRowPtr();
            const auto& colInd = A.vColInd();
            const auto& values = A.vValues();
            iSize globalRowStart = A.iGlobalRowBegin();
            iSize globalRowEnd = A.iGlobalRowEnd();

            std::vector<PetscInt> d_nnz(m, 0);
            std::vector<PetscInt> o_nnz(m, 0);

            for (iSize i = 0; i < m; ++i) {
                iSize start = rowPtr[i];
                iSize end = rowPtr[i + 1];
                for (iSize k = start; k < end; ++k) {
                    iIndex c = colInd[k];
                    if (c >= globalRowStart && c < globalRowEnd) {
                        d_nnz[i]++;
                    } else {
                        o_nnz[i]++;
                    }
                }
            }

            // Set Preallocation (Covers both Seq and MPI types)
            MatMPIAIJSetPreallocation(*pMat, 0, d_nnz.data(), 0, o_nnz.data());

            // For Serial/SeqAIJ, d_nnz holds all entries (o_nnz is 0 if running serial), so this is safe:
            MatSeqAIJSetPreallocation(*pMat, 0, d_nnz.data());

            MatSetUp(*pMat);
            MatSetOption(*pMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

            // Fill Values
            std::vector<PetscInt> cols_buffer;
            cols_buffer.reserve(128);

            DC_INFO("PetscAdapter::CreateMat() - Starting COPY path: {} rows, {} nnz", m, colInd.size());
            const iSize iProgressInterval = 10000;
            for (iSize i = 0; i < m; ++i) {
                // Progress logging
                if (i % iProgressInterval == 0 && i > 0) {
                    DC_INFO("PetscAdapter::CreateMat() - Progress: {}/{} rows filled ({:.1f}%)",
                        i, m, 100.0 * i / m);
                }

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
            DC_INFO("PetscAdapter::CreateMat() - All {} rows filled, starting assembly...", m);
            MatAssemblyBegin(*pMat, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(*pMat, MAT_FINAL_ASSEMBLY);
            DC_INFO("PetscAdapter::CreateMat() - Assembly complete");
        }
        return true;
    }
#endif
};

}
