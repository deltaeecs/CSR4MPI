/*
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2025-11-30 10:52:25
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2025-11-30 15:42:14
 * @FilePath: \CSR4MPI\include\DistributedOps.h
 * @Description:
 *
 * Copyright (c) 2025 by error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git, All Rights Reserved.
 */
#pragma once
#include "CSRMatrix.h"
#include "Distribution.h"
#include <mpi.h>
#include <vector>

namespace csr4mpi {
void DistributedSpMV(const cCSRMatrix& A, const cRowDistribution& dist,
    const std::vector<vScalar>& xLocalOrGlobal,
    std::vector<vScalar>& y,
    bool bReplicatedX, MPI_Comm comm);
}
