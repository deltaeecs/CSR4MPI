# AI Coding Agent Instructions for CSR4MPI

> Status: Draft – update as matrix/MPI features grow

## 项目概览

- **语言 / 领域**：C++ CSR 矩阵库，规划支持 MPI 下按连续行划分的分布式稀疏矩阵。
- **当前状态**：提供本地 CSR 矩阵类型 `cCSRMatrix`、行分布 `cRowDistribution`、通信 pattern `cCommPattern`、MPI 汇总 `cCSRComm` 和 MUMPS 适配 `cMumpsAdapter`，工程本身仅构建为静态库并附带 GoogleTest 单元测试，不再包含示例 `main` 可执行程序。

关键文件/目录：

- `CMakeLists.txt`：定义静态库 `csr4mpi`，通过 `CSR4MPI_ENABLE_MPI` 控制 MPI 支持（默认 ON），并使用 FetchContent 集成 GoogleTest，生成测试可执行 `csr4mpi_tests` 供 `ctest` 调用。
- `include/Global.h`：统一类型定义与“冷切换”标量类型（实数/复数、单/双精度），并提供索引类型别名。
- `include/CSRMatrix.h` / `src/CSRMatrix.cpp`：本地 CSR 矩阵块数据结构 `cCSRMatrix`，按连续行分配的局部视图。
- `include/Distribution.h` / `src/Distribution.cpp`：行分布描述 `cRowDistribution`，用于从全局行号映射到进程 rank 以及各 rank 拥有的行区间。
- `include/CommPattern.h` / `src/CommPattern.cpp`：通信模式 `cCommPattern`，描述当前进程需要发送到其他进程的远程条目 (globalRow, globalCol)。
- `include/CSRComm.h` / `src/CSRComm.cpp`：MPI 汇总和本地累加逻辑 `cCSRComm::Assemble`，在 owner 进程中对重叠元素进行累加。
- `include/MumpsAdapter.h` / `src/MumpsAdapter.cpp`：MUMPS 适配器 `cMumpsAdapter`，将本地 CSR 分块导出为 1-based 全局 COO 三元组。
- `tests/test_csr_matrix.cpp`：基于 GoogleTest 的单元测试，覆盖 `cCSRMatrix` 构造与本地汇总逻辑。

## 构建与运行

- 推荐在独立 `build` 目录中使用 CMake：
  - 生成工程（示例，使用 Ninja）：`cmake -G "Ninja" -DCSR4MPI_ENABLE_MPI=ON ..`
  - 构建：`cmake --build .`
- 单元测试：构建后在 `build` 下运行 `ctest`，会自动执行 `csr4mpi_tests` 中的 GoogleTest 用例。

## 类型与命名约定

- **匈牙利命名**：
  - 类名前缀 `c`（如 `cCSRMatrix`）。
  - 整型变量前缀 `i`（如 `iGlobalRowBegin`、`iSize`）。
  - 向量前缀 `v`（如 `vRowPtr`、`vValues`）。
- **全局类型**（在 `Global.h` 中定义）：
  - `vScalar`：矩阵数值标量类型，通过宏 `CSR4MPI_VALUE_TYPE` 冷切换：
    - `eFloat32Real` / `eFloat64Real` / `eFloat32Complex` / `eFloat64Complex`。
    - 默认 `eFloat64Real`（双精度实数）。
  - `iIndex`、`iSize`：索引及规模相关整型（当前为 `std::int64_t`）。
- 扩展代码时，应始终通过这些别名使用标量/索引类型，以保持数据类型可切换性。

## 现有数据结构模式

- `cCSRMatrix`：
  - 代表单个进程持有的连续行块，对应全局行范围 `[m_iGlobalRowBegin, m_iGlobalRowEnd)` 与统一的列数。
  - 成员：`m_vRowPtr`（长度 = 本地行数 + 1）、`m_vColInd`、`m_vValues`。
  - 接口仅暴露构造和只读访问函数，内部不直接包含 MPI 逻辑。
- `cRowDistribution`：
  - 描述全局行在各个 rank 上的分布，支持从全局行号查询 owner rank，以及获取当前 rank 拥有的行区间 `[iGlobalRowBegin, iGlobalRowEnd)`。
  - 提供 `CreateBlockDistribution` 工厂方法，按连续行均匀划分所有权，与 PETSc/Trilinos 风格一致。
- `cCommPattern`：
  - 在装配阶段，接受一组三元组 (globalRow, globalCol, value)，基于 `cRowDistribution` 计算这些条目的 owner rank，并在发送侧按目标进程分桶。
  - 发送侧仅存储需要跨进程发送的条目，本地条目直接由 owner 进程处理。
- `cCSRComm`：
  - 提供 `Assemble` 接口，在非 MPI 场景下对本地条目执行线性查找和累加，在启用 MPI 时使用 Alltoallv 等方式完成全局汇总，然后在 owner 进程累加重叠元素。
- `cMumpsAdapter`：
  - 将给定的本地 `cCSRMatrix` 分块和行分布导出到三个数组 (IRN, JCN, A)，使用 1-based 全局行/列索引，便于直接传入 MUMPS 等外部求解器。

## 后续实现建议（供扩展时参考）

> 注意：以下为基于当前设计的扩展方向，不是既有代码。

- 在新命名空间/模块中封装通信元数据，例如：
  - `cCommPattern`：描述从本地进程到其它进程的目标元素索引（全局行列或本地行 + 全局列），以及接收侧期望累加的位置。
  - 对重叠元素（多个源进程更新同一目标元素）应在接收方进行累加合并。
- 将 MPI 相关逻辑集中在单独源文件（如 `src/CSRComm.cpp` / `include/CSRComm.h`）。
- 在添加新模块后，请更新本文件，补充：
  - 进程间行分布/重映射策略（如何从全局行号映射到进程 rank）。
  - 发送/接收缓冲的组织方式（按目标进程分桶、按行分桶等）。
  - 典型工作流（例如“装配局部矩阵 → 构建通信模式 → 一次或多次 SpMV / 残量计算”）。

## 对 AI 代理的特别提示

- 遵守现有匈牙利命名风格，新类/函数/成员沿用相同前缀规则。
- 新增 MPI 功能时：
  - 设计接口时优先以“本地数据结构 + 通信描述”为边界，而不是直接在数据结构内部混入 MPI 调用。
- 修改 `Global.h` 时避免破坏现有别名名义；如需新增类型（例如不同精度索引），优先添加新的 typedef 而不是重命名现有别名。
