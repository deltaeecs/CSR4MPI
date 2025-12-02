# CSR4MPI

一个支持分布式（按连续行块划分）与对称存储扩展的 C++17 CSR 稀疏矩阵库，集成 MPI（可选）、OpenMP（可选）、GoogleTest 单元测试与 MUMPS 导出适配。提供标量类型冷切换（float/double/complex<float>/complex<double>），含远程条目通信装配与对称矩阵的上下三角展开。

> 当前构建为静态库 `csr4mpi` + 测试可执行 `csr4mpi_tests`。

> 项目完全使用 Github Copilot 开发，用时 3 小时左右。

## 主要特性

- 连续行块分布：`cRowDistribution` 描述全局行到进程的映射。
- 本地 CSR 结构：`cCSRMatrix` 保存所属行区间的行指针、列索引与数值。
- 通信模式：`cCommPattern` 聚合跨进程目标条目 (globalRow, globalCol, value)。
- 汇总装配：`cCSRComm::Assemble` 使用 MPI Alltoallv 交换并累加远程重复条目。
- 对称存储展开：支持下三角存储在 SpMV/SpMM 中自动补全镜像贡献（本地 + 远程）。
- MUMPS 导出：`cMumpsAdapter` 输出 1-based COO 三元组。
- 标量选择：`CSR4MPI_VALUE_TYPE` 控制四种标量类型（默认双精度实数）。
- Benchmark：`csr4mpi_bench_spmv` 输出平均耗时与 GFLOPs。

## 构建

推荐在独立目录：

```powershell
mkdir build
cd build
cmake -G "Ninja" -DCSR4MPI_VALUE_TYPE=1 ..
cmake --build . --config Release
```

参数说明：

- `-DCSR4MPI_ENABLE_OPENMP=ON|OFF`：控制并行内核的 OpenMP 支持。
- `-DCSR4MPI_VALUE_TYPE=0|1|2|3`：`0=float,1=double,2=complex<float>,3=complex<double>`。

## 下载示例矩阵 (Matrix Market / SuiteSparse)

提供脚本 `scripts/download_matrices.sh`（依赖 bash + curl + tar）：

```bash
bash scripts/download_matrices.sh
```

将尝试下载：

- `bcsstk13.mtx` (Boeing group) – 中等规模结构矩阵
- `add32.mtx` (HB group) – 较小测试矩阵

文件放置到 `tests/data/`。若访问失败请手动前往 <https://sparse.tamu.edu/> 对应页面下载 tar.gz 后解压并将 `.mtx` 文件置于该目录。

也支持通过 `scripts/matrices.txt` 指定自定义下载清单（每行一个 `.tar.gz` 完整 URL；忽略以 `#` 开头的行）：

```bash
echo "https://your-storage/MM/Boeing/bcsstk13.tar.gz" >> scripts/matrices.txt
echo "https://your-storage/MM/HB/add32.tar.gz" >> scripts/matrices.txt
bash scripts/download_matrices.sh
```

## 运行测试

构建后在 `build`：

```powershell
ctest -C Release --output-on-failure
```

多进程（示例 4 进程）针对单独的分布式测试：

```powershell
mpiexec -n 4 .\csr4mpi_tests.exe --gtest_filter=MpiMatrixMarketLargeTest.DistributedLargeSpMVAccuracy
mpiexec -n 4 .\csr4mpi_tests.exe --gtest_filter=MpiMatrixMarketLargeTest.DistributedDuplicateAssembly
mpiexec -n 4 .\csr4mpi_tests.exe --gtest_filter=MpiSymmetricSpMMTest.DistributedLowerExpansionGatherMM4Proc
```

大型矩阵测试回退逻辑：

- 首选外部下载的 `bcsstk13.mtx`。
- 若缺失自动尝试合成矩阵 `synthetic_12k12_general.mtx`（需先使用生成器创建）。
- 两者都不可用时测试标记为 skip 并输出生成提示。

生成兜底矩阵示例：

```powershell
.\csr4mpi_gen_mm.exe 12000 12000 12 0 .\tests\matrices\synthetic_12k12_general.mtx
```

## Benchmark 使用 (SpMV)

可执行：`csr4mpi_bench_spmv`

```powershell
mpiexec -n 4 .\csr4mpi_bench_spmv.exe bcsstk13.mtx 20
```

输出示例字段：

- `avg / max / min`：多次迭代（默认 10）后进程归约的平均/最大/最小单次耗时（秒）。
- `nnz`：全局非零元数。
- `GFLOPs`：采用公式 `2 * nnz / time / 1e9`（一次乘加视为 2 浮点运算）。

非 MPI 下直接运行：

```powershell
./csr4mpi_bench_spmv.exe bcsstk13.mtx 10

也可使用本地合成矩阵（不依赖外部下载），通过生成器 `csr4mpi_gen_mm`：

```powershell
.\csr4mpi_gen_mm.exe 8000 8000 12 0 .\tests\matrices\synthetic_8k12_general.mtx
.\csr4mpi_gen_mm.exe 12000 12000 10 1 .\tests\matrices\synthetic_12k10_sym_lower.mtx
mpiexec -n 4 .\csr4mpi_bench_spmv.exe synthetic_8k12_general.mtx 20
mpiexec -n 4 .\csr4mpi_bench_spmv.exe synthetic_12k10_sym_lower.mtx 20
```

## 标量类型切换

通过 CMake 变量 `CSR4MPI_VALUE_TYPE`：

| 值 | 类型 |
|----|------|
| 0  | float |
| 1  | double (默认) |
| 2  | std::complex<float> |
| 3  | std::complex<double> |

内部统一别名 `vScalar`，测试/核心代码不直接硬编码具体标量。

## 对称矩阵说明

若矩阵以下三角形式存储（Matrix Market symmetric lower），在分布式 SpMV/SpMM 中：

1. 本地展开：补全镜像列贡献。
2. 远程展开：对需要的镜像条目进行分桶发送与累加，避免重复计算。

## 目录要点

- `include/Global.h`：类型与宏定义。
- `include/CSRMatrix.h` / `src/CSRMatrix.cpp`：CSR 本地块。
- `include/Distribution.h` / `src/Distribution.cpp`：行分布。
- `include/CommPattern.h` / `src/CommPattern.cpp`：远程条目通信模式。
- `include/CSRComm.h` / `src/CSRComm.cpp`：汇总装配逻辑。
- `include/MumpsAdapter.h` / `src/MumpsAdapter.cpp`：MUMPS 导出。
- `bench/bench_large_spmv.cpp`：SpMV 性能基准。
- `tests/`：GoogleTest 单元与 MPI 分布式测试。

## 后续改进方向

- 更高效装配（行内列索引二分/哈希）。
- SpMM 进一步通信压缩与重用。
- 更多矩阵与可复现基准统计 (CSV)。

## License

本项目当前为实验性质；请在引用/分发前确认后续 License 或附加条款更新。
