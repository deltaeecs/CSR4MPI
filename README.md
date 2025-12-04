# CSR4MPI

一个支持分布式（按连续行块划分）与对称存储扩展的 C++17 CSR 稀疏矩阵库，集成 MPI（可选）、OpenMP（可选）、GoogleTest 单元测试与 MUMPS 导出适配。

**核心设计：模板化多标量类型支持** —— 库已完全模板化，四种标量类型（`float`、`double`、`std::complex<float>`、`std::complex<double>`）可在同一次编译中同时使用，无需切换编译开关重新构建。提供远程条目通信装配与对称矩阵的上下三角展开。

> 当前构建为静态库 `csr4mpi` + 测试可执行 `csr4mpi_tests`。

> 项目完全使用 Github Copilot 开发，用时 3 小时左右。

## 主要特性

- **模板化多标量类型**：所有核心类均为 C++ 模板，支持 `float`、`double`、`std::complex<float>`、`std::complex<double>` 四种标量类型，无需重新编译即可同时使用。
- **便利类型别名**：提供 `cCSRMatrixF`、`cCSRMatrixD`、`cCSRMatrixCF`、`cCSRMatrixCD` 等预定义别名，简化常用类型的声明。
- **类型特征工具**：`is_complex_v<T>`、`is_supported_scalar_v<T>`、`real_type_t<T>` 等编译期类型检测和转换工具。
- 连续行块分布：`cRowDistribution` 描述全局行到进程的映射。
- 本地 CSR 结构：`cCSRMatrix<Scalar>` 保存所属行区间的行指针、列索引与数值。
- 通信模式：`cCommPattern<Scalar>` 聚合跨进程目标条目 (globalRow, globalCol, value)。
- 汇总装配：`cCSRComm<Scalar>::Assemble` 使用 MPI Alltoallv 交换并累加远程重复条目。
- 对称存储展开：支持下三角存储在 SpMV/SpMM 中自动补全镜像贡献（本地 + 远程）。
- MUMPS 导出：`cMumpsAdapter<Scalar>` 输出 1-based COO 三元组。
- Benchmark：`csr4mpi_bench_spmv` 输出平均耗时与 GFLOPs。

## 构建

推荐在独立目录：

```powershell
mkdir build
cd build
cmake -G "Ninja" ..
cmake --build . --config Release
```

参数说明：

- `-DCSR4MPI_ENABLE_OPENMP=ON|OFF`：控制并行内核的 OpenMP 支持。
- 无需指定标量类型参数 —— 四种标量类型在同一次编译中均可使用。

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
```

也可使用本地合成矩阵（不依赖外部下载），通过生成器 `csr4mpi_gen_mm`：

```powershell
.\csr4mpi_gen_mm.exe 8000 8000 12 0 .\tests\matrices\synthetic_8k12_general.mtx
.\csr4mpi_gen_mm.exe 12000 12000 10 1 .\tests\matrices\synthetic_12k10_sym_lower.mtx
mpiexec -n 4 .\csr4mpi_bench_spmv.exe synthetic_8k12_general.mtx 20
mpiexec -n 4 .\csr4mpi_bench_spmv.exe synthetic_12k10_sym_lower.mtx 20
```

## 标量类型使用

库采用 C++ 模板实现多标量类型支持，四种类型在同一次编译中均可使用，无需切换编译开关：

### 预定义类型别名

| 别名 | 完整类型 |
|------|----------|
| `cCSRMatrixF` | `cCSRMatrix<float>` |
| `cCSRMatrixD` | `cCSRMatrix<double>` |
| `cCSRMatrixCF` | `cCSRMatrix<std::complex<float>>` |
| `cCSRMatrixCD` | `cCSRMatrix<std::complex<double>>` |

### 使用示例

```cpp
#include "CSRMatrix.h"
#include "Operations.h"

using namespace csr4mpi;

// 所有类型在同一程序中同时可用
cCSRMatrixD matDouble(0, n, n, rowPtr, colInd, valuesDouble);
cCSRMatrixCF matComplexFloat(0, n, n, rowPtr, colInd, valuesComplexFloat);

// SpMV 操作
std::vector<double> xD(n, 1.0), yD;
SpMV(matDouble, xD, yD);

std::vector<std::complex<float>> xCF(n, {1.0f, 0.0f}), yCF;
SpMV(matComplexFloat, xCF, yCF);

// 也可以直接使用模板形式
cCSRMatrix<std::complex<double>> matCD(...);
```

### 类型特征工具

`Global.h` 提供以下编译期类型检测与转换工具：

- `is_complex_v<T>`：判断 `T` 是否为 `std::complex<>` 类型。
- `is_supported_scalar_v<T>`：判断 `T` 是否为支持的四种标量之一。
- `real_type_t<T>`：获取标量的实部类型（实数类型返回自身，复数返回其分量类型）。

```cpp
static_assert(is_complex_v<std::complex<double>>);       // true
static_assert(!is_complex_v<double>);                    // true (double is not complex)
static_assert(is_supported_scalar_v<float>);             // true
static_assert(std::is_same_v<real_type_t<std::complex<float>>, float>);  // true
```

## 对称矩阵说明

若矩阵以下三角形式存储（Matrix Market symmetric lower），在分布式 SpMV/SpMM 中：

1. 本地展开：补全镜像列贡献。
2. 远程展开：对需要的镜像条目进行分桶发送与累加，避免重复计算。

## 目录要点

- `include/Global.h`：类型定义、类型特征（`is_complex_v`、`is_supported_scalar_v`、`real_type_t`）与 MPI 辅助函数。
- `include/CSRMatrix.h`：模板化 CSR 本地块 `cCSRMatrix<Scalar>` 与类型别名。
- `include/Distribution.h` / `src/Distribution.cpp`：行分布。
- `include/CommPattern.h`：模板化远程条目通信模式 `cCommPattern<Scalar>`。
- `include/CSRComm.h`：模板化汇总装配逻辑 `cCSRComm<Scalar>`。
- `include/MumpsAdapter.h`：模板化 MUMPS 导出 `cMumpsAdapter<Scalar>`。
- `include/Operations.h`：模板化 SpMV/SpMM 操作。
- `include/MatrixMarketLoader.h`：模板化 Matrix Market 文件加载器。
- `include/DistributedOps.h`：分布式 SpMV 操作。
- `bench/bench_large_spmv.cpp`：SpMV 性能基准。
- `tests/`：GoogleTest 单元与 MPI 分布式测试。
- `tests/test_all_scalar_types.cpp`：针对四种标量类型的全面测试（52 个测试用例）。

## TODO

- [ ] 更高效装配（行内列索引二分/哈希）
- [ ] SpMM 进一步通信压缩与重用
- [ ] 更多矩阵与可复现基准统计 (CSV)

## License

本项目基于 [MIT License](LICENSE) 开源。
