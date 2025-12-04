// Test file to verify all 4 scalar types work correctly
// This ensures all scalar types (float, double, complex<float>, complex<double>) compile and run correctly

#include "CSRMatrix.h"
#include "Operations.h"
#include "MumpsAdapter.h"
#include "Distribution.h"
#include <gtest/gtest.h>
#include <complex>
#include <cmath>

using namespace csr4mpi;

// Template test fixture for testing all scalar types
template <typename Scalar>
class AllScalarTypesTest : public ::testing::Test {
protected:
    using scalar_type = Scalar;

    // Helper to build a simple 3x3 local CSR matrix with rows [0,3)
    static cCSRMatrix<Scalar> BuildSimple3x3()
    {
        iSize iGlobalRowBegin = 0;
        iSize iGlobalRowEnd = 3;
        iSize iGlobalColCount = 3;
        // Row 0: cols 0,2 -> values 1,2
        // Row 1: col 1 -> value 3
        // Row 2: cols 0,1,2 -> values 4,5,6
        std::vector<iIndex> vRowPtr = { 0, 2, 3, 6 };
        std::vector<iIndex> vColInd = { 0, 2, 1, 0, 1, 2 };
        std::vector<Scalar> vValues = { 
            static_cast<Scalar>(1), static_cast<Scalar>(2), 
            static_cast<Scalar>(3), 
            static_cast<Scalar>(4), static_cast<Scalar>(5), static_cast<Scalar>(6) 
        };
        return cCSRMatrix<Scalar>(iGlobalRowBegin, iGlobalRowEnd, iGlobalColCount, vRowPtr, vColInd, vValues);
    }

    // Helper function to check approximate equality for floating-point and complex types
    static bool ApproxEqual(Scalar a, Scalar b, double tolerance = 1e-10) {
        // std::abs works for both real and complex types
        return std::abs(a - b) < tolerance;
    }
};

// Define type list for all supported scalar types
using ScalarTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(AllScalarTypesTest, ScalarTypes);

// Test basic matrix construction for all scalar types
TYPED_TEST(AllScalarTypesTest, BasicConstruction)
{
    using Scalar = TypeParam;
    
    std::vector<iIndex> vRowPtr = { 0, 2, 3 };
    std::vector<iIndex> vColInd = { 0, 2, 1 };
    std::vector<Scalar> vValues = { 
        static_cast<Scalar>(1.0), 
        static_cast<Scalar>(2.0), 
        static_cast<Scalar>(3.0) 
    };

    cCSRMatrix<Scalar> cLocal(0, 2, 3, vRowPtr, vColInd, vValues);

    EXPECT_EQ(cLocal.iGlobalRowBegin(), 0);
    EXPECT_EQ(cLocal.iGlobalRowEnd(), 2);
    EXPECT_EQ(cLocal.vRowPtr().size(), 3u);
    EXPECT_EQ(cLocal.vColInd().size(), 3u);
    EXPECT_EQ(cLocal.vValues().size(), 3u);
}

// Test SpMV for all scalar types
TYPED_TEST(AllScalarTypesTest, SpMVBasic)
{
    using Scalar = TypeParam;
    
    auto A = this->BuildSimple3x3();
    std::vector<Scalar> x = { 
        static_cast<Scalar>(10), 
        static_cast<Scalar>(20), 
        static_cast<Scalar>(30) 
    };
    std::vector<Scalar> y;
    SpMV(A, x, y);
    
    ASSERT_EQ(y.size(), 3u);
    // Row0: 1*10 + 2*30 = 70
    // Row1: 3*20 = 60
    // Row2: 4*10 + 5*20 + 6*30 = 320
    EXPECT_TRUE(this->ApproxEqual(y[0], static_cast<Scalar>(70)));
    EXPECT_TRUE(this->ApproxEqual(y[1], static_cast<Scalar>(60)));
    EXPECT_TRUE(this->ApproxEqual(y[2], static_cast<Scalar>(320)));
}

// Test SpMM for all scalar types
TYPED_TEST(AllScalarTypesTest, SpMMBasic)
{
    using Scalar = TypeParam;
    
    auto A = this->BuildSimple3x3();
    // X: 3x2 column-major -> columns x0={10,20,30}, x1={1,2,3}
    std::vector<Scalar> X = { 
        static_cast<Scalar>(10), static_cast<Scalar>(20), static_cast<Scalar>(30), 
        static_cast<Scalar>(1), static_cast<Scalar>(2), static_cast<Scalar>(3) 
    };
    std::vector<Scalar> Y;
    SpMM(A, X, 2, Y);
    
    iSize localRows = A.iGlobalRowEnd() - A.iGlobalRowBegin();
    ASSERT_EQ(Y.size(), static_cast<size_t>(localRows * 2));
    
    // Column 0 result equals SpMV with x0
    EXPECT_TRUE(this->ApproxEqual(Y[0], static_cast<Scalar>(70)));
    EXPECT_TRUE(this->ApproxEqual(Y[1], static_cast<Scalar>(60)));
    EXPECT_TRUE(this->ApproxEqual(Y[2], static_cast<Scalar>(320)));
    
    // Column 1: multiply with x1={1,2,3}
    // Row0: 1*1 + 2*3 = 7
    // Row1: 3*2 = 6
    // Row2: 4*1 + 5*2 + 6*3 = 32
    EXPECT_TRUE(this->ApproxEqual(Y[3], static_cast<Scalar>(7)));
    EXPECT_TRUE(this->ApproxEqual(Y[4], static_cast<Scalar>(6)));
    EXPECT_TRUE(this->ApproxEqual(Y[5], static_cast<Scalar>(32)));
}

// Test MumpsAdapter export for all scalar types
TYPED_TEST(AllScalarTypesTest, MumpsAdapterExport)
{
    using Scalar = TypeParam;
    
    std::vector<iIndex> rowPtr = { 0, 2, 3 };
    std::vector<iIndex> colInd = { 0, 2, 1 };
    std::vector<Scalar> values = {
        static_cast<Scalar>(10.0),
        static_cast<Scalar>(20.0),
        static_cast<Scalar>(30.0)
    };

    cCSRMatrix<Scalar> local(0, 2, 3, rowPtr, colInd, values);
    cRowDistribution dist = cRowDistribution::CreateBlockDistribution(2, 1, 0);

    std::vector<int> IRN;
    std::vector<int> JCN;
    std::vector<Scalar> A;

    cMumpsAdapter<Scalar>::ExportLocalBlock(local, dist, IRN, JCN, A);

    ASSERT_EQ(IRN.size(), 3u);
    ASSERT_EQ(JCN.size(), 3u);
    ASSERT_EQ(A.size(), 3u);

    // Verify 1-based indices
    EXPECT_EQ(IRN[0], 1);
    EXPECT_EQ(JCN[0], 1);
    EXPECT_TRUE(this->ApproxEqual(A[0], static_cast<Scalar>(10.0)));
}

// Test empty matrix for all scalar types
TYPED_TEST(AllScalarTypesTest, EmptyMatrix)
{
    using Scalar = TypeParam;
    
    std::vector<iIndex> vRowPtr = { 0 };
    std::vector<iIndex> vColInd;
    std::vector<Scalar> vValues;
    
    cCSRMatrix<Scalar> A(0, 0, 3, vRowPtr, vColInd, vValues);
    
    EXPECT_EQ(A.iGlobalRowBegin(), 0);
    EXPECT_EQ(A.iGlobalRowEnd(), 0);
    EXPECT_EQ(A.vValues().size(), 0u);
}

// Test type traits
TEST(TypeTraitsTest, IsComplex)
{
    EXPECT_FALSE(is_complex_v<float>);
    EXPECT_FALSE(is_complex_v<double>);
    EXPECT_TRUE(is_complex_v<std::complex<float>>);
    EXPECT_TRUE(is_complex_v<std::complex<double>>);
}

TEST(TypeTraitsTest, IsSupportedScalar)
{
    EXPECT_TRUE(is_supported_scalar_v<float>);
    EXPECT_TRUE(is_supported_scalar_v<double>);
    EXPECT_TRUE(is_supported_scalar_v<std::complex<float>>);
    EXPECT_TRUE(is_supported_scalar_v<std::complex<double>>);
    EXPECT_FALSE(is_supported_scalar_v<int>);
    EXPECT_FALSE(is_supported_scalar_v<long>);
}

TEST(TypeTraitsTest, RealType)
{
    static_assert(std::is_same_v<real_type_t<float>, float>, "real_type_t<float> should be float");
    static_assert(std::is_same_v<real_type_t<double>, double>, "real_type_t<double> should be double");
    static_assert(std::is_same_v<real_type_t<std::complex<float>>, float>, "real_type_t<complex<float>> should be float");
    static_assert(std::is_same_v<real_type_t<std::complex<double>>, double>, "real_type_t<complex<double>> should be double");
}

// Test type aliases
TEST(TypeAliasTest, MatrixAliases)
{
    // Verify type aliases compile correctly
    cCSRMatrixF matF;
    cCSRMatrixD matD;
    cCSRMatrixCF matCF;
    cCSRMatrixCD matCD;
    
    EXPECT_EQ(matF.iGlobalRowBegin(), 0);
    EXPECT_EQ(matD.iGlobalRowBegin(), 0);
    EXPECT_EQ(matCF.iGlobalRowBegin(), 0);
    EXPECT_EQ(matCD.iGlobalRowBegin(), 0);
}
