#pragma once

#include <cstdint>
#include <complex>
#include <type_traits>
#include <mpi.h>

namespace csr4mpi {
enum eValueType {
    eFloat32Real,
    eFloat64Real,
    eFloat32Complex,
    eFloat64Complex
};
enum eSymmStorage {
    eNone,
    eSymmLower,
    eSymmUpper,
    eSymmFull
};

using iIndex = std::int64_t;
using iSize = std::int64_t;

// Type traits to detect complex types
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Get the real type underlying a scalar (identity for real, component type for complex)
template <typename T>
struct real_type {
    using type = T;
};

template <typename T>
struct real_type<std::complex<T>> {
    using type = T;
};

template <typename T>
using real_type_t = typename real_type<T>::type;

// Check if a type is a supported scalar type
template <typename T>
struct is_supported_scalar : std::bool_constant<
    std::is_same_v<T, float> ||
    std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>> {};

template <typename T>
inline constexpr bool is_supported_scalar_v = is_supported_scalar<T>::value;

namespace mpi_helper {
    // Helper to get MPI datatype for a scalar type
    // pCreatedType is set if a custom MPI type was created (caller must free it)
    template <typename Scalar>
    inline MPI_Datatype GetMPIDatatype(MPI_Datatype* pCreatedType = nullptr) {
        if constexpr (std::is_same_v<Scalar, float>) {
            return MPI_FLOAT;
        } else if constexpr (std::is_same_v<Scalar, double>) {
            return MPI_DOUBLE;
        } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
#ifdef MPI_C_FLOAT_COMPLEX
            return MPI_C_FLOAT_COMPLEX;
#else
            MPI_Datatype dt;
            MPI_Type_contiguous(2, MPI_FLOAT, &dt);
            MPI_Type_commit(&dt);
            if (pCreatedType) *pCreatedType = dt;
            return dt;
#endif
        } else if constexpr (std::is_same_v<Scalar, std::complex<double>>) {
#ifdef MPI_C_DOUBLE_COMPLEX
            return MPI_C_DOUBLE_COMPLEX;
#else
            MPI_Datatype dt;
            MPI_Type_contiguous(2, MPI_DOUBLE, &dt);
            MPI_Type_commit(&dt);
            if (pCreatedType) *pCreatedType = dt;
            return dt;
#endif
        }
    }
}

}
