#pragma once
#include <assert.h>
#include <cmath>

#ifdef USE_CUDA
// #define CUDA_ALIGN_ALLOC
#include <cuda/cuda_runtime.h>
#include <cuda/device_launch_parameters.h>
#define BlasCudaFunc __device__ __host__ inline
#define BlasCudaConstruc __device__ __host__
#else
#define BlasCudaFunc
#define BlasCudaConstruc
#endif // USE_CUDA

#ifdef USE_MKL
#include <mkl.h>
#endif // USE_MKL

#if defined(__clang__) || defined(__GNUC__)
#define CPP_LANG __cplusplus
#elif defined(_MSC_VER)
#define CPP_LANG _MSVC_LANG
#endif

#if CPP_LANG >= 201703L
#define CPP_17
#define ConstexprIF(condition) if constexpr (condition)
#else
#define CPP_14
#define ConstexprIF(condition) if (condition)
#endif

#include <iostream>
#include <memory>
#include <new>
#include <omp.h>

#define EXPBLAS_CUDA_CALL(call)                                                \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                             \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));         \
    }                                                                          \
  }

#define EXPBLAS_CUDA_KERNELPASTCHECK()                                         \
  do {                                                                         \
    cudaDeviceSynchronize();                                                   \
    cudaError error = cudaPeekAtLastError();                                   \
    printf("error reason:%s\n", cudaGetErrorString(error));                    \
  } while (0)

#define EXP_ASSERT(condition, message)                                         \
  {                                                                            \
    if (!condition) {                                                          \
      printf("Error: %s", message);                                            \
      abort();                                                                 \
    }                                                                          \
  }

namespace Expblas {
using Uint = unsigned int;
enum class TensorState { Dynamic, Static };
enum class Device { UNIFY = 1, GPU = 7, CPU = 3 };
enum class OperatorType {
  container = 1,
  keepDim = 3,
  changeDim = 7,
  advance = 15
};
enum class Arch { AVX2 = 1, SSE = 3, Scalar = 7 };
enum class TransportDir { LocalToDevice, LocalToLocal };
enum class StorageMajor { Vector, RowMajor, ColumnMajor };

// #define USE_CUDA
#define BlasForceInline __forceinline
#define BlasUseSIMD false
#define BlasDefaultArch Arch::SSE

} // namespace Expblas