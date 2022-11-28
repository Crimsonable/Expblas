#ifndef CUDA_DISPATCH
#define CUDA_DISPATCH
#include "expbase.h"
#include "non_packet.h"

namespace Expblas {
namespace CudaPars {
const int MaxBlocksPerGrid = 65535;
const int SugBlocksPerGrid = 1024;
const int MaxThreadsPerBlock = 1024;
const int SugThreadsPerBlock = 256;
const int WrapSize = 32;
} // namespace CudaPars

template <typename Saver, size_t blocksize, typename Dst, typename Exp,
          typename... Args>
__device__ void WrapperKernel(Dst dst, const ExpWrapper<Exp> exp,
                              Shape<2> dst_shape, size_t size, Args &&...args) {
  const int64_t tid = blockIdx.x * blocksize + threadIdx.x;
  if (tid < size) {
    const int64_t x = tid % dst_shape[1];
    const int64_t y = tid / dst_shape[1];
    Saver::save(dst.eval_ref(y, x), exp.eval(y, x), y, x,
                std::forward<Args>(args)...);
  }
}

template <typename Saver, size_t blocksize, typename Dst, typename Exp,
          typename... Args>
__global__ void WrapperKernelCaller(Dst dst, const ExpWrapper<Exp> exp,
                                    Shape<2> dst_shape, size_t size,
                                    Args... args) {
  WrapperKernel<Saver, blocksize>(dst, exp, dst_shape, size,
                                  std::forward<Args>(args)...);
}

template <typename Saver, typename Container, typename DataType, typename Exp,
          typename... Args>
inline void GPUExecute(TensorBase<Container, DataType, Device::GPU> &dst,
                       Shape<2> dst_shape, const ExpWrapper<Exp> exp,
                       Args &&...args) {
  size_t xstride = dst_shape[1];
  size_t num_blocks =
      (xstride * dst_shape[0] + CudaPars::SugThreadsPerBlock - 1) /
      CudaPars::SugThreadsPerBlock;
  dim3 blockdim(CudaPars::SugThreadsPerBlock, 1, 1);
  if (num_blocks < CudaPars::MaxBlocksPerGrid) {
    dim3 griddim(num_blocks, 1, 1);
    if constexpr (std::is_same_v<Container,
                                 Tensor<DataType, Meta::traits<Container>::dim,
                                        Device::GPU>>)
      WrapperKernelCaller<Saver, CudaPars::SugThreadsPerBlock>
          <<<griddim, blockdim>>>(MakeExpWrapper(dst), exp, dst_shape,
                                  dst_shape.size(), args...);
    else
      WrapperKernelCaller<Saver, CudaPars::SugThreadsPerBlock>
          <<<griddim, blockdim>>>(*dst.derived(), exp, dst_shape,
                                  dst_shape.size(), args...);
  }
}

template <typename Saver, typename Container, typename Exp,
          OperatorType exp_type>
struct GPUEngine {
  template <typename DataType, typename... Args>
  inline static void dispatch(TensorBase<Container, DataType, Device::GPU> *dst,
                              const ExpBase<Exp, DataType, exp_type> &exp,
                              Args &&...args) {
    GPUExecute<Saver>(*(dst->derived()), dst->get_shape(),
                      MakeExpWrapper(exp.derived_to()),
                      std::forward<Args>(args)...);
  }
};

template <size_t blocksize, typename T>
__device__ void warpReduce(volatile T *buffer, size_t tid) {
  if (blocksize >= 512) {
    if (tid < 256)
      buffer[tid] += buffer[tid + 256];
    __syncthreads();
  }
  if (blocksize >= 256) {
    if (tid < 128)
      buffer[tid] += buffer[tid + 128];
    __syncthreads();
  }
  if (blocksize >= 128) {
    if (tid < 64)
      buffer[tid] += buffer[tid + 64];
    __syncthreads();
  }

  if (tid < 32) {
    if (blocksize >= 64)
      buffer[tid] += buffer[tid + 32];
    if (blocksize >= 32)
      buffer[tid] += buffer[tid + 16];
    if (blocksize >= 16)
      buffer[tid] += buffer[tid + 8];
    if (blocksize >= 8)
      buffer[tid] += buffer[tid + 4];
    if (blocksize >= 4)
      buffer[tid] += buffer[tid + 2];
    if (blocksize >= 2)
      buffer[tid] += buffer[tid + 1];
  }
}

template <typename Saver, typename Reducer, int blocksize, typename DataType,
          int Dim, typename Exp>
__global__ void
ReduceKernelH(ExpWrapper<Tensor<DataType, Dim, Device::GPU>> dst,
              const ExpWrapper<Exp> exp, size_t w, size_t n) {
  extern __shared__ DataType buffer[];
  size_t temp = gridDim.y * blockIdx.x * w + blockIdx.y;
  size_t reduce_stride = blockDim.y;
  size_t tid = threadIdx.x;
  buffer[tid] = Reducer::template init<DataType>();

  for (int i = 0; i < n; ++i) {
    int offset = i * blocksize + tid;
    if (offset < w) {
      Reducer::reduce(buffer[tid],
                      exp.eval(reduce_stride * offset + temp, blockIdx.z));
    }
    __syncthreads();
  }

  warpReduce<blocksize>(buffer, tid);

  if (tid == 0)
    Saver::save(
        dst.eval_ref(reduce_stride * blockIdx.x / w + blockIdx.y, blockIdx.z),
        buffer[0]);
}

template <typename Saver, typename Reducer, int blocksize, typename DataType,
          int Dim, typename Exp>
__global__ void
ReduceKernelL(ExpWrapper<Tensor<DataType, Dim, Device::GPU>> dst,
              const ExpWrapper<Exp> exp, size_t w, size_t n) {
  extern __shared__ DataType buffer[];
  size_t tid = threadIdx.x;
  buffer[tid] = Reducer::template init<DataType>();

  for (int i = 0; i < n; ++i) {
    int offset = i * blocksize + tid;
    if (offset < w)
      Reducer::reduce(buffer[tid], exp.eval(blockIdx.x, offset));
  }
  __syncthreads();

  warpReduce<blocksize>(buffer, tid);

  if (tid == 0)
    Saver::save(dst.eval_ref(blockIdx.x, 0), buffer[0]);
}

template <typename Saver, typename Reducer, typename DataType, int Dim,
          typename Exp, int ExpDim>
inline void GPUReduceLdimExecute(Tensor<DataType, Dim, Device::GPU> *dst,
                                 const ExpWrapper<Exp> exp,
                                 Shape<ExpDim> exp_shape) {
  size_t x = exp_shape.last();
  dim3 grid_shape(exp_shape.size() / x, 1, 1);
  dim3 block_shape(CudaPars::SugThreadsPerBlock, 1, 1);

  ReduceKernelL<Saver, Reducer, CudaPars::SugThreadsPerBlock>
      <<<grid_shape, block_shape,
         CudaPars::SugThreadsPerBlock * sizeof(DataType)>>>(
          MakeExpWrapper(*dst), exp, x,
          (x + CudaPars::SugThreadsPerBlock - 1) /
              CudaPars::SugThreadsPerBlock);
}

template <typename Saver, typename Reducer, typename DataType, int Dim,
          typename Exp, int ExpDim>
inline void GPUReduceHdimExecute(Tensor<DataType, Dim, Device::GPU> *dst,
                                 const ExpWrapper<Exp> exp,
                                 Shape<ExpDim> exp_shape, int axis) {
  auto reduce_shape = exp_shape.reduce(axis);

  size_t x = exp_shape.last();
  size_t w = exp_shape[axis];
  size_t out_size = reduce_shape[0] / w;
  size_t in_size = reduce_shape[1] / x;

  dim3 grid_shape(out_size, in_size, x);
  dim3 block_shape(CudaPars::SugThreadsPerBlock, 1, 1);

  ReduceKernelH<Saver, Reducer, CudaPars::SugThreadsPerBlock>
      <<<grid_shape, block_shape,
         CudaPars::SugThreadsPerBlock * sizeof(DataType)>>>(
          MakeExpWrapper(*dst), exp, w,
          (w + CudaPars::SugThreadsPerBlock - 1) /
              CudaPars::SugThreadsPerBlock);
  EXPBLAS_CUDA_KERNELPASTCHECK();
}

} // namespace Expblas

#endif // CUDA_DISPATCH
