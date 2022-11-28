#pragma once
#include "base.h"
#include "shape.h"

namespace Expblas {
template <typename T, Device device> struct Alloctor;

template <typename T> struct Alloctor<T, Device::CPU> {
  static void alloc(T *&ptr, size_t row, size_t col, size_t &stride,
                    size_t &size) {
    auto row_size = sizeof(T) * col;
    stride = row_size % alignof(T)
                 ? (row_size + alignof(T) - row_size % alignof(T)) / sizeof(T)
                 : row_size / sizeof(T);
    size = stride * row;
    ptr = static_cast<T *>(_aligned_malloc(size * sizeof(T), 32));
  }

  static void free(T *&ptr) { _aligned_free(ptr); }

  static void copy(T *target, size_t t_stride, T *src, size_t s_stride,
                   size_t width, size_t height, TransportDir method) {
    switch (method) {
    case Expblas::TransportDir::LocalToLocal:
      memcpy(target, src, width * height);
      break;
    case Expblas::TransportDir::LocalToDevice:
#ifdef USE_CUDA
#ifdef CUDA_ALIGN_ALLOC
      EXPBLAS_CUDA_CALL(cudaMemcpy2D(target, t_stride, src, s_stride, width,
                                     height, cudaMemcpyHostToDevice));
#else
      EXPBLAS_CUDA_CALL(
          cudaMemcpy(target, src, width * height, cudaMemcpyHostToDevice));
#endif // CUDA_ALIGN_ALLOC
#endif // USE_CUDA
      break;
    default:
      abort();
    }
  }
};

#ifdef USE_CUDA
template <typename T> struct Alloctor<T, Device::GPU> {
  static void alloc(T *&ptr, size_t row, size_t col, size_t &stride,
                    size_t &size) {
#ifdef CUDA_ALIGN_ALLOC
    EXPBLAS_CUDA_CALL(cudaMallocPitch(&ptr, &stride, col * sizeof(T), row));
    stride /= sizeof(T);
#else
    EXPBLAS_CUDA_CALL(cudaMalloc(&ptr, row * col * sizeof(T)));
    stride = col;
#endif // CUDA_ALIGN_ALLOC
    size = stride * row;
  }

  static void free(T *ptr) { EXPBLAS_CUDA_CALL(cudaFree(ptr)); }

  static void copy(T *target, size_t t_stride, T *src, size_t s_stride,
                   size_t width, size_t height, TransportDir method) {
    switch (method) {
    case Expblas::TransportDir::LocalToDevice:
#ifdef CUDA_ALIGN_ALLOC
      EXPBLAS_CUDA_CALL(cudaMemcpy2D(target, t_stride, src, s_stride, width,
                                     height, cudaMemcpyDeviceToHost));
#else
      EXPBLAS_CUDA_CALL(
          cudaMemcpy(target, src, width * height, cudaMemcpyDeviceToHost));
#endif // CUDA_ALIGN_ALLOC

      break;
    case Expblas::TransportDir::LocalToLocal:
      EXPBLAS_CUDA_CALL(cudaMemcpy2D(target, t_stride, src, s_stride, width,
                                     height, cudaMemcpyDeviceToDevice));
      break;
    default:
      abort();
    }
  }
#endif // USE_CUDA

template <typename T, Device device> struct DenseStorage {
  DenseStorage() {}

  template <int N> DenseStorage(const Shape<N> &shape) {
    Alloctor<T, device>::alloc(dataptr, shape.flat2D()[0], shape[N - 1], stride,
                               size);
    isAlloc = true;
  }

  DenseStorage(T *data, size_t _stride, size_t _size)
      : dataptr(data), stride(_stride), size(size), isAlloc(true) {}

  ~DenseStorage() { this->clear(); }

  BlasForceInline BlasCudaFunc T &eval_ref(size_t y, size_t x) {
    return dataptr[(y * stride + x) % size];
  }

  BlasForceInline BlasCudaFunc T eval(size_t y, size_t x) const {
    return dataptr[(y * stride + x) % size];
  }

  BlasForceInline BlasCudaFunc T eval_broadcast(size_t y, size_t x) const {
    return dataptr[(y * stride + x) % size];
  }

  BlasForceInline BlasCudaFunc T &eval(size_t idx) { return dataptr[idx]; }

  BlasForceInline BlasCudaFunc T eval_ref(size_t idx) const {
    return dataptr[idx];
  }

  inline void clear() {
    if (isAlloc) {
      Alloctor<T, device>::free(dataptr);
      isAlloc = false;
    }
    dataptr = nullptr;
  }

  template <int N> void resize(const Expblas::Shape<N> &new_shape) {
    bool isResize = false;
    if (new_shape.size() > size) {
      if (isAlloc)
        Alloctor<T, device>::free(dataptr);
      Alloctor<T, device>::alloc(dataptr, new_shape.flat2D()[0],
                                 new_shape[N - 1], stride, size);
      isResize = true;
    }
    if (!isResize)
      stride = new_shape.flat2D()[1];
    isAlloc = true;
  }

  void copy_local(T *target, size_t _stride, size_t width, size_t height) {
    Alloctor<T, device>::copy(target, _stride, dataptr, stride * sizeof(T),
                              width, height, TransportDir::LocalToLocal);
  }

  void copy_device(T *target, size_t _stride, size_t width, size_t height) {
    Alloctor<T, device>::copy(target, _stride, dataptr, stride * sizeof(T),
                              width, height, TransportDir::LocalToDevice);
  }

  bool isAlloc = false;
  size_t size = 0;
  size_t stride;
  T *dataptr = nullptr;
};
} // namespace Expblas