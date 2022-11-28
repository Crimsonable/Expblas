#pragma once
#include "base.h"
#include "expbase.h"
#include "shape.h"
#include "storage.h"

namespace Expblas {
template <typename T> struct ExpWrapper<Scalar<T>> {
  ExpWrapper(const Scalar<T> &data) : data(data.data) {}

  BlasForceInline BlasCudaFunc T eval(size_t y, size_t x) const { return data; }

  T data;
};

template <typename T, size_t... shapes>
struct ExpWrapper<FTensor<T, shapes...>> {

  ExpWrapper(const FTensor<T, shapes...> &_data) : data(_data) {}

  BlasForceInline BlasCudaFunc T eval(size_t y, size_t x) const {
    return data.eval(y, x);
  }

  BlasForceInline BlasCudaFunc T &eval_ref(size_t y, size_t x) {
    return data.eval_ref(y, x);
  }

  FTensor<T, shapes...> data;
};

template <typename T, int Dim, Device device>
struct ExpWrapper<Tensor<T, Dim, device>> {
  ExpWrapper(const Tensor<T, Dim, device> &data)
      : data(data.dataptr()), stride(data.stride()) {}

  BlasForceInline BlasCudaFunc T eval(size_t y, size_t x) const {
    return data[y * stride + x];
  }

  BlasForceInline BlasCudaFunc T &eval_ref(size_t y, size_t x) {
    return data[y * stride + x];
  }

  T *data;
  size_t stride;
};

template <typename T, int Dim, Device device>
struct ExpWrapper<TensorRange<T, Dim, device>> {

  ExpWrapper(const TensorRange<T, Dim, device> &_data) : data(_data) {}

  BlasForceInline BlasCudaFunc T eval(size_t y, size_t x) const {
    return data.eval(y, x);
  }

  BlasForceInline BlasCudaFunc T &eval_ref(size_t y, size_t x) {
    return data.eval_ref(y, x);
  }

  TensorRange<T, Dim, device> data;
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
struct ExpWrapper<BinaryExp<Op, Lhs, Rhs, DataType, type>> {
  ExpWrapper(const BinaryExp<Op, Lhs, Rhs, DataType, type> &exp)
      : lhs(exp.lhs), rhs(exp.rhs) {}

  BlasForceInline BlasCudaFunc DataType eval(size_t y, size_t x) const {
    return Op::eval(lhs.eval(y, x), rhs.eval(y, x));
  }

  const ExpWrapper<Lhs> lhs;
  const ExpWrapper<Rhs> rhs;
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
struct ExpWrapper<UnaryExp<Op, Exp, DataType, exp_type>> {
  ExpWrapper(const UnaryExp<Op, Exp, DataType, exp_type> &exp)
      : exp(exp.self) {}

  BlasForceInline BlasCudaFunc DataType eval(size_t x, size_t y) const {
    return Op::eval(exp.eval(x, y));
  }

  const ExpWrapper<Exp> exp;
};

template <typename Exp, typename DataType>
struct ExpWrapper<TransposeExp<Exp, DataType>> {
  ExpWrapper(const TransposeExp<Exp, DataType> &exp) : exp(exp.self) {}

  BlasForceInline BlasCudaFunc DataType eval(size_t y, size_t x) const {
    return exp.eval(x, y);
  }

  const ExpWrapper<Exp> exp;
};

template <typename T>
BlasForceInline auto MakeExpWrapper(const Scalar<T> &data) {
  return ExpWrapper<Scalar<T>>(data);
}

template <typename Container, typename T, Device device>
BlasForceInline auto
MakeExpWrapper(const TensorBase<Container, T, device> &data) {
  return ExpWrapper<Container>(data.derived_to());
}

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
BlasForceInline auto
MakeExpWrapper(const BinaryExp<Op, Lhs, Rhs, DataType, type> &exp) {
  return ExpWrapper<BinaryExp<Op, Lhs, Rhs, DataType, type>>(exp);
}

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
BlasForceInline auto
MakeExpWrapper(const UnaryExp<Op, Exp, DataType, exp_type> &exp) {
  return ExpWrapper<UnaryExp<Op, Exp, DataType, exp_type>>(exp);
}

template <typename Exp, typename DataType>
BlasForceInline auto MakeExpWrapper(const TransposeExp<Exp, DataType> &exp) {
  return ExpWrapper<TransposeExp<Exp, DataType>>(exp);
}

template <typename Saver, typename Container, Device device, typename T,
          typename Exp, OperatorType exp_type, typename... Args>
inline void nonPacketExecute(TensorBase<Container, T, device> *dst,
                             const ExpWrapper<Exp> &exp, Args &&...args) {
  auto _dst = dst->derived();
  Shape<2> dst_shape = _dst->get_shape().flat2D();

#pragma omp parallel for
  for (int64_t y = 0; y < dst_shape[0]; ++y)
    for (int64_t x = 0; x < dst_shape[1]; ++x) {
      Saver::save(_dst->eval_ref(y, x), exp.eval(y, x), y, x,
                  std::forward<Args>(args)...);
    }
}

template <typename Saver, typename Reducer, typename Container,
          typename DataType, typename Exp, int ExpDim>
inline void
nonPacketReduceHdimExecute(TensorBase<Container, DataType, Device::CPU> *dst,
                           const ExpWrapper<Exp> &exp, Shape<ExpDim> exp_shape,
                           int axis) {
  auto _dst = dst->derived();
  Shape<2> reduce_shape = exp_shape.reduce(axis);
  size_t last = exp_shape.last();
  size_t axis_dim = exp_shape[axis];
  size_t reduce_stride = reduce_shape[1] / last;
  for (int out = 0; out < reduce_shape[0] / axis_dim; ++out) {
    for (int in = 0; in < reduce_stride; ++in) {
      auto temp = reduce_stride * out * axis_dim + in;
      for (int x = 0; x < last; ++x) {
        DataType val(Reducer::template init<DataType>());
        for (int w = 0; w < axis_dim; ++w) {
          Reducer::reduce(val, exp.eval(reduce_stride * w + temp, x));
        }
        Saver::save(_dst->eval_ref(reduce_stride * out / axis_dim + in, x),
                    val);
      }
    }
  }
}

template <typename Saver, typename Reducer, typename DataType, int Dim,
          typename Exp, int ExpDim>
inline void nonPacketReduceLdimExecute(Tensor<DataType, Dim, Device::CPU> *dst,
                                       const ExpWrapper<Exp> &exp,
                                       Shape<ExpDim> exp_shape) {
  for (int out = 0; out < exp_shape.size() / exp_shape[ExpDim - 1]; ++out) {
    DataType val(Reducer::template init<DataType>());
    for (int x = 0; x < exp_shape[ExpDim - 1]; ++x)
      Reducer::reduce(val, exp.eval(out, x));
    Saver::save(dst->eval_ref(out, 0), val);
  }
}

template <typename T>
inline void DiagonalFill(Tensor<T, 2, Device::CPU> *dst, T val) {
  for (int i = 0; i < dst->shape[0]; ++i)
    dst->eval_ref(i, i) = val;
}

} // namespace Expblas