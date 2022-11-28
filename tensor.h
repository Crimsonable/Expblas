#pragma once
#include "base.h"
#include "exp_engine.h"
#include "expbase.h"
#include "range.h"
#include "shape.h"
#include "storage.h"

namespace Expblas {
template <typename Container, typename T, Device device>
class TensorBase : public ContainerExtention<Container> {
public:
  template <typename OtherContainer, Device srcDevice>
  Container &_copy(TensorBase<OtherContainer, T, srcDevice> &t, Meta::TrueTag) {
    auto ptr = this->derived();
    auto sptr = t.derived();
    auto dst_shape = ptr->shape.flat2D();
    Alloctor<T, srcDevice>::copy(ptr->dataptr(), ptr->stride() * sizeof(T),
                                 sptr->dataptr(), sptr->stride() * sizeof(T),
                                 dst_shape[1] * sizeof(T), dst_shape[0],
                                 TransportDir::LocalToLocal);
    return *ptr;
  }

  template <typename OtherContainer, Device srcDevice>
  Container &_copy(TensorBase<OtherContainer, T, srcDevice> &t,
                   Meta::FalseTag) {
    auto ptr = this->derived();
    auto sptr = t.derived();
    auto dst_shape = ptr->shape.flat2D();
    Alloctor<T, srcDevice>::copy(ptr->dataptr(), ptr->stride() * sizeof(T),
                                 sptr->dataptr(), sptr->stride() * sizeof(T),
                                 dst_shape[1] * sizeof(T), dst_shape[0],
                                 TransportDir::LocalToDevice);
    return *ptr;
  }

  template <typename Container, Device srcDevice>
  inline auto &copy(TensorBase<Container, T, srcDevice> &t) {
    return this->_copy(t,
                       typename Meta::TagDispatch<device == srcDevice>::type{});
  }

  BlasForceInline auto get_shape() const {
    return this->derived_to().get_shape();
  }

  BlasForceInline bool AllocCheck() const {
    return this->derived_to().AllocCheck();
  }

  BlasForceInline Uint stride() const { return this->derived_to().stride(); }

  BlasForceInline BlasCudaFunc T *dataptr() const {
    return this->derived_to().dataptr();
  }

  BlasForceInline BlasCudaFunc T *dataptr() {
    return this->derived()->dataptr();
  }
};

template <typename T, int Dim, Device device>
class Tensor : public TensorBase<Tensor<T, Dim, device>, T, device> {
public:
  using DataType = T;

  Tensor(const Shape<Dim> &shape) { this->shape = shape; }

  Tensor() {}

  template <typename Exp, OperatorType exp_type>
  BlasForceInline auto &operator=(const ExpBase<Exp, T, exp_type> &exp) {
    return this->assign(exp.derived_to());
  }

  BlasForceInline BlasCudaFunc T eval(Uint y, Uint x) const {
    return data->eval(y, x);
  }

  BlasForceInline BlasCudaFunc T &eval_ref(Uint y, Uint x) {
    return data->eval_ref(y, x);
  }

  BlasForceInline BlasCudaFunc T eval_broadcast(Uint y, Uint x) const {
    return data->eval_broadcast(y, x);
  }

  BlasForceInline BlasCudaFunc T *dataptr() const { return data->dataptr; }

  BlasForceInline BlasCudaFunc T *dataptr() { return data->dataptr; }

  BlasForceInline Uint stride() const { return this->data->stride; }

  BlasForceInline constexpr bool AllocCheck() const { return data->isAlloc; }

  BlasForceInline auto get_shape() const { return shape; }

  template <int N1> inline void resize(const Shape<N1> &new_shape) {
    shape = new_shape;
    data->resize(new_shape);
  }

  inline void clear() {
    if (data)
      data->clear();
  }

  Shape<Dim> shape;
  DenseStorage<T, device> *data = nullptr;
};

template <typename T, int Dim, Device device>
void allocSpace(Tensor<T, Dim, device> &t) {
  t.data = new DenseStorage<T, device>;
  t.data->resize(t.shape);
}

template <typename T, int Dim, Device device>
void freeSpace(Tensor<T, Dim, device> &t) {
  delete t.data;
  t.data = nullptr;
}

template <typename T, typename DContainer, Device dst_device,
          typename SContainer, Device src_device>
void Copy(TensorBase<DContainer, T, dst_device> &dst,
          TensorBase<SContainer, T, src_device> &src) {
  if (dst.get_shape().size() == src.get_shape().size() && dst.AllocCheck())
    dst.copy(src);
}

template <typename T, int Dim, Device device>
auto init(Tensor<T, Dim, device> &t, T val) {
  ExpEngine<OP::SAVER::assign, T>::eval(&t, Scalar<T>(val));
  return t;
}

template <typename T, int Dim, Device device> auto Zeros(Shape<Dim> shape) {
  auto res = Tensor<T, Dim, device>(shape);
  allocSpace(res);
  return init(res, T(0));
}

template <typename T, int Dim, Device device> auto Ones(Shape<Dim> shape) {
  auto res = Tensor<T, Dim, device>(shape);
  allocSpace(res);
  return init(res, T(1));
}
template <typename T, Device device> auto Eye(Uint width) {
  auto res = Zeros<T, 2, device>(Shape<2>{width, width});
  DiagonalFill(&res, T(1));
  return res;
}

template <typename T, typename Exp, OperatorType exp_type>
auto Emax(const ExpBase<Exp, T, exp_type> &exp, int axis) {
  return ReduceTemplate<Meta::traits<Exp>::device>::template eval<
      OP::SAVER::assign, OP::Reduce::max>(exp, axis);
}

template <typename T, typename Exp, OperatorType exp_type>
auto Emin(const ExpBase<Exp, T, exp_type> &exp, int axis) {
  return ReduceTemplate<Meta::traits<Exp>::device>::template eval<
      OP::SAVER::assign, OP::Reduce::min>(exp, axis);
}

template <typename T, typename Exp, OperatorType exp_type>
auto sum(const ExpBase<Exp, T, exp_type> &exp, int axis) {
  return ReduceTemplate<Meta::traits<Exp>::device>::template eval<
      OP::SAVER::assign, OP::Reduce::sum>(exp, axis);
}
} // namespace Expblas