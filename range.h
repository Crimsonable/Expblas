#pragma once
#include "ftensor.h"
#include "tensor.h"

namespace Expblas {
template <typename T, int Dim, Device device>
class TensorRange : public TensorBase<TensorRange<T, Dim, device>, T, device> {
public:
  TensorRange(T *_data, Shape<Dim> _shape, Uint yoffset, Uint ystride,
              Uint xoffset)
      : data(_data), offset_y(yoffset), stride_y(ystride), offset_x(xoffset),
        shape(_shape) {}

  template <typename Exp, OperatorType exp_type>
  BlasForceInline auto &operator=(const ExpBase<Exp, T, exp_type> &exp) {
    return this->assign(exp.derived_to());
  }

  BlasForceInline BlasCudaFunc T eval(Uint y, Uint x) const {
    return data[(y + offset_y) * stride_y + x + offset_x];
  }

  BlasForceInline BlasCudaFunc T &eval_ref(Uint y, Uint x) {
    return data[(y + offset_y) * stride_y + x + offset_x];
  }

  BlasForceInline BlasCudaFunc T eval_broadcast(Uint y, Uint x) const {
    return data[(y + offset_y) * stride_y + x + offset_x];
  }

  BlasForceInline BlasCudaFunc T *dataptr() const {
    return data[offset_y * stride_y + offset_x];
  }

  BlasForceInline BlasCudaFunc T *dataptr() {
    return data[offset_y * stride_y + offset_x];
  }

  BlasForceInline Uint stride() const { return this->data->stride; }

  BlasForceInline auto get_shape() const { return shape; }

private:
  T *data;
  Uint stride_y, offset_x, offset_y;
  Shape<Dim> shape;
};

template <typename Src, typename T, Device device,
          int dim = Meta::traits<Src>::dim>
BlasForceInline auto range(TensorBase<Src, T, device> &t, Shape<dim> start,
                           Shape<dim> shape) {
  Shape<dim> ts = t.get_shape();
  return TensorRange<T, dim, device>(t.dataptr(), shape, start.flat2D()[0],
                                     t.stride(), start[dim - 1]);
}
} // namespace Expblas