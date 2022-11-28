#pragma once
#include "base.h"
#include "exp_engine.h"
#include "expbase.h"
#include "shape.h"
#include "storage.h"

namespace Expblas {
template <typename T, Uint... shapes>
class FTensor : public TensorBase<FTensor<T, shapes...>, T, Device::UNIFY> {
public:
  BlasCudaConstruc FTensor() {}

  template <typename... Args> BlasCudaConstruc FTensor(Args... args) {
    Meta::array_init(data, args...);
  }

  template <typename Exp, OperatorType exp_type>
  BlasForceInline BlasCudaConstruc auto &
  operator=(const ExpBase<Exp, T, exp_type> &exp) {
    return this->assign(exp.derived_to());
  }

  BlasForceInline BlasCudaConstruc T eval(Uint y, Uint x) const {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline BlasCudaConstruc T eval(Uint idx) const { return data[idx]; }

  BlasForceInline BlasCudaConstruc T &eval_ref(Uint y, Uint x) {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline BlasCudaConstruc T &eval_ref(Uint idx) { return data[idx]; }

  BlasForceInline BlasCudaConstruc T eval_broadcast(Uint y, Uint x) const {
    return data[y * Meta::get_last(shapes...) + x];
  }

  BlasForceInline BlasCudaFunc T operator[](Uint idx) const {
    return data[idx];
  }

  BlasForceInline BlasCudaFunc T &operator[](Uint idx) { return data[idx]; }

  BlasForceInline T *dataptr() const { return const_cast<T *>(data); }

  BlasForceInline T *dataptr() { return data; }

  BlasForceInline constexpr bool AllocCheck() const { return true; }

  BlasForceInline constexpr Uint stride() const {
    return Meta::get_last(shapes...);
  }

  BlasForceInline constexpr auto get_shape() const {
    return Shape<sizeof...(shapes)>{shapes...};
  }

  BlasForceInline constexpr static Uint getSize() {
    return Meta::FoldMul_naive(shapes...);
  }

#ifdef CPP_17
  T data[Meta::FoldMul_naive(shapes...)]{T(0)};
#elif defined(CPP_14)
  // constexpr static Uint _size = Uint{Meta::FoldMul_naive(shapes...)};
  T data[getSize()]{T(0)};
#endif // CPP_17
};

namespace Fixed {
template <typename T, Uint _N>
BlasForceInline BlasCudaConstruc T dot_base_imp(const FTensor<T, _N> &v1,
                                                const FTensor<T, _N> &v2) {
  T res = 0;
  Meta::LoopUnroll<_N - 1>::unroll(
      [] BlasCudaFunc(Uint N, T & res, const FTensor<T, _N> &v1,
                      const FTensor<T, _N> &v2) {
        res += v1.data[N] * v2.data[N];
      },
      res, v1, v2);
  return res;
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_op(const FTensor<T, shapes...> &v1,
                const FTensor<T, shapes...> &v2) {
  auto res = FTensor<T, shapes...>();
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>::unroll(
      [] BlasCudaFunc(const Uint &N, FTensor<T, shapes...> &res,
                      const FTensor<T, shapes...> &v1,
                      const FTensor<T, shapes...> &v2) {
        res.data[N] = Op::eval(v1.data[N], v2.data[N]);
      },
      res, v1, v2);
  return res;
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc void
element_wise_op(FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>::unroll(
      [] BlasCudaFunc(const Uint &N, FTensor<T, shapes...> &res,
                      const FTensor<T, shapes...> &v) {
        res.data[N] = Op::eval(res.data[N], v.data[N]);
      },
      v1, v2);
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_op(const FTensor<T, shapes...> &v1, T v2) {
  auto res = FTensor<T, shapes...>();
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>::unroll(
      [] BlasCudaFunc(Uint N, FTensor<T, shapes...> & res,
                      const FTensor<T, shapes...> &v1,
                      T &v2) { res.data[N] = Op::eval(v1.data[N], v2); },
      res, v1, v2);
  return res;
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc void element_wise_op(FTensor<T, shapes...> &v1,
                                                      T v2) {
  Meta::LoopUnroll<Meta::FoldMul_naive(shapes...) - 1>::unroll(
      [] BlasCudaFunc(Uint N, FTensor<T, shapes...> & res, T & v) {
        res.data[N] = Op::eval(res.data[N], v);
      },
      v1, v2);
}

template <typename T, Uint _N>
BlasForceInline BlasCudaConstruc T
dot_base_naive_imp(const FTensor<T, _N> &v1, const FTensor<T, _N> &v2) {
  T res = 0;
  for (int i = 0; i < _N; ++i)
    res += v1[i] * v2[i];
  return res;
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_naive_op(const FTensor<T, shapes...> &v1,
                      const FTensor<T, shapes...> &v2) {
  auto res = FTensor<T, shapes...>();
  for (int i = 0; i < Meta::FoldMul_naive(shapes...); ++i)
    res[i] = Op::eval(v1[i], v2[i]);
  return res;
}

template <typename Op, typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc FTensor<T, shapes...>
element_wise_naive_op(const FTensor<T, shapes...> &v1, T v2) {
  auto res = FTensor<T, shapes...>();
  for (int i = 0; i < Meta::FoldMul_naive(shapes...); ++i)
    res[i] = Op::eval(v1[i], v2);
  return res;
}
} // namespace Fixed

template <Uint Ndst, Uint Nsrc, typename T,
          typename = std::enable_if_t<(Ndst <= Nsrc)>>
BlasForceInline BlasCudaConstruc FTensor<T, Ndst> *
FTensorDownCast(FTensor<T, Nsrc> &src) {
  return reinterpret_cast<FTensor<T, Ndst> *>(&src);
}

template <typename T, Uint N1, Uint N2,
          typename = std::enable_if_t<(N1 > 2) && (N2 > 2)>>
BlasForceInline BlasCudaConstruc auto cross(const FTensor<T, N1> &v1,
                                            const FTensor<T, N2> &v2) {
  auto res = FTensor<T, 3>();
  res[0] = v1[1] * v2[2] - v1[2] * v2[1];
  res[1] = v1[2] * v2[0] - v1[0] * v2[2];
  res[2] = v1[0] * v2[1] - v1[1] * v2[0];
  return res;
}

template <typename T, Uint N>
BlasForceInline BlasCudaConstruc auto dot(const FTensor<T, N> &v1,
                                          const FTensor<T, N> &v2) {
  return Fixed::dot_base_imp(v1, v2);
}

template <typename T, Uint _N>
BlasForceInline BlasCudaConstruc auto dot(FTensor<T, _N, _N> &mat1,
                                          const FTensor<T, _N> &v) {
  Expblas::FTensor<T, _N> res;
  Meta::LoopUnroll<_N - 1>::unroll(
      [] BlasCudaFunc(Uint N, FTensor<T, _N> & res, FTensor<T, _N, _N> & mat,
                      const FTensor<T, _N> &v) {
        auto vx = reinterpret_cast<FTensor<T, _N> *>(&mat.eval_ref(N, 0));
        res[N] = dot(*vx, v);
      },
      res, mat1, v);
  return res;
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator+(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::plus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc void
operator+=(FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  Fixed::element_wise_op<OP::Binary::plus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator-(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::minus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc void
operator-=(FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  Fixed::element_wise_op<OP::Binary::minus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator*(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::mul>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator/(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::div>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator+(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::plus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator-(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::minus>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator*(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::mul>(v1, v2);
}

template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
operator/(T v2, const FTensor<T, shapes...> &v1) {
  return Fixed::element_wise_op<OP::Binary::div>(v1, v2);
}
// return the L2 value of the vector
template <typename T, Uint N>
BlasForceInline BlasCudaConstruc auto normal(const FTensor<T, N> &v1) {
  return dot(v1, v1);
}

template <typename T, Uint N>
BlasForceInline BlasCudaConstruc auto normalized(const FTensor<T, N> &v1) {
  return T(1) / sqrt(dot(v1, v1)) * v1;
}

// return the largest element of the vector
template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
lowerBound(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::_min>(v1, v2);
}
// return the smallest element of the vector
template <typename T, Uint... shapes>
BlasForceInline BlasCudaConstruc auto
upperBound(const FTensor<T, shapes...> &v1, const FTensor<T, shapes...> &v2) {
  return Fixed::element_wise_op<OP::Binary::_max>(v1, v2);
}
// swap the element of vec2
template <typename T>
BlasForceInline BlasCudaConstruc auto swap_vec2(const FTensor<T, 2> &v) {
  return FTensor<T, 2>{v[1], v[0]};
}

template <typename T, Uint N>
BlasForceInline BlasCudaConstruc auto slerp(const FTensor<T, N> &v1,
                                            const FTensor<T, N> &v2, T t) {
  T theta = dot(v1, v2);
  if (std::fabs(std::fabs(theta) - T(1.0)) < std::numeric_limits<T>::epsilon())
    return (1 - t) * v1 + t * v2;
  theta = acos(theta);
  T st = T(1.0) / sin(theta);
  return st * sin((1 - t) * theta) * v1 + st * sin(t * theta) * v2;
}
} // namespace Expblas