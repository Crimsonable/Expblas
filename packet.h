#pragma once
#include "base.h"
#include "expbase.h"
#include "metatools.h"
#include "shape.h"
#include "storage.h"
#include <immintrin.h>

namespace Expblas {
namespace SIMD {
template <typename T, Arch arch> struct PacketType;

template <> struct PacketType<float, Arch::AVX2> {
  using type = __m256;
  BlasForceInline static type load(float *src) { return _mm256_load_ps(src); }
  BlasForceInline static void save(float *target, type src) {
    _mm256_store_ps(target, src);
  }
  BlasForceInline static type set1(float src) { return _mm256_set1_ps(src); }
  BlasForceInline static type add(type v1, type v2) {
    return _mm256_add_ps(v1, v2);
  }
  BlasForceInline static type minus(type v1, type v2) {
    return _mm256_sub_ps(v1, v2);
  }
  BlasForceInline static type mul(type v1, type v2) {
    return _mm256_mul_ps(v1, v2);
  }
  BlasForceInline static type div(type v1, type v2) {
    return _mm256_div_ps(v1, v2);
  }
};

template <> struct PacketType<float, Arch::SSE> {
  using type = __m128;
  BlasForceInline static type load(float *src) { return _mm_load_ps(src); }
  BlasForceInline static void save(float *target, type src) {
    _mm_store_ps(target, src);
  }
  BlasForceInline static type set1(float src) { return _mm_set1_ps(src); }
  BlasForceInline static type add(type v1, type v2) {
    return _mm_add_ps(v1, v2);
  }
  BlasForceInline static type minus(type v1, type v2) {
    return _mm_sub_ps(v1, v2);
  }
  BlasForceInline static type mul(type v1, type v2) {
    return _mm_mul_ps(v1, v2);
  }
  BlasForceInline static type div(type v1, type v2) {
    return _mm_div_ps(v1, v2);
  }
};

template <typename T> struct PacketType<T, Arch::Scalar> {
  using type = T;
  BlasForceInline static type load(T *src) { return *src; }
  BlasForceInline static void save(T *target, type src) { *target = src; }
  BlasForceInline static type set1(T src) { return src; }
  BlasForceInline static type add(type v1, type v2) { return v1 + v2; }
  BlasForceInline static type minus(type v1, type v2) { return v1 - v2; }
  BlasForceInline static type mul(type v1, type v2) { return v1 * v2; }
  BlasForceInline static type div(type v1, type v2) { return v1 / v2; }
};

} // namespace SIMD

template <typename Exp> struct ArchCheck;

template <typename T> struct ArchCheck<Scalar<T>> {
  constexpr static Arch arch = BlasDefaultArch;
};

template <typename T, int Dim> struct ArchCheck<Tensor<T, Dim, Device::CPU>> {
  constexpr static Arch arch = BlasDefaultArch;
};
// Check if FTensor suit for the Pack requirement,FTensor's size has to be
// interger times of Packsize
template <typename T, size_t... shapes>
struct ArchCheck<FTensor<T, shapes...>> {
  constexpr static Arch arch = Arch(
      int(BlasDefaultArch) |
      (Meta::FoldMul_naive(shapes...) %
               (sizeof(typename SIMD::PacketType<T, BlasDefaultArch>::type) /
                sizeof(T))
           ? int(Arch::Scalar)
           : int(BlasDefaultArch)));
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
struct ArchCheck<BinaryExp<Op, Lhs, Rhs, DataType, type>> {
  constexpr static Arch arch =
      Arch(int(BlasDefaultArch) | int(ArchCheck<Lhs>::arch) |
           int(ArchCheck<Rhs>::arch));
};

template <typename Exp, typename DataType>
struct ArchCheck<TransposeExp<Exp, DataType>> {
  constexpr static Arch arch = Arch::Scalar;
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
struct ArchCheck<UnaryExp<Op, Exp, DataType, exp_type>> {
  constexpr static Arch arch =
      Arch(int(BlasDefaultArch) | int(ArchCheck<Exp>::arch) | int(Op::arch));
};

template <typename Exp, typename T, Arch arch> struct PacketAlignCheck {
  BlasForceInline static bool Check(const Exp &exp) { return false; }
};

template <typename T, Arch arch> struct PacketAlignCheck<Scalar<T>, T, arch> {
  BlasForceInline static bool Check(const Scalar<T> &exp) { return true; }
};

template <typename T, Arch arch> struct PacketAlignCheck<void, T, arch> {
  BlasForceInline static bool Check(const void *ptr) {
    return reinterpret_cast<size_t>(ptr) % sizeof(T) ? false : true;
  }
};

template <typename T, Arch arch> struct PacketAlignCheck<size_t, T, arch> {
  BlasForceInline static bool Check(size_t stride) {
    return stride % sizeof(typename SIMD::PacketType<T, arch>::type);
  }
};

template <typename Container, typename T, Arch arch>
struct PacketAlignCheck<TensorBase<Container, T, Device::CPU>, T, arch> {
  BlasForceInline static bool
  Check(const TensorBase<Container, T, Device::CPU> &exp) {
    return PacketAlignCheck<void, T, arch>::Check(exp.dataptr());
  }
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type, Arch arch>
struct PacketAlignCheck<BinaryExp<Op, Lhs, Rhs, DataType, type>, DataType,
                        arch> {
  BlasForceInline static bool
  Check(const BinaryExp<Op, Lhs, Rhs, DataType, type> &exp) {
    return PacketAlignCheck<Lhs, DataType, arch>::Check(exp.lhs) &&
           PacketAlignCheck<Rhs, DataType, arch>::Check(exp.rhs);
  }
};

template <typename Exp, typename DataType, Arch arch>
struct PacketAlignCheck<TransposeExp<Exp, DataType>, DataType, arch> {
  BlasForceInline static bool Check(const TransposeExp<Exp, DataType> &exp) {
    return false;
  }
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type,
          Arch arch>
struct PacketAlignCheck<UnaryExp<Op, Exp, DataType, exp_type>, DataType, arch> {
  BlasForceInline static bool
  Check(const UnaryExp<Op, Exp, DataType, exp_type> &exp) {
    return true;
  }
};

template <typename Derived, typename DataType, OperatorType type, Arch arch>
struct PacketAlignCheck<ExpBase<Derived, DataType, type>, DataType, arch> {
  BlasForceInline static bool
  Check(const ExpBase<Derived, DataType, type> &exp) {
    return false;
  }
};

template <typename T, Arch arch> struct Packet {
  using PacketOp = SIMD::PacketType<T, arch>;

  Packet(typename PacketOp::type data) : data(data) {}
  Packet() {}

  BlasForceInline Packet<T, arch> load(T *src) {
    data = PacketOp::load(src);
    return *this;
  }

  BlasForceInline static void save(T *dst, Packet<T, arch> data, size_t y = 0,
                                   size_t x = 0) {
    PacketOp::save(dst, data.data);
  }

  BlasForceInline Packet<T, arch> set1(T src) {
    data = PacketOp::set1(src);
    return *this;
  }
  BlasForceInline Packet<T, arch> operator+(Packet<T, arch> other) {
    return Packet<T, arch>(PacketOp::add(data, other.data));
  }
  BlasForceInline Packet<T, arch> operator-(Packet<T, arch> other) {
    return Packet<T, arch>(PacketOp::minus(data, other.data));
  }
  BlasForceInline Packet<T, arch> operator*(Packet<T, arch> other) {
    return Packet<T, arch>(PacketOp::mul(data, other.data));
  }
  BlasForceInline Packet<T, arch> operator/(Packet<T, arch> other) {
    return Packet<T, arch>(PacketOp::div(data, other.data));
  }

  typename PacketOp::type data;
};

template <typename T, Arch arch> struct PacketExpWrapper<Scalar<T>, arch> {
  PacketExpWrapper(const Scalar<T> &data) : data(data.data) {}

  BlasForceInline auto eval_packet(size_t y, size_t x) const {
    ConstexprIF(arch == Arch::Scalar)
      return data;
    else {
      auto res = Packet<T, arch>();
      res.set1(data);
      return res;
    }
  }

  BlasForceInline T eval(size_t y, size_t x) const { return data; }

  const T data;
};

template <typename Container, Arch arch> struct PacketExpWrapper {
  using T = typename Meta::traits<Container>::DataType;

  PacketExpWrapper(const TensorBase<Container, T, Device::CPU> &data)
      : dataptr(data.dataptr()), stride(data.stride()) {}

  BlasForceInline auto eval_packet(size_t y, size_t x) const {
    ConstexprIF(arch == Arch::Scalar)
      return dataptr[y * stride + x];
    else
      return Packet<T, arch>().load(&dataptr[y * stride + x]);
  }
  BlasForceInline T eval(size_t y, size_t x) const {
    return dataptr[y * stride + x];
  }

  T *dataptr;
  size_t stride;
};

template <typename T, Arch arch, size_t... shapes>
struct PacketExpWrapper<FTensor<T, shapes...>, arch> {

  PacketExpWrapper(const FTensor<T, shapes...> &_data) : data(_data) {}

  BlasForceInline auto eval_packet(size_t y, size_t x) const {
    ConstexprIF(arch == Arch::Scalar)
      return data.eval(y, x);
    else
      return Packet<T, arch>().load(&data.eval_ref(y, x));
  }
  BlasForceInline T eval(size_t y, size_t x) const { return data.eval(y, x); }

  FTensor<T, shapes...> data;
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type, Arch arch>
struct PacketExpWrapper<BinaryExp<Op, Lhs, Rhs, DataType, type>, arch> {
  PacketExpWrapper(const BinaryExp<Op, Lhs, Rhs, DataType, type> &exp)
      : lhs(exp.lhs), rhs(exp.rhs) {}

  BlasForceInline auto eval_packet(size_t y, size_t x) const {
    return Op::eval(lhs.eval_packet(y, x), rhs.eval_packet(y, x));
  }
  BlasForceInline DataType eval(size_t y, size_t x) const {
    return Op::eval(lhs.eval(y, x), rhs.eval(y, x));
  }

  const PacketExpWrapper<Lhs, arch> lhs;
  const PacketExpWrapper<Rhs, arch> rhs;
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type,
          Arch arch>
struct PacketExpWrapper<UnaryExp<Op, Exp, DataType, exp_type>, arch> {
  PacketExpWrapper(const UnaryExp<Op, Exp, DataType, exp_type> &exp)
      : self(exp.self) {}

  BlasForceInline auto eval_packet(size_t y, size_t x) const {
    return Op::eval(self.eval_packet(y, x));
  }
  BlasForceInline DataType eval(size_t y, size_t x) const {
    return Op::eval(self.eval(y, x));
  }

  const PacketExpWrapper<Exp, arch> self;
};

template <typename Exp, typename DataType, Arch arch>
struct PacketExpWrapper<TransposeExp<Exp, DataType>, arch> {
  PacketExpWrapper(const TransposeExp<Exp, DataType> &exp) : self(exp.self) {}

  BlasForceInline DataType eval_packet(size_t y, size_t x) const {
    return self.eval(x, y);
  }
  BlasForceInline DataType eval(size_t y, size_t x) const {
    return self.eval(x, y);
  }

  const PacketExpWrapper<Exp, arch> self;
};

template <typename T>
BlasForceInline auto MakePacketExpWrapper(const Scalar<T> &data) {
  return PacketExpWrapper<Scalar<T>, BlasDefaultArch>(data);
}

template <typename Container, typename T>
BlasForceInline auto
MakePacketExpWrapper(const TensorBase<Container, T, Device::CPU> &data) {
  return PacketExpWrapper<TensorBase<Container, T, Device::CPU>,
                          BlasDefaultArch>(data);
}

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
BlasForceInline auto
MakePacketExpWrapper(const BinaryExp<Op, Lhs, Rhs, DataType, type> &exp) {
  return PacketExpWrapper<
      BinaryExp<Op, Lhs, Rhs, DataType, type>,
      ArchCheck<BinaryExp<Op, Lhs, Rhs, DataType, type>>::arch>(
      exp.derived_to());
}

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
BlasForceInline auto
MakePacketExpWrapper(const UnaryExp<Op, Exp, DataType, exp_type> &exp) {
  return PacketExpWrapper<
      UnaryExp<Op, Exp, DataType, exp_type>,
      ArchCheck<UnaryExp<Op, Exp, DataType, exp_type>>::arch>(exp.derived_to());
}

template <typename Exp, typename DataType>
BlasForceInline auto
MakePacketExpWrapper(const TransposeExp<Exp, DataType> &exp) {
  return PacketExpWrapper<TransposeExp<Exp, DataType>, Arch::Scalar>(
      exp.derived_to());
}

template <typename Saver, typename Container, typename DataType, typename Exp,
          OperatorType exp_type, Arch arch, typename... Args>
inline void PacketExecute(TensorBase<Container, DataType, Device::CPU> *dst,
                          const PacketExpWrapper<Exp, arch> &exp,
                          Args &&...args) {
  auto _dst = dst->derived();
  Shape<2> dshape = _dst->get_shape().flat2D();
  size_t packet_stride =
      sizeof(typename SIMD::PacketType<DataType, arch>::type) /
      sizeof(DataType);

#pragma omp parallel for
  for (int64_t y = 0; y < dshape[0]; ++y) {
    for (int64_t x = 0; x < dshape[1] - dshape[1] % packet_stride;
         x += packet_stride)
      Packet<DataType, arch>::save(&_dst->eval_ref(y, x), exp.eval_packet(y, x),
                                   y, x, std::forward<Args>(args)...);
    for (int64_t x = dshape[1] - dshape[1] % packet_stride; x < dshape[1]; ++x)
      Saver::save(_dst->eval_ref(y, x), exp.eval(y, x), y, x,
                  std::forward<Args>(args)...);
  }
}

} // namespace Expblas