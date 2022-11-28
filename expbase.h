#pragma once
#include "base.h"
#include "metatools.h"
#include "op.h"
#include "pre_declearation.h"
#include "shape.h"

namespace Expblas {
template <typename Derived, typename DataType, OperatorType type>
class ExpBase {
public:
  Derived *derived() { return static_cast<Derived *>(this); }
  inline const Derived &derived_to() const {
    return *static_cast<const Derived *>(this);
  }
};

template <typename ElementType>
class Scalar : public ExpBase<Scalar<ElementType>, ElementType,
                              OperatorType::container> {
public:
  Scalar(ElementType _data) : data(_data) {}

  BlasForceInline BlasCudaFunc ElementType eval(size_t idx) const {
    return data;
  }

  ElementType data;
};

template <typename Container,
          typename ElementType = typename Meta::traits<Container>::DataType>
class ContainerExtention
    : public ExpBase<Container, ElementType, OperatorType::container> {
public:
  inline Container &operator+=(ElementType val) {
    ExpEngine<OP::SAVER::plusto, ElementType>::eval(this->derived(),
                                                    Scalar<ElementType>(val));
    return *(this->derived());
  }
  inline Container &operator-=(ElementType val) {
    ExpEngine<OP::SAVER::minusto, ElementType>::eval(this->derived(),
                                                     Scalar<ElementType>(val));
    return *(this->derived());
  }
  inline Container &operator*=(ElementType val) {
    ExpEngine<OP::SAVER::multo, ElementType>::eval(this->derived(),
                                                   Scalar<ElementType>(val));
    return *(this->derived());
  }
  inline Container &operator/=(ElementType val) {
    ExpEngine<OP::SAVER::multo, ElementType>::eval(
        this->derived(), Scalar<ElementType>(ElementType(1) / val));
    return *(this->derived());
  }

  template <typename Rhs, OperatorType type>
  BlasForceInline Container &
  assign(const ExpBase<Rhs, ElementType, type> &exp) {
    ExpEngine<OP::SAVER::assign, ElementType>::eval(this->derived(),
                                                    exp.derived_to());
    return *(this->derived());
  }
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType type>
class BinaryExp
    : public ExpBase<BinaryExp<Op, Lhs, Rhs, DataType, type>, DataType, type> {
public:
  BinaryExp(const Lhs &_lhs, const Rhs &_rhs) : lhs(_lhs), rhs(_rhs) {}

  static constexpr Device CheckDevice() {
    static_assert(
        (Device(int(Meta::traits<Lhs>::device) |
                int(Meta::traits<Rhs>::device)) == Device::GPU &&
         Device(int(Meta::traits<Lhs>::device) &
                int(Meta::traits<Rhs>::device)) != Device::CPU) ||
            (Device(int(Meta::traits<Lhs>::device) |
                    int(Meta::traits<Rhs>::device)) == Device::UNIFY &&
             Device(int(Meta::traits<Lhs>::device) &
                    int(Meta::traits<Rhs>::device)) == Device::UNIFY) ||
            (Device(int(Meta::traits<Lhs>::device) |
                    int(Meta::traits<Rhs>::device)) == Device::CPU &&
             Device(int(Meta::traits<Lhs>::device) &
                    int(Meta::traits<Rhs>::device)) != Device::GPU),
        "Devices of BinaryExp don't match");
    return Device(int(Meta::traits<Lhs>::device) |
                  int(Meta::traits<Rhs>::device));
  }

  const Lhs &lhs;
  const Rhs &rhs;
};

template <typename Op, typename Lhs, typename Rhs, typename DataType,
          OperatorType lhs_type, OperatorType rhs_type>
auto MakeBinaryExp(const ExpBase<Lhs, DataType, lhs_type> &_lhs,
                   const ExpBase<Rhs, DataType, rhs_type> &_rhs) {
  return BinaryExp<Op, Lhs, Rhs, DataType,
                   OperatorType(int(lhs_type) | int(rhs_type) |
                                int(OperatorType::keepDim))>(_lhs.derived_to(),
                                                             _rhs.derived_to());
}

template <typename Lhs, typename Rhs, typename DataType, OperatorType lhs_type,
          OperatorType rhs_type>
auto operator+(const ExpBase<Lhs, DataType, lhs_type> &_lhs,
               const ExpBase<Rhs, DataType, rhs_type> &_rhs) {
  return MakeBinaryExp<OP::Binary::plus>(_lhs, _rhs);
}

template <typename Lhs, typename Rhs, typename DataType, OperatorType lhs_type,
          OperatorType rhs_type>
auto operator-(const ExpBase<Lhs, DataType, lhs_type> &_lhs,
               const ExpBase<Rhs, DataType, rhs_type> &_rhs) {
  return MakeBinaryExp<OP::Binary::minus>(_lhs, _rhs);
}

template <typename Lhs, typename Rhs, typename DataType, OperatorType lhs_type,
          OperatorType rhs_type>
auto operator*(const ExpBase<Lhs, DataType, lhs_type> &_lhs,
               const ExpBase<Rhs, DataType, rhs_type> &_rhs) {
  return MakeBinaryExp<OP::Binary::mul>(_lhs, _rhs);
}

template <typename Lhs, typename Rhs, typename DataType, OperatorType lhs_type,
          OperatorType rhs_type>
auto operator/(const ExpBase<Lhs, DataType, lhs_type> &_lhs,
               const ExpBase<Rhs, DataType, rhs_type> &_rhs) {
  return MakeBinaryExp<OP::Binary::div>(_lhs, _rhs);
}

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
class UnaryExp : public ExpBase<UnaryExp<Op, Exp, DataType, exp_type>, DataType,
                                exp_type> {
public:
  UnaryExp(const Exp &_self) : self(_self) {}

  const Exp &self;
};

template <typename Op, typename Exp, typename DataType, OperatorType exp_type>
auto MakeUnaryExp(const ExpBase<Exp, DataType, exp_type> &exp) {
  return UnaryExp<Op, Exp, DataType,
                  OperatorType(int(exp_type) | int(OperatorType::keepDim))>(
      exp.derived_to());
}

template <typename Exp, typename DataType, OperatorType exp_type>
auto abs(const ExpBase<Exp, DataType, exp_type> &exp) {
  return MakeUnaryExp<OP::Unary::abs>(exp);
}

template <typename Exp, typename DataType>
class TransposeExp : public ExpBase<TransposeExp<Exp, DataType>, DataType,
                                    OperatorType::changeDim> {
public:
  TransposeExp(const Exp &exp) : self(exp) {}

  const Exp &self;
};

template <typename Exp, typename DataType, OperatorType exp_type>
auto Transpose(const ExpBase<Exp, DataType, exp_type> &exp) {
  return TransposeExp<Exp, DataType>(exp.derived_to());
}

template <typename LTensor, typename RTensor, typename DataType, bool transL,
          bool transR>
class DotExp
    : public ExpBase<DotExp<LTensor, RTensor, DataType, transL, transR>,
                     DataType, OperatorType::advance> {
public:
  DotExp(const LTensor &lhs, const RTensor &rhs) : lhs(lhs), rhs(rhs) {}

  const LTensor &lhs;
  const RTensor &rhs;
};

template <typename DataType, int LDim, int RDim, Device device>
auto dot(const Tensor<DataType, LDim, device> &lhs,
         const Tensor<DataType, RDim, device> &rhs) {
  return DotExp<Tensor<DataType, LDim, device>, Tensor<DataType, RDim, device>,
                DataType, false, false>(lhs, rhs);
}

template <typename DstExp, typename Exp> constexpr Device DeviceDecsion() {
  static_assert(Meta::traits<DstExp>::device >= Meta::traits<Exp>::device,
                "Device of Dst doesn't match the device of Exp");
  return Meta::traits<DstExp>::device;
}

/*template <typename LTensor, typename RTensor, typename DataType>
auto dot(const TensorBase<LTensor, DataType> &lhs,
         const TensorBase<RTensor, DataType> &rhs) {
  return DotExp<LTensor, RTensor, DataType, false, false>(lhs, rhs);
}*/
} // namespace Expblas