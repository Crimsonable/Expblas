#pragma once
#include "non_packet.h"
#include "packet.h"

namespace Expblas {
template <typename Saver, typename Container, Device device, typename DataType,
          typename Exp, OperatorType exp_type>
struct CPUEngine<Saver, TensorBase<Container, DataType, device>, Exp,
                 exp_type> {
  template <typename... Args>
  BlasForceInline static void
  dispatch(TensorBase<Container, DataType, device> *dst,
           const ExpBase<Exp, DataType, exp_type> &exp, Args &&...args) {
#if BlasUseSIMD
    if (PacketAlignCheck<Exp, DataType, BlasDefaultArch>::Check(
            exp.derived_to()) &&
        PacketAlignCheck<TensorBase<Container, DataType, Device::CPU>, DataType,
                         BlasDefaultArch>::Check(dst->derived_to())) {
      auto wrapper = MakePacketExpWrapper(exp.derived_to());
      PacketExecute<Saver, Container, DataType, Exp, exp_type,
                    Arch(int(BlasDefaultArch) |
                         int(Meta::traits<decltype(wrapper)>::arch))>(
          dst, wrapper, std::forward<Args>(args)...);
    } else {
      auto wrapper_non = MakeExpWrapper(exp.derived_to());
      nonPacketExecute<Saver, Container, DataType, Exp, exp_type>(
          dst, wrapper_non, std::forward<Args>(args)...);
    }
#else
    auto wrapper_non = MakeExpWrapper(exp.derived_to());
    std::cout << typeid(decltype(wrapper_non)).name() << std::endl;
    nonPacketExecute<Saver, Container, device, DataType, Exp, exp_type>(
        dst, wrapper_non, std::forward<Args>(args)...);
#endif // BlasUseSIMD
  }
};
} // namespace Expblas