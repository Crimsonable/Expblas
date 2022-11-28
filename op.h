#include "base.h"

namespace OP {
namespace SAVER {
struct plusto {
  template <typename T>
  BlasForceInline BlasCudaFunc static void save(T &src, T val, size_t y = 0,
                                                size_t x = 0) {
    src += val;
  }
};
struct minusto {
  template <typename T>
  BlasForceInline BlasCudaFunc static void save(T &src, T val, size_t y = 0,
                                                size_t x = 0) {
    src -= val;
  }
};
struct multo {
  template <typename T>
  BlasForceInline BlasCudaFunc static void save(T &src, T val, size_t y = 0,
                                                size_t x = 0) {
    src *= val;
  }
};
struct assign {
  template <typename T>
  BlasForceInline BlasCudaFunc static void save(T &src, T val, size_t y = 0,
                                                size_t x = 0) {
    src = val;
  }
};
} // namespace SAVER

namespace Reduce {
struct max {
  template <typename T>
  BlasForceInline BlasCudaFunc static void reduce(T &src, T val) {
    src = src > val ? src : val;
  }

  template <typename T> BlasForceInline BlasCudaFunc static T init() {
    return -FLT_MAX;
  }
};

struct min {
  template <typename T>
  BlasForceInline BlasCudaFunc static void reduce(T &src, T val) {
    src = src < val ? src : val;
  }

  template <typename T> BlasForceInline BlasCudaFunc static T init() {
    return FLT_MAX;
  }
};

struct sum {
  template <typename T>
  BlasForceInline BlasCudaFunc static void reduce(T &src, T val) {
    src = src + val;
  }

  template <typename T> BlasForceInline BlasCudaFunc static T init() {
    return 0;
  }
};
} // namespace Reduce
namespace Unary {
struct abs {
  constexpr static Expblas::Arch arch = Expblas::Arch::Scalar;

  template <typename T> BlasForceInline BlasCudaFunc static T eval(T val) {
    return std::abs(val);
  }
};

struct none {
  constexpr static Expblas::Arch arch = Expblas::Arch::AVX2;

  template <typename T> BlasForceInline BlasCudaFunc static T eval(T val) {
    return val;
  }
};
} // namespace Unary

namespace Binary {
struct plus {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return v1 + v2;
  }
};
struct minus {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return v1 - v2;
  }
};
struct mul {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return v1 * v2;
  }
};
struct div {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return v1 / v2;
  }
};

struct _max {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return max(v1, v2);
  }
};

struct _min {
  template <typename T> BlasForceInline BlasCudaFunc static T eval(T v1, T v2) {
    return min(v1, v2);
  }
};
} // namespace Binary
} // namespace OP