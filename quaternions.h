#pragma once
#include "ftensor.h"

namespace Expblas {
namespace quat {
template <typename T> class Quat : public FTensor<T, 4> {
public:
  Quat(T r, T i, T j, T k) : FTensor<T, 4>(r, i, j, k) {}

  Quat(const FTensor<T, 4> &v) : FTensor<T, 4>(v) {}

  Quat(const FTensor<T, 3> &v, T r) : FTensor<T, 4>(r, v[0], v[1], v[2]) {}

  Quat() : FTensor<T, 4>() {}

  BlasForceInline BlasCudaConstruc void unit() { *this = normalized(*this); }

  BlasForceInline BlasCudaConstruc T real() const { return this->data[0]; }

  BlasForceInline BlasCudaConstruc T &real() { return this->data[0]; }

  BlasForceInline BlasCudaConstruc const FTensor<T, 3> &imaginary() const {
    return *reinterpret_cast<const FTensor<T, 3> *>((const char *)this +
                                                    sizeof(T));
  }

  BlasForceInline BlasCudaConstruc FTensor<T, 3> &imaginary_ref() {
    return *reinterpret_cast<FTensor<T, 3> *>((char *)this + sizeof(T));
  }
};
// mul operation for quaternions
template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> operator*(const Quat<T> &q1,
                                                   const Quat<T> &q2) {
  Quat<T> res;
  const FTensor<T, 3> &v1 = q1.imaginary(), &v2 = q2.imaginary();
  res[0] = q1[0] * q2[0] - dot(v1, v2);
  res.imaginary_ref() = q1[0] * v2 + q2[0] * v1 + cross(v1, v2);
  return res;
}

template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> operator*(const Quat<T> &q1,
                                                   const FTensor<T, 3> &q2) {
  Quat<T> res;
  const FTensor<T, 3> &v1 = q1.imaginary();
  res[0] = -dot(v1, q2);
  res.imaginary_ref() = q1[0] * q2 + cross(v1, q2);
  return res;
}
// return the conjugate of a quaternion
template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> conjugate(const Quat<T> &q) {
  Quat<T> res;
  res.imaginary_ref() = T(-1) * q.imaginary();
  res[0] = q[0];
  return res;
}
// return the inverse of a quaternion
template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> inverse(const Quat<T> &q) {
  return T(1) / normal(q) * conjugate(q);
}
// use if quat q is an unit quat
template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> inverse_unit(const Quat<T> &q) {
  Quat<T> res;
  res[0] = q[0];
  res.imaginary_ref() = T(-1.0) * q.imaginary();
  return res;
}
// convert a quaternion(i.e. q=a+bi+cj+dk) to mat4, which is a skew symmetric
// matrix:
//[ a, d,-c, b]
//[-d, a,-b, c]
//[ c, b, a,-d]
//[-b, c, d, a]
template <typename T>
BlasForceInline BlasCudaConstruc FTensor<T, 4, 4> toMat4(const Quat<T> &q) {
  FTensor<T, 4, 4> res;
  res[0] = q[0];
  res[1] = q[3];
  res[2] = -q[2];
  res[3] = q[1];

  res[5] = q[0];
  res[6] = -q[1];
  res[7] = -q[2];

  res[10] = q[0];
  res[11] = -q[3];

  res[15] = q[0];

  res[4] = -q[3];
  res[8] = q[2];
  res[12] = -q[1];

  res[9] = q[1];
  res[13] = q[2];

  res[14] = q[3];
  return res;
}

// origin and dst has to be normalized
template <typename T>
BlasForceInline BlasCudaConstruc Quat<T> rotation(const FTensor<T, 3> &origin,
                                                  const FTensor<T, 3> &dst) {
  FTensor<T, 3> rotation_axis = cross(origin, dst);
  Quat<T> res;
  T cosTheta = dot(origin, dst);

  T s = sqrt((T(1) + cosTheta) * T(2));
  T invs = static_cast<T>(1) / s;

  return Quat<T>(s * T(0.5f), rotation_axis[0] * invs, rotation_axis[1] * invs,
                 rotation_axis[2] * invs);
}

template <typename T>
BlasForceInline BlasCudaConstruc FTensor<T, 3> rotate(const FTensor<T, 3> &v,
                                                      const Quat<T> &rot) {
  return (rot * v * inverse_unit(rot)).imaginary();
}

template <typename T>
BlasForceInline BlasCudaConstruc FTensor<T, 3>
rotate(const FTensor<T, 3> &v, const FTensor<T, 3> &origin,
       const FTensor<T, 3> &dst) {
  auto rot = rotation(origin, dst);
  return (rot * v * inverse_unit(rot)).imaginary();
}

template <typename T>
BlasForceInline BlasCudaConstruc FTensor<T, 3>
rotate(const FTensor<T, 3> &v, const FTensor<T, 3> &axis, T theta) {
  Quat<T> rot(T(sin(theta * 0.5)) * axis, cos(theta * 0.5));
  return (rot * v * inverse_unit(rot)).imaginary();
}

template <typename T>
BlasForceInline BlasCudaConstruc auto log(const Quat<T> &q) {
  Quat<T> res;
  res.imaginary_ref() = acos(q.real()) * q.imaginary();
  return res;
}
#ifdef CPP_17
template <typename T, template <typename> class... Q>
BlasForceInline BlasCudaConstruc Quat<T> blend(const Q<T> &...quats) {
  Quat<T> res;
  T theta = 0;
  res = (log(quats) + ...);
  theta = Expblas::normal(res.imaginary());
  res.imaginary_ref() = sin(theta) * Expblas::normalized(res.imaginary_ref());
  res.real() = cos(theta);
  return Expblas::normalized(res);
}
#endif // CPP_17
} // namespace quat
} // namespace Expblas