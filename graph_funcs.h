#pragma once
#include "ftensor.h"
#include "quaternions.h"

namespace Expblas {
template <StorageMajor Major, typename T>
BlasForceInline BlasCudaConstruc auto
lookAt(const FTensor<T, StorageMajor::Vector, 3> &position,
       const FTensor<T, StorageMajor::Vector, 3> &target,
       const FTensor<T, StorageMajor::Vector, 3> &up) {
  FTensor<T, Major, 4, 4> mat;
  mat.eval_ref(3, 3) = T(1.0);

  FTensor<T, StorageMajor::Vector, 3> z = normalized(position - target);
  mat.eval_ref(2, 0) = z[0];
  mat.eval_ref(2, 1) = z[1];
  mat.eval_ref(2, 2) = z[2];

  FTensor<T, StorageMajor::Vector, 3> x = normalized(cross(up, z));
  mat.eval_ref(0, 0) = x[0];
  mat.eval_ref(0, 1) = x[1];
  mat.eval_ref(0, 2) = x[2];

  FTensor<T, StorageMajor::Vector, 3> y = normalized(cross(z, x));
  mat.eval_ref(1, 0) = y[0];
  mat.eval_ref(1, 1) = y[1];
  mat.eval_ref(1, 2) = y[2];
  mat.eval_ref(0, 3) = -dot(x, position);
  mat.eval_ref(1, 3) = -dot(y, position);
  mat.eval_ref(2, 3) = -dot(z, position);

  return mat;
}

BlasForceInline BlasCudaConstruc auto
perspective(float fovy, float src_ratio, float near_plane, float far_plane) {
  float t = tan(0.5f * fovy);
  float r = src_ratio * t;
  float f_n_revert = 1.0f / (far_plane - near_plane);

  mat4f res;
  res.eval_ref(0, 0) = float(1.0) / r;
  res.eval_ref(1, 1) = float(1.0) / t;
  res.eval_ref(2, 2) = -(far_plane + near_plane) * f_n_revert;
  res.eval_ref(3, 2) = -float(1.0);
  res.eval_ref(2, 3) = -float(2.0) * far_plane * near_plane * f_n_revert;
  return res;
}

template <typename T, StorageMajor Major>
BlasForceInline BlasCudaConstruc auto
translate(FTensor<T, Major, 4, 4> &mat,
          const FTensor<T, StorageMajor::Vector, 3> &trans) {
  mat.eval_ref(0, 3) += trans[0];
  mat.eval_ref(1, 3) += trans[1];
  mat.eval_ref(2, 3) += trans[2];
  return mat;
}

template <typename T, StorageMajor Major>
BlasForceInline BlasCudaConstruc auto
scale(FTensor<T, Major, 4, 4> &mat,
      const FTensor<T, StorageMajor::Vector, 3> &scale) {
  mat.eval_ref(0, 0) *= scale[0];
  mat.eval_ref(1, 1) *= scale[1];
  mat.eval_ref(2, 2) *= scale[2];
  return mat;
}
} // namespace Expblas