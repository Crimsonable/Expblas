#pragma once
#include "ftensor.h"

namespace Expblas {
template <typename T>
BlasForceInline BlasCudaConstruc auto lookAt(const FTensor<T, 3> &position,
                                             const FTensor<T, 3> &target,
                                             const FTensor<T, 3> &up) {
  FTensor<T, 4, 4> mat;
  mat.eval_ref(0, 3) = position[0];
  mat.eval_ref(1, 3) = position[1];
  mat.eval_ref(2, 3) = position[2];
  mat.eval_ref(3, 3) = T(1.0);

  FTensor<T, 3> z = normalized(position - target);
  mat.eval_ref(0, 2) = z[0];
  mat.eval_ref(1, 2) = z[1];
  mat.eval_ref(2, 2) = z[2];

  FTensor<T, 3> x = normalized(cross(up, z));
  mat.eval_ref(0, 0) = x[0];
  mat.eval_ref(1, 0) = x[1];
  mat.eval_ref(2, 0) = x[2];

  FTensor<T, 3> y = normalized(cross(z, x));
  mat.eval_ref(0, 1) = y[0];
  mat.eval_ref(1, 1) = y[1];
  mat.eval_ref(2, 1) = y[2];

  return mat;
}

BlasForceInline BlasCudaConstruc auto
perspective(float fovy, float src_ratio, float near_plane, float far_plane) {
  float t = tan(0.5f * fovy) * near_plane;
  float r = src_ratio * t;
  float f_n_revert = 1.0f / (far_plane - near_plane);
  mat4f res(near_plane / r, 0, 0, 0, 0, near_plane / t, 0, 0, 0, 0,
            -(far_plane + near_plane) * f_n_revert,
            -2 * far_plane * near_plane * f_n_revert, 0, 0, -1, 0);

  return res;
}
} // namespace Expblas