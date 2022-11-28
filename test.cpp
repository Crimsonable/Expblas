#include "ftensor.h"
#include "quaternions.h"
#include "range.h"
#include "tensor.h"
#include <chrono>
#include <iostream>

using namespace Expblas;
using namespace std::chrono;

template <typename Container, typename T, Device device,
          typename = std::enable_if_t<(device <= Device::CPU)>>
void printTensor(const TensorBase<Container, T, device> &_t) {
  auto t = _t.derived_to();
  auto shape = t.get_shape().flat2D();
  for (int i = 0; i < shape[0]; ++i) {
    for (int j = 0; j < shape[1]; ++j)
      std::cout << t.eval(i, j) << " ";
    std::cout << std::endl;
  }
}

// void benchmark(Uint x, Uint y, int axis) {
//  Shape<2> s({x, y});
//  Shape<2> s_reduce = s;
//  s_reduce[axis] = 1;
//  auto t_d = Ones<float, 2, Device::GPU>(s);
//  auto t_h = Zeros<float, 2, Device::CPU>(s_reduce);
//  // t_d = t_d + t_d;
//  t_h = t_h + t_h;
//  float time = 0;
//  cudaEvent_t t1, t2;
//  cudaEventCreate(&t1);
//  cudaEventCreate(&t2);
//  cudaEventRecord(t1, 0);
//  auto t_d_res = sum(t_d, axis);
//  cudaEventRecord(t2, 0);
//  cudaEventSynchronize(t1);
//  cudaEventSynchronize(t2);
//  cudaEventElapsedTime(&time, t1, t2);
//  std::cout << time << std::endl;
//  std::cout << x * y * 4 / time / 1024 / 1024 << std::endl;
//  Copy(t_h, t_d_res);
//  std::cout << t_h.eval(0, 0) << std::endl;
//  freeSpace(t_d);
//  freeSpace(t_d_res);
//  freeSpace(t_h);
//}

// void benchmark_cwise(Uint x, Uint y) {
//  Shape<2> s({x, y});
//  auto t_d = Ones<float, 2, Device::GPU>(s);
//  auto t_d1 = Ones<float, 2, Device::GPU>(s);
//  auto t_dres = Ones<float, 2, Device::GPU>(s);
//  auto t_h = Zeros<float, 2, Device::CPU>(s);
//  float time = 0;
//  cudaEvent_t t1, t2;
//  cudaEventCreate(&t1);
//  cudaEventCreate(&t2);
//  cudaEventRecord(t1, 0);
//  t_dres = t_d + t_d1;
//  cudaEventRecord(t2, 0);
//  cudaEventSynchronize(t1);
//  cudaEventSynchronize(t2);
//  cudaEventElapsedTime(&time, t1, t2);
//  std::cout << time << std::endl;
//  std::cout << x * y * 4 / time / 1024 / 1024 << std::endl;
//  Copy(t_h, t_dres);
//  auto tt = sum(sum(t_h, 1), 0);
//  std::cout << tt.eval(0, 0) << std::endl;
//  freeSpace(t_d);
//  freeSpace(t_dres);
//  freeSpace(t_h);
//}
//
// inline void simple_gemv(float *dst, float *lhs, float *rhs) {
//  for (int i = 0; i < 4; ++i) {
//    float val = 0;
//    for (int j = 0; j < 4; ++j)
//      val += lhs[j + i * 4] * rhs[j];
//    dst[i] = val;
//  }
//}

/*void benchmark_gemv(Uint n) {
  Shape<2> s({4, 4});
  Shape<1> s1({4});
  auto t_res = Zeros<float, 1, Device::CPU>(s1);
  //t_res = 1.0f;
  auto t_l = Ones<float, 2, Device::CPU>(s);
  auto t_r = Ones<float, 1, Device::CPU>(s1);
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i)
    BlasExcute<float, Device::CPU>::gemv(4, 4, t_l.dataptr(), 4, false,
                                         t_r.dataptr(), 1, t_res.dataptr(), 1);
  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i)
    simple_gemv(t_res.dataptr(), t_l.dataptr(), t_r.dataptr());
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << duration_cast<duration<double>>(t1 - t0).count() << std::endl;
  std::cout << duration_cast<duration<double>>(t2 - t1).count() << std::endl;
}*/

template <typename Func, typename... Args>
void timer(Func &&f, Args &&...args) {
  auto t0 = std::chrono::high_resolution_clock::now();
  f(std::forward<Args>(args)...);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << duration_cast<duration<double>>(t1 - t0).count() << std::endl;
}

void FTensor_add(FTensor<float, 4> &v, FTensor<float, 4> &v1,
                 FTensor<float, 4> &v2, int n) {
  for (int i = 0; i < n; ++i)
    v = v1 + v2;
}

struct vec4 {
  float data[4];
};

void navie_add(vec4 &v, vec4 &v1, vec4 &v2, int n) {
  for (int i = 0; i < n; ++i) {
    v.data[0] = v1.data[0] + v2.data[0];
    v.data[1] = v1.data[1] + v2.data[1];
    v.data[2] = v1.data[2] + v2.data[2];
    v.data[3] = v1.data[3] + v2.data[3];
  }
}

int main() {
  // benchmark(2048, 2048, 1);
  // benchmark_gemv(10240);
  /*vec4 n, n1, n2;
  FTensor<float, 4> v, v1{1, 1, 1, 1}, v2{1, 1, 1, 1};
  v = abs(v1 + v2);
  printTensor(v);
  timer(navie_add, n, n1, n2, 1 << 20);
  timer(FTensor_add, v, v1, v2, 1 << 20);
  printTensor(v);*/
  /*FTensor<float, 4> v0, v{1, 2, 3, 4};
  v0 = 1.0f + v;
  printTensor(v0);*/
  // benchmark_cwise(768, 512);
  // FTensor<float, 4> v{1, 2, 3, 4};
  // Tensor<float, 2, Device::GPU> t(Shape<2>{4, 4});
  // Tensor<float, 2, Device::CPU> t_c(Shape<2>{4, 4});
  // allocSpace(t);
  // allocSpace(t_c);
  // auto _r = range(t, {0, 0}, {1, 4});
  //// t = v + Scalar<float>{1};
  // Copy(t_c, t);
  float theta = 90.0 / 180.0 * 3.1415926;
  quat::Quat<float> q(cos(theta * 0.5), 0, sin(theta * 0.5), 0);
  quat::Quat<float> q2(cos(theta * 0.5), 0, 0, sin(theta * 0.5));
  FTensor<float, 3> v(1, 0, 0);
  auto blended = quat::blend(q, q2);
  auto rt = quat::rotate(v, blended);
  printTensor(rt);
  system("pause");
  return 0;
}