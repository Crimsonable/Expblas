#pragma once
#include "base.h"
#include <utility>

namespace Expblas {
namespace Meta{
template <typename T> struct traits;

template <typename T1, typename T2> struct TypeResolution;

struct TrueTag {};
struct FalseTag {};
template <bool Flag> struct TagDispatch;
template <> struct TagDispatch<true> { using type = TrueTag; };
template <> struct TagDispatch<false> { using type = FalseTag; };

/*template <typename... Args> constexpr auto FoldMul(Args... args) {
  return (... * args);
}*/

template <typename Head>
BlasCudaConstruc constexpr auto FoldMul_naive(Head head) {
  return head;
}

template <typename Head, typename... Tail>
BlasCudaConstruc constexpr auto FoldMul_naive(Head head, Tail... tail) {
  return head * FoldMul_naive(tail...);
}

//template<Uint...I> BlasCudaConstruc constexpr auto FoldMul_naive(I... args) {
//  Uint ret = 1;
//  ret = ((ret = ret * args), ...);
//  return ret;
//}

template <int idx> struct getI;
template <int idx> struct getI {
  template <typename Head, typename... Tail>
  BlasCudaConstruc constexpr static auto get(Head head, Tail... tail) {
    return getI<idx - 1>::get(tail...);
  }
};

template <> struct getI<0> {
  template <typename Head, typename... Tail>
  BlasCudaConstruc constexpr static auto get(Head head, Tail... tail) {
    return head;
  }
};

template <typename... Args>
BlasCudaConstruc constexpr auto get_last(Args... args) {
  return getI<sizeof...(Args) - 1>::get(args...);
}

template <size_t N> struct LoopUnroll;
template <size_t N> struct LoopUnroll {
  template <typename Func, typename... Args>
  static BlasCudaConstruc void unroll(Func &&f, Args &&...args) {
    f(N, std::forward<Args>(args)...);
    LoopUnroll<N - 1>::unroll(std::forward<Func>(f),
                              std::forward<Args>(args)...);
  }
};	

template <> struct LoopUnroll<0> {
  template <typename Func, typename... Args>
  static BlasCudaConstruc void unroll(Func &&f, Args &&...args) {
    f(0, std::forward<Args>(args)...);
  }
};

template <int N> struct init_helper;
template <int N> struct init_helper {
  template <typename T, typename... Args>
  static BlasCudaConstruc void init(T *data, Args &&...args) {
    data[N] = getI<N>::get(std::forward<Args>(args)...);
    init_helper<N - 1>::init(data, std::forward<Args>(args)...);
  }
};

template <> struct init_helper<0> {
  template <typename T, typename... Args>
  static BlasCudaConstruc void init(T *data, Args &&...args) {
    data[0] = getI<0>::get(std::forward<Args>(args)...);
  }
};

template <typename T, typename... Args>
BlasCudaConstruc void array_init(T *data, Args &&...args) {
  init_helper<sizeof...(args) - 1>::init(data, std::forward<Args>(args)...);
}

} // namespace Meta
} // namespace Expblas

/* template <typename T, typename... Args, size_t... N>
BlasCudaConstruc void init_helper(T *data, Args &&...args,std::index_sequence<N...>)
{
  ((data[N] = args), ...);
}

template <typename T, typename... Args>
BlasCudaConstruc void array_init(T *data, Args &&...args) {
  init_helper<T, Args...>(data, std::forward<Args>(args)...,std::make_index_sequence<sizeof...(args)>{});
}*/
