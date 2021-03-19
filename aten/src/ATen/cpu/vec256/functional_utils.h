#pragma once

#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>

// This header specializes funcions from functional.h with <scalar_t=BFloat16>
//
// Note that we already have specializes member of Vec256<scalar_t> for BFloat16
// so the following functionw would run smoothly:
//   using Vec = Vec256<BFloat16>;
//   Vec one = Vec(BFloat16(1));
//   vec256::map([](Vec x) { return one / (one + x.exp()); }, y_ptr, x_ptr, N);
//
// Why we still need to specializes "funtional"?
//   If we do specialization at Vec256<> level, the above example would need 3 pairs of
// conversion of bf16->fp32/fp32->bf16, each for ".exp()", "+" and "/".
//   If we do specialization at vec256::map<>() level, we have only 1 pair of conversion
// of bf16->fp32/fp32->bf16, for the input and output BFloat16 vector only.
//
// Functionalities in this file only do data type conversion for input and output
// vector (reduce functionalities will only convert the final scalar back to bf16).
// Compared to Vec256<> specialization,
//   1. better performance since we have less dtype conversion;
//   2. less rounding error since immediate results are kept in fp32;
//   3. accumulation done on data type of fp32.
//
//  If you plan to extend this file, make sure add unit test at
//    aten/src/ATen/test/vec256_test_all_types.cpp
//

namespace at { namespace vec256 {

template <typename scalar_t>
struct VecScalarType { };

template <> struct VecScalarType<float> { using type = float; };
template <> struct VecScalarType<double> { using type = double; };
template <> struct VecScalarType<BFloat16> { using type = float; };

// we can't use at::acc_type here since it will use double as
// acc type for float on CPU.
template <typename scalar_t>
using vec_scalar_t = typename VecScalarType<scalar_t>::type;

template <typename scalar_t>
struct ReduceAll {
  template <typename Op>
  static inline scalar_t apply(
      const Op& vec_fun,
      const scalar_t* data,
      int64_t size) {
    return vec256::reduce_all(vec_fun, data, size);
  }
};

template <typename scalar_t>
struct Reduce2All {
  template <typename Op1, typename Op2>
  static inline std::pair<scalar_t, scalar_t> apply(
      const Op1& vec_fun1,
      const Op2& vec_fun2,
      const scalar_t* data,
      int64_t size) {
    return vec256::reduce2_all(vec_fun1, vec_fun2, data, size);
  }
};

template <typename scalar_t>
struct MapReduceAll {
  template<typename MapOp, typename ReduceOp>
  static inline scalar_t apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const scalar_t* data,
      int64_t size) {
    return vec256::map_reduce_all(map_fun, red_fun, data, size);
  }
};

template <typename scalar_t>
struct Map2ReduceAll {
  template <typename MapOp, typename ReduceOp>
  static inline scalar_t apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const scalar_t* data,
      const scalar_t* data2,
      int64_t size) {
    return vec256::map2_reduce_all(map_fun, red_fun, data, data2, size);
  }
};

template <typename scalar_t>
struct Map3ReduceAll {
  template <typename MapOp, typename ReduceOp>
  static inline scalar_t apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const scalar_t* data,
      const scalar_t* data2,
      const scalar_t* data3,
      int64_t size) {
    return vec256::map3_reduce_all(map_fun, red_fun, data, data2, data3, size);
  }
};

template <typename scalar_t>
struct Map {
  template <typename Op>
  static inline void apply(
      const Op& vec_fun,
      scalar_t* output_data,
      const scalar_t* input_data,
      int64_t size) {
    vec256::map(vec_fun, output_data, input_data, size);
  }
};

template <typename scalar_t>
struct Map2 {
  template <typename Op>
  static inline void apply(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size) {
      vec256::map2(vec_fun, output_data, input_data, input_data2, size);
    }
};

template <typename scalar_t>
struct Map3 {
  template <typename Op>
  static inline void apply(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size) {
      vec256::map3(vec_fun, output_data, input_data, input_data2, input_data3, size);
    }
};

template <typename Op>
inline BFloat16 reduce_all_bf16(
    const Op& vec_fun,
    const BFloat16* data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = fVec::set(data_fvec0, vec_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(vec_fun, data_fvec0, fVec::size());
    } else {
      return vec_reduce_all<float>(vec_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = vec_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      acc_fvec0 = vec_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, vec_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      acc_fvec0 = fVec::set(acc_fvec0, vec_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = vec_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(vec_fun, acc_fvec0, fVec::size());
}

template <typename Op1, typename Op2>
inline std::pair<BFloat16, BFloat16> reduce2_all_bf16(
    const Op1& vec_fun1,
    const Op2& vec_fun2,
    const BFloat16* data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      fVec acc1_fvec = fVec::set(data_fvec0, vec_fun1(data_fvec0, data_fvec1), size - fVec::size());
      fVec acc2_fvec = fVec::set(data_fvec0, vec_fun2(data_fvec0, data_fvec1), size - fVec::size());
      return std::pair<BFloat16, BFloat16>(
          vec_reduce_all<float>(vec_fun1, acc1_fvec, fVec::size()),
          vec_reduce_all<float>(vec_fun2, acc2_fvec, fVec::size()));
    } else {
      return std::pair<BFloat16, BFloat16>(
          vec_reduce_all<float>(vec_fun1, data_fvec0, size),
          vec_reduce_all<float>(vec_fun2, data_fvec0, size));
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc1_fvec0, acc1_fvec1;
  std::tie(acc1_fvec0, acc1_fvec1) = convert_bfloat16_float(acc_bvec);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc_bvec);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
    acc1_fvec1 = vec_fun1(acc1_fvec1, data_fvec1);
    acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
    acc2_fvec1 = vec_fun2(acc2_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      acc1_fvec0 = vec_fun1(acc1_fvec0, data_fvec0);
      acc1_fvec1 = fVec::set(acc1_fvec1, vec_fun1(acc1_fvec1, data_fvec1), size - d - fVec::size());
      acc2_fvec0 = vec_fun2(acc2_fvec0, data_fvec0);
      acc2_fvec1 = fVec::set(acc2_fvec1, vec_fun2(acc2_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      acc1_fvec0 = fVec::set(acc1_fvec0, vec_fun1(acc1_fvec0, data_fvec0), size - d);
      acc2_fvec0 = fVec::set(acc2_fvec0, vec_fun2(acc2_fvec0, data_fvec0), size - d);
    }
  }
  acc1_fvec0 = vec_fun1(acc1_fvec0, acc1_fvec1);
  acc2_fvec0 = vec_fun2(acc2_fvec0, acc2_fvec1);
  return std::pair<BFloat16, BFloat16>(
      vec_reduce_all<float>(vec_fun1, acc1_fvec0, fVec::size()),
      vec_reduce_all<float>(vec_fun2, acc2_fvec0, fVec::size()));
}

template <typename MapOp, typename ReduceOp>
inline BFloat16 map_reduce_all_bf16(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  acc_fvec0 = map_fun(acc_fvec0);
  acc_fvec1 = map_fun(acc_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    data_fvec0 = map_fun(data_fvec0);
    data_fvec1 = map_fun(data_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0);
      data_fvec1 = map_fun(data_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename MapOp, typename ReduceOp>
inline BFloat16 map2_reduce_all_bf16(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc2_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename MapOp, typename ReduceOp>
inline BFloat16 map3_reduce_all_bf16(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const BFloat16* data,
    const BFloat16* data2,
    const BFloat16* data3,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  if (size < bVec::size()) {
    bVec data_bvec = bVec::loadu(data, size);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2, size);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3, size);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    if (size > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      data_fvec0 = fVec::set(data_fvec0, red_fun(data_fvec0, data_fvec1), size - fVec::size());
      return vec_reduce_all<float>(red_fun, data_fvec0, fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      return vec_reduce_all<float>(red_fun, data_fvec0, size);
    }
  }
  int64_t d = bVec::size();
  bVec acc_bvec = bVec::loadu(data);
  fVec acc_fvec0, acc_fvec1;
  std::tie(acc_fvec0, acc_fvec1) = convert_bfloat16_float(acc_bvec);
  bVec acc2_bvec = bVec::loadu(data2);
  fVec acc2_fvec0, acc2_fvec1;
  std::tie(acc2_fvec0, acc2_fvec1) = convert_bfloat16_float(acc2_bvec);
  bVec acc3_bvec = bVec::loadu(data3);
  fVec acc3_fvec0, acc3_fvec1;
  std::tie(acc3_fvec0, acc3_fvec1) = convert_bfloat16_float(acc3_bvec);
  acc_fvec0 = map_fun(acc_fvec0, acc2_fvec0, acc3_fvec0);
  acc_fvec1 = map_fun(acc_fvec1, acc2_fvec1, acc3_fvec1);
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
    data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
    acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
    acc_fvec1 = red_fun(acc_fvec1, data_fvec1);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(data3 + d, size - d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    if (size - d > fVec::size()) {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      data_fvec1 = map_fun(data_fvec1, data2_fvec1, data3_fvec1);
      acc_fvec0 = red_fun(acc_fvec0, data_fvec0);
      acc_fvec1 = fVec::set(acc_fvec1, red_fun(acc_fvec1, data_fvec1), size - d - fVec::size());
    } else {
      data_fvec0 = map_fun(data_fvec0, data2_fvec0, data3_fvec0);
      acc_fvec0 = fVec::set(acc_fvec0, red_fun(acc_fvec0, data_fvec0), size - d);
    }
  }
  acc_fvec0 = red_fun(acc_fvec0, acc_fvec1);
  return vec_reduce_all<float>(red_fun, acc_fvec0, fVec::size());
}

template <typename Op>
inline void map_bf16(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename Op>
inline void map2_bf16(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    const BFloat16* input_data2,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <typename Op>
inline void map3_bf16(
    const Op& vec_fun,
    BFloat16* output_data,
    const BFloat16* input_data,
    const BFloat16* input_data2,
    const BFloat16* input_data3,
    int64_t size) {
  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;
  int64_t d = 0;
  for (; d < size - (size % bVec::size()); d += bVec::size()) {
    bVec data_bvec = bVec::loadu(input_data + d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d);
  }
  if (size - d > 0) {
    bVec data_bvec = bVec::loadu(input_data + d, size - d);
    fVec data_fvec0, data_fvec1;
    std::tie(data_fvec0, data_fvec1) = convert_bfloat16_float(data_bvec);
    bVec data2_bvec = bVec::loadu(input_data2 + d, size - d);
    fVec data2_fvec0, data2_fvec1;
    std::tie(data2_fvec0, data2_fvec1) = convert_bfloat16_float(data2_bvec);
    bVec data3_bvec = bVec::loadu(input_data3 + d, size - d);
    fVec data3_fvec0, data3_fvec1;
    std::tie(data3_fvec0, data3_fvec1) = convert_bfloat16_float(data3_bvec);
    fVec output_fvec0 = vec_fun(data_fvec0, data2_fvec0, data3_fvec0);
    fVec output_fvec1 = vec_fun(data_fvec1, data2_fvec1, data3_fvec1);
    bVec output_bvec = convert_float_bfloat16(output_fvec0, output_fvec1);
    output_bvec.store(output_data + d, size - d);
  }
}

template <>
struct ReduceAll<BFloat16> {
  template <typename Op>
  static inline BFloat16 apply(
      const Op& vec_fun,
      const BFloat16* data,
      int64_t size) {
    return vec256::reduce_all_bf16(vec_fun, data, size);
  }
};

template <>
struct Reduce2All<BFloat16> {
  template <typename Op1, typename Op2>
  static inline std::pair<BFloat16, BFloat16> apply(
      const Op1& vec_fun1,
      const Op2& vec_fun2,
      const BFloat16* data,
      int64_t size) {
    return vec256::reduce2_all_bf16(vec_fun1, vec_fun2, data, size);
  }
};

template <>
struct MapReduceAll<BFloat16> {
  template <typename MapOp, typename ReduceOp>
  static inline BFloat16 apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const BFloat16* data,
      int64_t size) {
    return vec256::map_reduce_all_bf16(map_fun, red_fun, data, size);
  }
};

template <>
struct Map2ReduceAll<BFloat16> {
  template <typename MapOp, typename ReduceOp>
  static inline BFloat16 apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const BFloat16* data,
      const BFloat16* data2,
      int64_t size) {
    return vec256::map2_reduce_all_bf16(map_fun, red_fun, data, data2, size);
  }
};

template <>
struct Map3ReduceAll<BFloat16> {
  template <typename MapOp, typename ReduceOp>
  static inline BFloat16 apply(
      const MapOp& map_fun,
      const ReduceOp& red_fun,
      const BFloat16* data,
      const BFloat16* data2,
      const BFloat16* data3,
      int64_t size) {
    return vec256::map3_reduce_all_bf16(map_fun, red_fun, data, data2, data3, size);
  }
};

template <>
struct Map<BFloat16> {
  template <typename Op>
  static inline void apply(
      const Op& vec_fun,
      BFloat16* output_data,
      const BFloat16* input_data,
      int64_t size) {
    vec256::map_bf16(vec_fun, output_data, input_data, size);
  }
};

template <>
struct Map2<BFloat16> {
  template <typename Op>
  static inline void apply(
      const Op& vec_fun,
      BFloat16* output_data,
      const BFloat16* input_data,
      const BFloat16* input_data2,
      int64_t size) {
    vec256::map2_bf16(vec_fun, output_data, input_data, input_data2, size);
  }
};

template <>
struct Map3<BFloat16> {
  template <typename Op>
  static inline void apply(
      const Op& vec_fun,
      BFloat16* output_data,
      const BFloat16* input_data,
      const BFloat16* input_data2,
      const BFloat16* input_data3,
      int64_t size) {
    vec256::map3_bf16(vec_fun, output_data, input_data, input_data2, input_data3, size);
  }
};

}} // namespace at::vec256
