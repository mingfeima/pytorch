#pragma once

#include <ATen/ATen.h>
#include <ideep.hpp>

using namespace ideep;

namespace at { namespace native {

// MKLDNN supported data type, currently only float32
inline tensor::data_type get_mkldnn_dtype(const ScalarType scalar_type) {
  if (scalar_type == at::kFloat) {
    return tensor::data_type::f32;
  }
  AT_ERROR("get_mkldnn_dtype: unsupported data type");
}

inline tensor::data_type get_mkldnn_dtype(const Tensor& input) {
  return get_mkldnn_dtype(input.scalar_type());
}

inline tensor::data_type get_mkldnn_dtype(const caffe2::TypeMeta dtype) {
  return get_mkldnn_dtype(typeMetaToScalarType(dtype));
}

}} // namespace at::native
