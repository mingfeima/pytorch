#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using offset_range_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(offset_range_fn, offset_range_kernel);

}} // at::native
