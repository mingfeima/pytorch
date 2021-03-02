#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using lamb_fn = void(*)(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    int64_t, double, double, double, double, double);
using adagrad_fn = void(*)(
    const Tensor&, const Tensor&, const Tensor&,
    int64_t, double, double, double, double);

DECLARE_DISPATCH(lamb_fn, lamb_fused_step_kernel);
DECLARE_DISPATCH(adagrad_fn, adagrad_fused_step_kernel);

}} // at::native
