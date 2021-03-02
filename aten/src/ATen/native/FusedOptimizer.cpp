#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <ATen/native/FusedOptimizer.h>

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> lamb_fused_step_cpu(
    const Tensor& param_,
    const Tensor& exp_avg_,
    const Tensor& exp_avg_sq_,
    const Tensor& grad_,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  TORCH_CHECK(learning_rate >= 0,
      "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(eps >= 0,
      "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(beta1 >= 0 && beta1 < 1,
      "Expect 0.0 <= beta1 < 1.0, got", beta1);
  TORCH_CHECK(beta2 >= 0 && beta2 < 1,
      "Expect 0.0 <= beta2 < 1.0, got", beta2);
  TORCH_CHECK(weight_decay >= 0,
      "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(param_.sizes() == grad_.sizes(),
      "Expect param and grad have the same sizes, param sizes: ",
      param_.sizes(), "; grad sizes: ", grad_.sizes());
  TORCH_CHECK(param_.sizes() == exp_avg_.sizes(),
      "Expect param and exp_avg have the same sizes, param sizes: ",
      param_.sizes(), "; exp_avg sizes: ", exp_avg_.sizes());
  TORCH_CHECK(param_.sizes() == exp_avg_sq_.sizes(),
      "Expect param and exp_avg_sq_ have the same sizes, param sizes: ",
      param_.sizes(), "; exp_avg_sq sizes: ", exp_avg_sq_.sizes());

  auto param = param_.contiguous();
  auto exp_avg = exp_avg_.contiguous();
  auto exp_avg_sq = exp_avg_sq_.contiguous();
  auto grad = grad_.contiguous();

  lamb_fused_step_kernel(
      kCPU, param, exp_avg, exp_avg_sq, grad, step, beta1, beta2, learning_rate, weight_decay, eps);

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!exp_avg_.is_contiguous()) {
    exp_avg_.copy_(exp_avg);
  }
  if (!exp_avg_sq_.is_contiguous()) {
    exp_avg_sq_.copy_(exp_avg_sq);
  }

  return std::make_tuple(param_, exp_avg_, exp_avg_sq_);
}

std::tuple<Tensor, Tensor> adagrad_fused_step_cpu(
    const Tensor& param_,
    const Tensor& grad_,
    const Tensor& state_sum_,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  TORCH_CHECK(learning_rate >= 0,
      "Expect learning rate >= 0.0, got ", learning_rate);
  TORCH_CHECK(lr_decay >= 0,
      "Expect lr_decay >=0.0 , got ", lr_decay);
  TORCH_CHECK(eps >= 0,
      "Expect eps >= 0.0, got ", eps);
  TORCH_CHECK(weight_decay >= 0,
      "Expect weight_decay >= 0.0, got ", weight_decay);

  TORCH_CHECK(param_.sizes() == grad_.sizes(),
      "Expect param and grad have the same sizes, param sizes: ",
      param_.sizes(), "; grad sizes: ", grad_.sizes());
  TORCH_CHECK(param_.sizes() == state_sum_.sizes(),
      "Expect param and state_sum have the same sizes, param sizes: ",
      param_.sizes(), "; state_sum sizes: ", state_sum_.sizes());

  auto param = param_.contiguous();
  auto grad = grad_.contiguous();
  auto state_sum = state_sum_.contiguous();

  adagrad_fused_step_kernel(
      kCPU, param, grad, state_sum, step, learning_rate, weight_decay, lr_decay, eps);

  if (!param_.is_contiguous()) {
    param_.copy_(param);
  }
  if (!state_sum_.is_contiguous()) {
    state_sum_.copy_(state_sum);
  }

  return std::make_tuple(param_, state_sum_);
}

DEFINE_DISPATCH(lamb_fused_step_kernel);
DEFINE_DISPATCH(adagrad_fused_step_kernel);

}} // at::native
