#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor> mkldnn_sgd_step(
    const Tensor& param, const Tensor& gradient, const Tensor& momentum_buf,
    Scalar lr, Scalar momentum, Scalar dampening, Scalar weight_decay, bool nesterov) {
  AT_ERROR("mkldnn_sgd_step: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

namespace at { namespace native {

using scalar_t = float;

std::tuple<Tensor, Tensor> mkldnn_sgd_step(
    const Tensor& param, const Tensor& gradient, const Tensor& momentum_buf,
    Scalar learning_rate, Scalar momentum, Scalar dampening, Scalar weight_decay, bool nesterov) {

  auto n = param.numel();

  auto p = param;
  auto dp = gradient;
  auto buf = momentum_buf.numel() == 0 ? at::zeros_like(p) : momentum_buf;
  auto lr = learning_rate.to<scalar_t>();
  auto mom = momentum.to<scalar_t>();
  auto dam = dampening.to<scalar_t>();
  auto wd = weight_decay.to<scalar_t>();

  auto p_ = p.data<scalar_t>();
  auto dp_ = dp.data<scalar_t>();
  auto buf_ = buf.data<scalar_t>();

  parallel_for(0, n, 1, [=](int64_t begin, int64_t end){
    for (int64_t index = begin; index < end; index++) {
      if (wd != static_cast<scalar_t>(0)) {
        dp_[index] += wd * p_[index];
      }
      if (mom != static_cast<scalar_t>(0)) {
        if (momentum_buf.numel() == 0) {
          buf_[index] = dp_[index];
        } else {
          buf_[index] = buf_[index] * mom + (1 - dam) * dp_[index];
        }
        if (nesterov) {
          dp_[index] += mom * buf_[index];
        } else {
          dp_[index] = buf_[index];
        }
      }
      p_[index] -= lr * dp_[index];
    }
  });

  return std::tuple<Tensor, Tensor>{p, buf};
}

}} // namespace at::native

#endif
