#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/FusedOptimizer.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

template<typename scalar_t>
static inline scalar_t acc_vec(const vec256::Vec256<scalar_t>& v) {
  const int64_t K = vec256::Vec256<scalar_t>::size();
  std::array<scalar_t, K> arr;
  v.store(arr.data());
  return std::accumulate(arr.cbegin(), arr.cend(), scalar_t(0));
}

template <typename scalar_t>
void cpu_lamb_fused_step(
    const Tensor& param,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    const Tensor& grad,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* exp_avg_data = exp_avg.data_ptr<scalar_t>();
  scalar_t* exp_avg_sq_data = exp_avg_sq.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();

  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  int num_threads = at::get_num_threads();
  scalar_t param_norm_acc[num_threads];
  scalar_t rtw_norm_acc[num_threads];
  std::fill_n(&param_norm_acc[0], num_threads, scalar_t(0));
  std::fill_n(&rtw_norm_acc[0], num_threads, scalar_t(0));

  using Vec = vec256::Vec256<scalar_t>;

  int64_t grain_size = 512;

  // update momentum vt and mt
  // also accumulate sum of param_norm and rtw_norm
  at::parallel_for(0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();

    // local pointers
    scalar_t* param_ptr = param_data + begin;
    scalar_t* exp_avg_ptr = exp_avg_data + begin;
    scalar_t* exp_avg_sq_ptr = exp_avg_sq_data + begin;
    scalar_t* grad_ptr = grad_data + begin;

    const int64_t size = end - begin;

    // local sum for param_norm and rtw_norm
    Vec sum1_vec = Vec(scalar_t(0));
    Vec sum2_vec = Vec(scalar_t(0));
    scalar_t sum1_val = scalar_t(0);
    scalar_t sum2_val = scalar_t(0);

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec grad_vec = Vec::loadu(grad_ptr + d);
      Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d) * Vec(scalar_t(beta1)) +
          grad_vec * Vec(scalar_t(1 - beta1));
      Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2)) +
          grad_vec * grad_vec * Vec(scalar_t(1 - beta2));
      Vec adam_step_vec = exp_avg_vec / Vec(scalar_t(bias_correction1)) /
          ((exp_avg_sq_vec / Vec(scalar_t(bias_correction2))).sqrt() + Vec(scalar_t(eps)));

      exp_avg_vec.store(exp_avg_ptr + d);
      exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

      Vec param_vec = Vec::loadu(param_ptr + d);
      adam_step_vec = adam_step_vec + param_vec * Vec(scalar_t(weight_decay));
      // reuse grad to store adam_step
      adam_step_vec.store(grad_ptr + d);

      sum1_vec = sum1_vec + param_vec * param_vec;
      sum2_vec = sum2_vec + adam_step_vec * adam_step_vec;
    }
    for (; d < size; d++) {
      exp_avg_ptr[d] = exp_avg_ptr[d] * beta1 + grad_ptr[d] * (1 - beta1);
      exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * beta2 + grad_ptr[d] * grad_ptr[d] * (1 - beta2);
      scalar_t adam_step_val = (exp_avg_ptr[d] / bias_correction1) /
          (std::sqrt(exp_avg_sq_ptr[d] / bias_correction2) + eps);

      adam_step_val += param_ptr[d] * weight_decay;
      // reuse grad to store adam_step
      grad_ptr[d] = adam_step_val;

      sum1_val += param_ptr[d] * param_ptr[d];
      sum2_val += adam_step_val * adam_step_val;
    }
    sum1_val += acc_vec(sum1_vec);
    sum2_val += acc_vec(sum2_vec);

    param_norm_acc[tid] = sum1_val;
    rtw_norm_acc[tid] = sum2_val;
  });

  // synchronize before update true_ratio
  //
  // [Note]: we could use #pragma omp barrier so that finish within a single omp session
  //   but at::parallel_for partition rule will not guarantee ALL threads in the same
  //   team will be used, so the unused thread will keep on waiting since it never reaches
  //   the barrier.
  //
  scalar_t param_norm_sum = scalar_t(0);
  scalar_t rtw_norm_sum = scalar_t(0);
  for (int64_t tid = 0; tid < num_threads; tid++) {
    param_norm_sum += param_norm_acc[tid];
    rtw_norm_sum += rtw_norm_acc[tid];
  }
  scalar_t true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  // update param
  at::parallel_for(0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    scalar_t* param_ptr = param_data + begin;
    scalar_t* grad_ptr = grad_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec param_vec = Vec::loadu(param_ptr + d) - Vec::loadu(grad_ptr + d) * Vec(scalar_t(learning_rate * true_ratio));
      param_vec.store(param_ptr + d);
    }
    for (; d < size; d++) {
      param_ptr[d] -= grad_ptr[d] * learning_rate * true_ratio;
    }
  });
}

template <typename scalar_t>
void cpu_adagrad_fused_step(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& state_sum,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  scalar_t* param_data = param.data_ptr<scalar_t>();
  scalar_t* grad_data = grad.data_ptr<scalar_t>();
  scalar_t* state_sum_data = state_sum.data_ptr<scalar_t>();

  // update learning rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  using Vec = vec256::Vec256<scalar_t>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    scalar_t* param_ptr = param_data + begin;
    scalar_t* grad_ptr = grad_data + begin;
    scalar_t* state_sum_ptr = state_sum_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
      Vec param_vec = Vec::loadu(param_ptr + d);
      Vec grad_vec = Vec::loadu(grad_ptr + d) + param_vec * Vec(scalar_t(weight_decay));

      Vec sum_vec = Vec::loadu(state_sum_ptr + d) + grad_vec * grad_vec;
      sum_vec.store(state_sum_ptr + d);

      Vec std_vec = sum_vec.sqrt() + Vec(scalar_t(eps));
      param_vec = param_vec - grad_vec / std_vec * Vec(scalar_t(clr));
      param_vec.store(param_ptr + d);
    }
    for (; d < size; d++) {
      scalar_t grad_val = grad_ptr[d] + param_ptr[d] * weight_decay;
      state_sum_ptr[d] += grad_val * grad_val;

      scalar_t std_val = std::sqrt(state_sum_ptr[d]) + eps;
      param_ptr[d] -= grad_val / std_val * clr;
    }
  });
}

void lamb_fused_step_kernel_impl(
    const Tensor& param,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    const Tensor& grad,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "lamb_fused_step", [&] {
    cpu_lamb_fused_step<scalar_t>(
        param, exp_avg, exp_avg_sq, grad, step, beta1, beta2, learning_rate, weight_decay, eps);
  });
}

void adagrad_fused_step_kernel_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& state_sum,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adagrad_fused_step", [&] {
    cpu_adagrad_fused_step<scalar_t>(
        param, grad, state_sum, step, learning_rate, weight_decay, lr_decay, eps);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(lamb_fused_step_kernel, &lamb_fused_step_kernel_impl);
REGISTER_DISPATCH(adagrad_fused_step_kernel, &adagrad_fused_step_kernel_impl);

}} // at::native
