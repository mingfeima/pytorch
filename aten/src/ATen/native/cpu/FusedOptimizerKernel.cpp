#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/native/FusedOptimizer.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>

namespace at { namespace native {

namespace {

using namespace vec256;

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
    const Tensor& param2,
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

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <>
void cpu_lamb_fused_step<BFloat16>(
    const Tensor& param,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    const Tensor& grad,
    const Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  TORCH_CHECK(param.scalar_type() == kBFloat16,
      "cpu_lamb_fused_step: expect param to be BFloat16");
  TORCH_CHECK(grad.scalar_type() == kBFloat16,
      "cpu_lamb_fused_step: expect grad to be BFloat16");
  TORCH_CHECK(exp_avg.scalar_type() == kFloat,
      "cpu_lamb_fused_step: expect exp_avg to be float32");
  TORCH_CHECK(exp_avg_sq.scalar_type() == kFloat,
      "cpu_lamb_fused_step: expect exp_avg_sq to be float32");
  TORCH_CHECK(param2.scalar_type() == kBFloat16,
      "cpu_adagrad_fused_step: expect param2 to be BFloat16");

  BFloat16* param_data = param.data_ptr<BFloat16>();
  float* exp_avg_data = exp_avg.data_ptr<float>();
  float* exp_avg_sq_data = exp_avg_sq.data_ptr<float>();
  BFloat16* grad_data = grad.data_ptr<BFloat16>();
  BFloat16* param2_data = param2.data_ptr<BFloat16>();

  double bias_correction1 = 1 - std::pow(beta1, step);
  double bias_correction2 = 1 - std::pow(beta2, step);

  int num_threads = at::get_num_threads();
  float param_norm_acc[num_threads];
  float rtw_norm_acc[num_threads];
  std::fill_n(&param_norm_acc[0], num_threads, float(0));
  std::fill_n(&rtw_norm_acc[0], num_threads, float(0));

  // for float32 path, we can reuse grad to store adam_step
  // but for bfloat16 path, this can't be done since grad is in bfloat16
  // and we want to keep adam_step to be float32
  int64_t numel = param.numel();
  Tensor workspace = at::empty({numel}, exp_avg.options());
  float* workspace_data = workspace.data_ptr<float>();

  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;

  int64_t grain_size = 512;

  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    int tid = at::get_thread_num();

    // local pointers
    BFloat16* param_ptr = param_data + begin;
    float* exp_avg_ptr = exp_avg_data + begin;
    float* exp_avg_sq_ptr = exp_avg_sq_data + begin;
    BFloat16* grad_ptr = grad_data + begin;
    BFloat16* param2_ptr = param2_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    // local sum for param_norm and rtw_norm
    fVec sum1_fvec = fVec(float(0));
    fVec sum2_fvec = fVec(float(0));
    float sum1_val = float(0);
    float sum2_val = float(0);

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec grad_bvec = bVec::loadu(grad_ptr + d);
      fVec grad_fvec, grad_fvec2;
      std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

      fVec exp_avg_fvec = fVec::loadu(exp_avg_ptr + d) * fVec(float(beta1)) +
          grad_fvec * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec = fVec::loadu(exp_avg_sq_ptr + d) * fVec(float(beta2)) +
          grad_fvec * grad_fvec * fVec(float(1 - beta2));
      fVec adam_step_fvec = exp_avg_fvec / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec / fVec(float(bias_correction2))).sqrt() + fVec(float(eps)));

      fVec exp_avg_fvec2 = fVec::loadu(exp_avg_ptr + d + fVec::size()) * fVec(float(beta1)) +
          grad_fvec2 * fVec(float(1 - beta1));
      fVec exp_avg_sq_fvec2 = fVec::loadu(exp_avg_sq_ptr + d + fVec::size()) * fVec(float(beta2)) +
          grad_fvec2 * grad_fvec2 * fVec(float(1 - beta2));
      fVec adam_step_fvec2 = exp_avg_fvec2 / fVec(float(bias_correction1)) /
          ((exp_avg_sq_fvec2 / fVec(float(bias_correction2))).sqrt() + fVec(float(eps)));

      exp_avg_fvec.store(exp_avg_ptr + d);
      exp_avg_fvec2.store(exp_avg_ptr + d + fVec::size());
      exp_avg_sq_fvec.store(exp_avg_sq_ptr + d);
      exp_avg_sq_fvec2.store(exp_avg_sq_ptr + d + fVec::size());

      bVec param_bvec = bVec::loadu(param_ptr + d);
      bVec param2_bvec = bVec::loadu(param2_ptr + d);
      fVec param_fvec, param_fvec2;
      std::tie(param_fvec, param_fvec2) = pack_bfloat16_float(param_bvec, param2_bvec);

      adam_step_fvec = adam_step_fvec + param_fvec * fVec(float(weight_decay));
      adam_step_fvec2 = adam_step_fvec2 + param_fvec2 * fVec(float(weight_decay));
      adam_step_fvec.store(workspace_ptr + d);
      adam_step_fvec2.store(workspace_ptr + d + fVec::size());

      sum1_fvec += param_fvec * param_fvec;
      sum1_fvec += param_fvec2 * param_fvec2;
      sum2_fvec += adam_step_fvec * adam_step_fvec;
      sum2_fvec += adam_step_fvec2 * adam_step_fvec2;
    }
    for (; d < size; d++) {
      float grad_val = float(grad_ptr[d]);
      exp_avg_ptr[d] = exp_avg_ptr[d] * beta1 + grad_val * (1 - beta1);
      exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * beta2 + grad_val * grad_val * (1 - beta2);
      float adam_step_val = (exp_avg_ptr[d] / bias_correction1) /
          (std::sqrt(exp_avg_sq_ptr[d] / bias_correction2) + eps);

      float param_val = pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
      adam_step_val += param_val * weight_decay;
      workspace_ptr[d] = adam_step_val;

      sum1_val += param_val * param_val;
      sum2_val += adam_step_val * adam_step_val;
    }
    sum1_val += acc_vec(sum1_fvec);
    sum2_val += acc_vec(sum2_fvec);

    param_norm_acc[tid] = sum1_val;
    rtw_norm_acc[tid] = sum2_val;
  });

  float param_norm_sum = float(0);
  float rtw_norm_sum = float(0);
  for (int64_t tid = 0; tid < num_threads; tid++) {
    param_norm_sum += param_norm_acc[tid];
    rtw_norm_sum += rtw_norm_acc[tid];
  }
  float true_ratio = std::sqrt(param_norm_sum) / std::sqrt(rtw_norm_sum);

  // update param
  at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    BFloat16* param_ptr = param_data + begin;
    BFloat16* param2_ptr = param2_data + begin;
    float* workspace_ptr = workspace_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec param_bvec = bVec::loadu(param_ptr + d);
      bVec param2_bvec = bVec::loadu(param2_ptr + d);
      fVec param_fvec, param_fvec2;
      std::tie(param_fvec, param_fvec2) = pack_bfloat16_float(param_bvec, param2_bvec);

      param_fvec -= fVec::loadu(workspace_ptr + d) * fVec(float(learning_rate * true_ratio));
      param_fvec2 -= fVec::loadu(workspace_ptr + d + fVec::size()) * fVec(float(learning_rate * true_ratio));

      std::tie(param_bvec, param2_bvec) = unpack_float_bfloat16(param_fvec, param_fvec2);
      param_bvec.store(param_ptr + d);
      param2_bvec.store(param2_ptr + d);
    }
    for (; d < size; d++) {
      float param_val = pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
      param_val -= workspace_ptr[d] * learning_rate * true_ratio;
      std::tie(param_ptr[d], param2_ptr[d]) = unpack_float_bfloat16(param_val);
    }
  });
}

#endif

template <typename scalar_t>
void cpu_adagrad_fused_step(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& state_sum,
    const Tensor& param2,
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

#if defined(CPU_CAPABILITY_AVX2) && !defined(_MSC_VER)

template <>
void cpu_adagrad_fused_step<BFloat16>(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& state_sum,
    const Tensor& param2,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  TORCH_CHECK(param.scalar_type() == kBFloat16,
      "cpu_adagrad_fused_step: expect param to be BFloat16");
  TORCH_CHECK(grad.scalar_type() == kBFloat16,
      "cpu_adagrad_fused_step: expect grad to be BFloat16");
  TORCH_CHECK(state_sum.scalar_type() == kFloat,
      "cpu_adagrad_fused_step: expect stats_sum to be float32");
  TORCH_CHECK(param2.scalar_type() == kBFloat16,
      "cpu_adagrad_fused_step: expect param2 to be BFloat16");

  BFloat16* param_data = param.data_ptr<BFloat16>();
  BFloat16* grad_data = grad.data_ptr<BFloat16>();
  float* state_sum_data = state_sum.data_ptr<float>();
  BFloat16* param2_data = param2.data_ptr<BFloat16>();

  // update learning rate
  double clr = learning_rate / (1 + (step - 1) * lr_decay);

  using bVec = vec256::Vec256<BFloat16>;
  using fVec = vec256::Vec256<float>;

  int64_t grain_size = 512;

  // purely element-wise operations
  at::parallel_for(0, param.numel(), grain_size, [&](int64_t begin, int64_t end) {
    // local pointers
    BFloat16* param_ptr = param_data + begin;
    BFloat16* grad_ptr = grad_data + begin;
    float* state_sum_ptr = state_sum_data + begin;
    BFloat16* param2_ptr = param2_data + begin;

    const int64_t size = end - begin;

    int64_t d = 0;
    for (; d < size - (size % bVec::size()); d += bVec::size()) {
      bVec param_bvec = bVec::loadu(param_ptr + d);
      bVec param2_bvec = bVec::loadu(param2_ptr + d);
      fVec param_fvec, param_fvec2;
      std::tie(param_fvec, param_fvec2) = pack_bfloat16_float(param_bvec, param2_bvec);

      bVec grad_bvec = bVec::loadu(grad_ptr + d);
      fVec grad_fvec, grad_fvec2;
      std::tie(grad_fvec, grad_fvec2) = convert_bfloat16_float(grad_bvec);

      grad_fvec = grad_fvec + param_fvec * fVec(float(weight_decay));
      grad_fvec2 = grad_fvec2 + param_fvec2 * fVec(float(weight_decay));

      fVec sum_fvec = fVec::loadu(state_sum_ptr + d) + grad_fvec * grad_fvec;
      fVec sum_fvec2 = fVec::loadu(state_sum_ptr + d + fVec::size()) + grad_fvec2 * grad_fvec2;
      sum_fvec.store(state_sum_ptr + d);
      sum_fvec2.store(state_sum_ptr + d + fVec::size());

      fVec std_fvec = sum_fvec.sqrt() + fVec(float(eps));
      fVec std_fvec2 = sum_fvec2.sqrt() + fVec(float(eps));
      param_fvec = param_fvec - grad_fvec / std_fvec * fVec(float(clr));
      param_fvec2 = param_fvec2 - grad_fvec2 / std_fvec2 * fVec(float(clr));

      std::tie(param_bvec, param2_bvec) = unpack_float_bfloat16(param_fvec, param_fvec2);
      param_bvec.store(param_ptr + d);
      param2_bvec.store(param2_ptr + d);
    }
    for (; d < size; d++) {
      float param_val = pack_bfloat16_float(param_ptr[d], param2_ptr[d]);
      float grad_val = float(grad_ptr[d]) + param_val * weight_decay;
      state_sum_ptr[d] += grad_val * grad_val;

      float std_val = std::sqrt(state_sum_ptr[d]) + eps;
      param_val -= grad_val / std_val * clr;
      std::tie(param_ptr[d], param2_ptr[d]) = unpack_float_bfloat16(param_val);
    }
  });
}

#endif

void lamb_fused_step_kernel_impl(
    const Tensor& param,
    const Tensor& exp_avg,
    const Tensor& exp_avg_sq,
    const Tensor& grad,
    const Tensor& param2,
    int64_t step,
    double beta1,
    double beta2,
    double learning_rate,
    double weight_decay,
    double eps) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, param.scalar_type(), "lamb_fused_step", [&] {
    cpu_lamb_fused_step<scalar_t>(
        param, exp_avg, exp_avg_sq, grad, param2, step, beta1, beta2, learning_rate, weight_decay, eps);
  });
}

void adagrad_fused_step_kernel_impl(
    const Tensor& param,
    const Tensor& grad,
    const Tensor& state_sum,
    const Tensor& param2,
    int64_t step,
    double learning_rate,
    double weight_decay,
    double lr_decay,
    double eps) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, param.scalar_type(), "adagrad_fused_step", [&] {
    cpu_adagrad_fused_step<scalar_t>(
        param, grad, state_sum, param2, step, learning_rate, weight_decay, lr_decay, eps);
  });
}

} // anonymous namespace

REGISTER_DISPATCH(lamb_fused_step_kernel, &lamb_fused_step_kernel_impl);
REGISTER_DISPATCH(adagrad_fused_step_kernel, &adagrad_fused_step_kernel_impl);

}} // at::native
