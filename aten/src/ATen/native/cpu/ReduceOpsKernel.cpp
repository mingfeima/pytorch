#include <numeric>
#include <iterator>
#include <algorithm>

#include <ATen/Dispatch.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/cpu/vec256/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>

#include <c10/util/Optional.h>
#include <ATen/AccumulateType.h>

namespace at { namespace native { namespace {

using namespace vec256;

template <typename scalar_t>
static inline void cpu_cumsum_lastdim_kernel(
    Tensor& result,
    const Tensor& self,
    int64_t dim) {
  const auto input_ndim = self.dim();
  TORCH_CHECK(dim == input_ndim - 1,
      "cpu_cumsum_lastdim_kernel: expect dim to be ", input_ndim - 1, " got ", dim);
  TORCH_CHECK(self.scalar_type() == result.scalar_type(),
      "cpu_cumsum_lastdim_kernel: expect same data type for self and result");

  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  int64_t N = self.size(dim);
  int64_t M = self.numel() / N;
  const scalar_t* self_data = self.data_ptr<scalar_t>();
  scalar_t* result_data = result.data_ptr<scalar_t>();

  int64_t T = at::get_num_threads();

  // bytes per core for each chunk, set to 256KB (L2 cache reside)
  constexpr int64_t CHUNK_SIZE_PER_CORE = 256 * 1024 / sizeof(scalar_t);
  int64_t CHUNK_SIZE = std::max(1, CHUNK_SIZE_PER_CORE / M * T);
  int64_t K = divup(N, CHUNK_SIZE);

  // offset value per chunk
  std::vector<scalar_t> outer_offsets(M, scalar_t(0));

  // offset value per thread
  std::vector<scalar_t> inner_offsets(M * T, scalar_t(0));

  for (int64_t k = 0; k < K; k++) {
    int64_t k_begin = k * CHUNK_SIZE;
    int64_t k_end = std::min(k_begin + CHUNK_SIZE, N);

    // Parallel Path I: accumulate locally per thread
    at::parallel_for(k_begin, k_end, 1, [&](int64_t begin, int64_t end) {
      int64_t tid = at::get_thread_num();
      for (int64_t m = 0; m < M; m++) {
        const scalar_t* self_ptr = self_data + m * N + begin;
        scalar_t* result_ptr = result_data + m * N + begin;
        int64_t len = end - begin;

        prefix_sum<scalar_t>(self_ptr, result_ptr, scalar_t(0), len);
        inner_offsets[m * T + tid] = result_ptr[len - 1];
      }
    });

    // update offset value for each thread
    for (int64_t m = 0; m < M; m++) {
      for (int64_t t = T - 1; t >= 0; t--) {
        scalar_t offset = scalar_t(0);
        for (int64_t i = t - 1; i >= 0; i--) {
          offset += inner_offsets[m * T + i];
        }
        inner_offsets[m * T + t] = offset;
      }
    }

    // Parallel Path II: apply offset (result should be in L2)
    at::parallel_for(k_begin, k_end, 1, [&](int64_t begin, int64_t end) {
      int64_t tid = at::get_thread_num();
      for (int64_t m = 0; m < M; m++) {
        const scalar_t* self_ptr = self_data + m * N + begin;
        scalar_t* result_ptr = result_data + m * N + begin;
        int64_t len = end - begin;

        scalar_t offset = outer_offsets[m] + inner_offsets[m * T + tid];
        vec256::map<scalar_t>(
            [=](Vec256<scalar_t> x) { return x + Vec256<scalar_t>(offset); },
            result_ptr,
            result_ptr,
            len);
      }
    });

    // reinit inner offset value
    std::fill(inner_offsets.begin(), inner_offsets.end(), scalar_t(0));

    // update outer offset value
    for (int64_t m = 0; m < M; m++) {
      outer_offsets[m] += result_data[m * N + k_end - 1];
    }
  }
}

template <typename scalar_t, typename func_t>
static inline void cpu_cum_base_kernel(Tensor& result,
    const Tensor& self,
    int64_t dim,
    const func_t& f,
    scalar_t init_val) {
  if (result.sizes() != self.sizes()) {
    result.resize_as_(self);
  }
  if (self.numel() == 0) {
    return;
  }
  const auto input_ndim = self.dim();
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
    .add_output(result)
    .add_input(self)
    .build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for (int64_t i = 0; i < n; ++i) {
      f(
        (scalar_t*)result_data_bytes, result_dim_stride,
        (scalar_t*)self_data_bytes, self_dim_stride, init_val
      );
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  iter.for_each(loop);
}

static void cumsum_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  auto dtype = self.scalar_type();
  bool is_contig = self.is_contiguous() && result.is_contiguous();
  bool is_dtype_enabled = dtype == ScalarType::Double || dtype == ScalarType::Float || dtype == ScalarType::Long;
  if ((wrap_dim == self.dim() - 1) && is_contig && is_dtype_enabled) {
    AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, self.scalar_type(), "cumsum_lastdim_cpu", [&] {
      cpu_cumsum_lastdim_kernel<scalar_t>(result, self, wrap_dim);
    });
    return;
  }

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "cumsum_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          cum_number += self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

static void cumprod_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(self.scalar_type(), "cumprod_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          cum_number *= self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 1
    );
  });
}

static void logcumsumexp_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        scalar_t cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (int64_t i = 0; i < self_dim_size; ++i) {
          scalar_t x = self_data[i * self_dim_stride];

          // Reference : https://www.tensorflow.org/api_docs/python/tf/math/cumulative_logsumexp
          auto log_add_exp = [](scalar_t x, scalar_t y) -> scalar_t {
            return std::log1p(std::exp(std::min(x, y) - std::max(x, y))) + std::max(x, y);
          };
          cum_number = log_add_exp(x, cum_number);
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

// TODO: Implement `nansum` similar to the stable `sum`
// implementation in cpu/SumKernel.cpp
static void nansum_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Half){
    binary_kernel_reduce(iter, NanSumOps<float, c10::Half>{}, float{0});
  } else {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "nansum_cpu", [&] {
    binary_kernel_reduce(iter, NanSumOps<scalar_t, scalar_t>{}, scalar_t{0});
  });
  }
}

static void mean_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "mean_cpu", [&] {
    scalar_t factor = scalar_t(iter.num_output_elements()) / scalar_t(iter.numel());
    binary_kernel_reduce(
      iter,
      MeanOps<scalar_t, scalar_t> {factor},
      scalar_t(0)
    );
  });
}

static void std_var_kernel_impl(TensorIterator &iter, bool unbiased, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
      iter,
      WelfordOps<scalar_t, double, int64_t, double, std::tuple<scalar_t, scalar_t>> { unbiased, take_sqrt },
      WelfordData<double, int64_t, double>()
    );
  });
}

static void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead [-Werror=int-in-bool-context]
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    binary_kernel_reduce_vec(
      iter,
      [=](scalar_t a, scalar_t b) -> scalar_t { return a && b; },
      [=](Vec256<scalar_t> a, Vec256<scalar_t> b) { return a && b; },
      /*identity=*/1);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "prod_cpu", [&] {
      binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
        [=](Vec256 <scalar_t> a, Vec256 <scalar_t> b) { return a * b; },
        /*identity=*/1);
      });
  }
}

static void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    Scalar p) {
  float val;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    AT_ERROR("norm_kernel_tensor_iterator_impl expects norm to be integer or float");
  }

  // In the dispatch code blocks below, reduction kernels accumulate results as
  // the type `acc_t`. When `scalar_t` is complex, `acc_t` is the downgraded
  // real number type. Otherwise, `acc_t` and `scalar_t` are the same type.
  if (val == 0) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormZeroOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormOneOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == 2) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormTwoOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        AbsMaxOps<scalar_t, acc_t>(),
        acc_t(0)
      );
    });
  } else if (val == -INFINITY) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        AbsMinOps<scalar_t, acc_t>(),
        std::numeric_limits<acc_t>::max()
      );
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kHalf, kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
      using acc_t = typename scalar_value_type<scalar_t>::type;
      binary_kernel_reduce(
        iter,
        NormOps<scalar_t, acc_t> { acc_t(val) },
        acc_t(0)
      );
    });
  }

  // For complex outputs, the above kernels do not touch the imaginary values,
  // so we must zero them out
  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

static void and_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
        [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
          Vec256<uint8_t> c = Vec256<uint8_t>();

          for (decltype(c.size()) i = 0; i != Vec256<uint8_t>::size(); i++) {
            c[i] = (a[i] && b[i]) ? 1 : 0;
          }
          return c;
        },
        /*ident=*/true);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a && b; },
        [=](Vec256<bool> a, Vec256<bool> b) {
          // Adding the implementation here instead of in vec256_base to avoid
          // return value inconsistency. Other comparison operators in
          // vec256_base return -1/0 (all bit 1 / all bit 0) as true/false to
          // follow the AVX2 convention. This would be convenient when combined
          // with other vectorized operations. For example, one can use the
          // logical operation results as a mask for a bit operation to
          // retrieve/reset multiple elements in a vector.
          //
          // In this method, users would expect, e.g., all(), to return 1/0 as
          // true/false.
          Vec256<bool> c = Vec256<bool>();

          for (decltype(c.size()) i = 0; i != Vec256<bool>::size(); i++) {
            c[i] = a[i] && b[i];
          }
          return c;
        },
        /*ident=*/true);
  }
}

static void or_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a || b) ? 1 : 0; },
        [=](Vec256<uint8_t> a, Vec256<uint8_t> b) {
          Vec256<uint8_t> c = Vec256<uint8_t>();

          for (decltype(c.size()) i = 0; i != Vec256<uint8_t>::size(); i++) {
            c[i] = (a[i] || b[i]) ? 1 : 0;
          }
          return c;
        },
        /*ident=*/false);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a || b; },
        [=](Vec256<bool> a, Vec256<bool> b) {
          Vec256<bool> c = Vec256<bool>();

          for (decltype(c.size()) i = 0; i != Vec256<bool>::size(); i++) {
            c[i] = a[i] || b[i];
          }
          return c;
        },
        /*ident=*/false);
  }
}

template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

static void min_values_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == kLong) {
    // This case is special because of Vec256<int64_t> does not
    // handle upper_bound<int64_t>().
    // See: https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}

static void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
      [](Vec256<scalar_t> a, Vec256<scalar_t> b) { return maximum(a, b); },
      lower_bound<scalar_t>());
  });
}

static void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(1), "argmax_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), 0));
  });
}

static void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND(kHalf, iter.dtype(1), "argmin_cpu", [&] {
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), 0));
  });
}

}  // anonymous namespace

REGISTER_DISPATCH(nansum_stub, &nansum_kernel_impl);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl);
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl);
REGISTER_DISPATCH(mean_stub, &mean_kernel_impl);
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl);
REGISTER_DISPATCH(and_stub, &and_kernel_impl);
REGISTER_DISPATCH(or_stub, &or_kernel_impl);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl);
REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel);
REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel);
REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel);

}}  // namespace at::native
