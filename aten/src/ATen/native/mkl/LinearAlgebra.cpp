#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Config.h"

#if !AT_MKL_ENABLED()

namespace at { namespace native {

Tensor bmm_mkl(const Tensor& self, const Tensor& tensor) {
  throw std::runtime_error("bmm: ATen not compiled with MKL support");
}

}}

#else // AT_MKL_ENABLED

#include "ATen/ATen.h"
#include "ATen/Config.h"
#include "ATen/Dispatch.h"
#include "ATen/Utils.h"
#include "ATen/NativeFunctions.h"

#include <algorithm>
#include <vector>
#include <numeric>
#include <cmath>

#include <mkl.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Limits.h>

namespace at { namespace native {

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const float alpha,
  const float** A, const float** B, const float beta, float** C) {
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;

  cblas_sgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const double alpha,
  const double** A, const double** B, const double beta, double** C) {
  const int lda = (trans_A == CblasNoTrans) ? K : M;
  const int ldb = (trans_B == CblasNoTrans) ? N : K;
  const int ldc = N;

  cblas_dgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

template <typename scalar_t>
static inline void _bmm_mkl(Tensor& res, const Tensor& mat1, const Tensor& mat2) {
  const int batch_size = mat1.size(0);
  const int M = mat1.size(1);
  const int N = mat2.size(2);
  const int K = mat1.size(2);
  scalar_t alpha = static_cast<scalar_t>(1.0);
  scalar_t beta = static_cast<scalar_t>(0.0);

  std::vector<const scalar_t*> A(batch_size);
  std::vector<const scalar_t*> B(batch_size);
  std::vector<scalar_t*> C(batch_size);
  for (int64_t batch = 0; batch < batch_size; batch++) {
    A[batch] = mat1[batch].data<scalar_t>();
    B[batch] = mat2[batch].data<scalar_t>();
    C[batch] = res[batch].data<scalar_t>();
  }

  gemm_batched(CblasNoTrans, CblasNoTrans, batch_size, M, N, K, alpha, A.data(), B.data(), beta, C.data());
}

// MKL BMM
Tensor bmm_mkl(const Tensor& self, const Tensor& tensor) {
  if (self.dim() != 3 || tensor.dim() != 3) {
    std::ostringstream ss;
    ss << "expected 3D tensors, got " << self.dim() << "D and " << tensor.dim() << "D";
    throw std::runtime_error(ss.str());
  }
  if (self.size(0) != tensor.size(0)) {
    std::ostringstream ss;
    ss << "equal number of batches expected, got " << self.size(0) << ", " << tensor.size(0);
    throw std::runtime_error(ss.str());
  }
  if (self.size(2) != tensor.size(1)) {
    std::ostringstream ss;
    ss << "wrong matrix size, batch1: " << self.size(1) << "x" << self.size(2)
       << ", batch2 " << tensor.size(1) << "x" << tensor.size(2);
    throw std::runtime_error(ss.str());
  }

  std::vector<int64_t> output_sz{self.size(0), self.size(1), tensor.size(2)};
  auto result = self.type().tensor(output_sz);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "bmm_mkl", [&] {
    _bmm_mkl<scalar_t>(result, self, tensor);
  });

  return result;
}

}} // namespace at::native

#endif
