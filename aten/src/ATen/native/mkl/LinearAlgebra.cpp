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
  const float** A, const int lda, const float** B, const int ldb, const float beta,
  float** C, const int ldc) {
  cblas_sgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

static inline void gemm_batched(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
  const int batch_size, const int M, const int N, const int K, const double alpha,
  const double** A, const int lda, const double** B, const int ldb, const double beta,
  double** C, const int ldc) {
  cblas_dgemm_batch(CblasRowMajor, &trans_A, &trans_B, &M, &N, &K, &alpha,
    A, &lda, B, &ldb, &beta, C, &ldc, 1, &batch_size);
}

template <typename scalar_t>
static inline void _bmm_mkl(Tensor& res, const Tensor& mat1, const Tensor& mat2) {
  auto is_transposed = [&](const Tensor& t) {
    return t.stride(0) == 1 && t.stride(1) >= t.size(0);
  };
  const CBLAS_TRANSPOSE trans_A = is_transposed(mat1[0]) ? CblasTrans : CblasNoTrans;
  const CBLAS_TRANSPOSE trans_B = is_transposed(mat2[0]) ? CblasTrans : CblasNoTrans;

  const int batch_size = mat1.size(0);
  const int M = mat1.size(1);
  const int N = mat2.size(2);
  const int K = mat1.size(2);
  const int lda = (trans_A == CblasNoTrans) ? mat1[0].stride(0) : mat1[0].stride(1);
  const int ldb = (trans_B == CblasNoTrans) ? mat2[0].stride(0) : mat2[0].stride(1);
  const int ldc = N;

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

  gemm_batched(trans_A, trans_B, batch_size, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
}

// MKL BMM
Tensor bmm_mkl(const Tensor& self, const Tensor& tensor) {
  AT_CHECK(self.dim() == 3, "expected 3D tensor, got ", self.dim(), "D");
  AT_CHECK(tensor.dim() == 3, "expected 3D tensor, got ", tensor.dim(), "D");
  AT_CHECK(self.size(0) == tensor.size(0),
          "equal number of batches expected, got ", self.size(0), ", ", tensor.size(0));
  AT_CHECK(self.size(2) == tensor.size(1),
          "wrong matrix size, batch1: ", self.size(1), "x", self.size(2),
          ", batch2 ", tensor.size(1), "x", tensor.size(2));

  std::vector<int64_t> output_sz{self.size(0), self.size(1), tensor.size(2)};
  auto result = self.type().tensor(output_sz);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "bmm_mkl", [&] {
    _bmm_mkl<scalar_t>(result, self, tensor);
  });

  return result;
}

}} // namespace at::native

#endif
