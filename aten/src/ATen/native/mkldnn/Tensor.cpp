#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_to_dense(const MKLDNNTensor& self) {
  AT_ERROR("mkldnn_to_dense: ATen not compiled with MKLDNN support");
}

MKLDNNTensor dense_to_mkldnn(const Tensor& self) {
  AT_ERROR("dense_to_mkldnn: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/OpaqueTensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/mkldnn/Types.h>
#include <ATen/mkldnn/TensorUtils.h>
// needs to be included only once in library.
#include <ideep_pin_singletons.hpp>

namespace at { namespace native {

Tensor mkldnn_to_dense(const MKLDNNTensor& self) {
  AT_ASSERTM(self.type_id() == MkldnnCPUTensorId(), "mkldnn_to_dense: expects MKLDNN tensor");

  auto it = itensor_from_mkldnn(self);
  auto dim = it.get_dims();
  std::vector<int64_t> size(dim.begin(), dim.end());
  Tensor output = at::empty(size, self.options().layout(kStrided));
  it.reorder_to(output.data_ptr());

  return output;
}

MKLDNNTensor dense_to_mkldnn(const Tensor& self) {
  AT_ASSERTM(self.type_id() == CPUTensorId(), "dense_to_mkldnn: expects dense CPU tensor");
  AT_ASSERTM(self.scalar_type() == kFloat, "dense_to_mkldnn: expects float tensor");

  auto input = self.contiguous();
  auto dtype = get_mkldnn_dtype(input);
  MKLDNNTensor output = new_with_sizes_mkldnn(input.sizes(), input.options());
  auto it = itensor_from_mkldnn(output);
  it.reorder_from(it.get_dims(), dtype, input.data_ptr());

  return output;
}

}} // namespace at::native

#endif
