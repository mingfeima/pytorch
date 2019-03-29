#pragma once

#include <ATen/mkldnn/Runtime.h>
#include <c10/core/OpaqueHandle.h>

namespace at { namespace native {

// Just for documentary purposes
using MKLDNNTensor = Tensor;

inline itensor& get_mkldnn_itensor(const MKLDNNTensor& self) {
  AT_ASSERTM(self.type_id() == MkldnnCPUTensorId(), "get_mkldnn_itensor: expects MKLDNN tensor");
  auto it_handle = (OpaqueHandle<itensor>*)self.unsafeGetTensorImpl()->unsafe_opaque_handle();
  return it_handle->get_handle();
}

}} // namespace at::native
