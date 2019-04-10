#pragma once

#include <ATen/ATen.h>
#include <ATen/mkldnn/Types.h>
#include <ideep.hpp>

using namespace ideep;

namespace at { namespace native {

using desc = ideep::tensor::descriptor;
using itensor = ideep::tensor;

/**
 * `IntrusivePtrTargetWrapper` wraps a custom storage handle  of a tensor
 * (as template param) and inherits `c10::intrusive_ptr_target` so that it
 * can be used with `c10::intrusive_ptr`.
 *
 * It currently only supports wrapping the custom handle by:
 * - Constructing with an existing custom handle by copy/move constructor.
 *
 * See `OpaqueTensorImpl::opaque_handle_`.
 *
 * NOTE: if this is generally useful we may want to move this to its own header.
 */
template <typename T>
struct CAFFE2_API IntrusivePtrTargetWrapper : c10::intrusive_ptr_target {
private:
  T target_;

public:
  IntrusivePtrTargetWrapper() = delete;
  IntrusivePtrTargetWrapper(const T& target): target_(target) {}
  IntrusivePtrTargetWrapper(T&& target): target_(std::move(target)) {}

  T& get_target() {
    return target_;
  }
};

using IDeepTensorWrapper = IntrusivePtrTargetWrapper<itensor>;
using IDeepTensorWrapperPtr = c10::intrusive_ptr<IDeepTensorWrapper>;
using MKLDNNTensorImpl = OpaqueTensorImpl<IDeepTensorWrapperPtr>;
using MKLDNNTensor = Tensor;

// Custom allocator using c10 CPU allocator for `ideep::tensor`
struct AllocForMKLDNN {
  template<class computation_t = void>
  static char* malloc(size_t size) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    return (char*)allocator->raw_allocate(size);
  }

  template<class computation_t = void>
  static void free(void* p) {
    auto allocator = c10::GetAllocator(c10::DeviceType::CPU);
    allocator->raw_deallocate(p);
  }
};

// create MKLDNN Tensor from ideep::tensor
inline Tensor new_with_itensor_mkldnn(itensor&& it, const TensorOptions& options) {
  AT_ASSERT(!it.has_extra());
  auto dim = it.get_dims();
  std::vector<int64_t> size(dim.begin(), dim.end());
  IDeepTensorWrapperPtr handle = c10::make_intrusive<IDeepTensorWrapper>(std::move(it));
  return detail::make_tensor<MKLDNNTensorImpl>(
      MkldnnCPUTensorId(), options.dtype(), options.device(), handle, size);
}

// create MKLDNN Tensor from sizes
inline Tensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options) {
  auto dtype = get_mkldnn_dtype(options.dtype());
  tensor::dims dim(sizes.begin(), sizes.end());
  itensor it;
  it.resize<AllocForMKLDNN>(dim, dtype);
  return new_with_itensor_mkldnn(std::move(it), options);
}

// get ideep::tensor from MKLDNN Tensor
inline itensor& itensor_from_mkldnn(const MKLDNNTensor& self) {
  AT_ASSERTM(self.type_id() == MkldnnCPUTensorId(), "itensor_from_mkldnn: expects MKLDNN tensor");
  AT_ASSERTM(!self.is_variable(), "itensor_from_mkldnn: should not be a variable");
  MKLDNNTensorImpl *mklimpl = static_cast<MKLDNNTensorImpl *>(self.unsafeGetTensorImpl());
  return mklimpl->unsafe_opaque_handle()->get_target();
}

}} // namespace at::native
