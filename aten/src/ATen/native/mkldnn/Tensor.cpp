#include <ATen/ATen.h>
#include <ATen/Layout.h>
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

#include <ATen/mkldnn/Runtime.h>
#include <ATen/mkldnn/TensorUtils.h>

namespace at { namespace native {

namespace {

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

// Helper function to construct a `Storage` given allocated `ideep::tensor`.
// The storage does not own the buffer. The assumption is that there would be
// no reallocation from `ideep::tensor` later.
c10::Storage new_with_itensor_storage(const itensor& it, const TensorOptions& options) {
  c10::DataPtr data_ptr(it.get_data_handle(), c10::DeviceType::CPU);
  return c10::Storage(
      options.dtype(), it.get_size() / options.dtype().itemsize(),
      std::move(data_ptr), /*allocator=*/nullptr, /*resizeable=*/false);
}

MKLDNNTensor new_with_itensor_mkldnn(itensor&& it, const TensorOptions& options) {
  auto dims = it.get_dims();
  auto size = std::vector<int64_t>(dims.begin(), dims.end());
  c10::Storage storage(new_with_itensor_storage(it, options));
  return detail::make_tensor<TensorImpl>(
      std::move(storage), MkldnnCPUTensorId(), false,
      c10::make_intrusive<c10::OpaqueHandle<itensor>>(std::move(it)),
      size);
}

MKLDNNTensor new_with_sizes_mkldnn(IntArrayRef sizes, const TensorOptions& options) {
  auto dtype = get_mkldnn_dtype(options.dtype());
  tensor::dims dst_dims(sizes.begin(), sizes.end());
  itensor it;
  it.resize<AllocForMKLDNN>(dst_dims, dtype);
  return new_with_itensor_mkldnn(std::move(it), options);
}

} // anonymous namespace

Tensor mkldnn_to_dense(const MKLDNNTensor& self) {
  AT_ASSERTM(self.type_id() == MkldnnCPUTensorId(), "mkldnn_to_dense: expects MKLDNN tensor");

  auto stensor = get_mkldnn_itensor(self);
  auto dims = stensor.get_dims();
  auto size = std::vector<int64_t>(dims.begin(), dims.end());
  Tensor dst = at::empty(size, self.options().layout(kStrided));

  stensor.reorder_to(dst.data_ptr());

  return dst;
}

MKLDNNTensor dense_to_mkldnn(const Tensor& self) {
  AT_ASSERTM(self.type_id() == CPUTensorId(), "dense_to_mkldnn: expects dense CPU tensor");
  AT_ASSERTM(self.scalar_type() == kFloat, "dense_to_mkldnn: expects float tensor");

  auto input = self.contiguous();
  auto dtype = get_mkldnn_dtype(input);
  Tensor dst = new_with_sizes_mkldnn(input.sizes(), input.options());
  auto dtensor = get_mkldnn_itensor(dst);

  dtensor.reorder_from(dtensor.get_dims(), dtype, input.data_ptr());

  return dst;
}

}} // namespace at::native

#endif
