#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(false, "mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined) {
  TORCH_CHECK(false, "mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask) {
  TORCH_CHECK(false, "mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/ConvUtils.h>

namespace at { namespace native {

namespace {

// Helper function for getting an ideep tensor out of an aten Tensor.
// Note in case the aten Tensor is a dense tensor, the returned ideep
// tensor is just a view of the storage of the aten dense tensor, so
// caller needs to make sure the aten dense tensor's lifetime is
// longer than the ideep tensor.
inline ideep::tensor get_mkldnn_tensor(const Tensor& tensor) {
  if (tensor.is_mkldnn()) {
    return itensor_from_mkldnn(tensor);
  } else {
    return itensor_view_from_dense(tensor);
  }
}

// Reorder tensor in blocked format to plain formats
// according to memory formats
Tensor reorder_mkldnn_tensor(
    const ideep::tensor& mkldnn_tensor,
    const TensorOptions& options,
    at::MemoryFormat memory_format = at::MemoryFormat::Contiguous) {
  if (memory_format == at::MemoryFormat::ChannelsLast) {
    auto ndims = mkldnn_tensor.ndims();
    TORCH_CHECK(ndims == 4 || ndims == 5, "ChannelsLast memory format supports only 4D or 5D tensor");
  }

  auto dims = mkldnn_tensor.get_dims();
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    options.layout(c10::kStrided),
    memory_format);
  if (mkldnn_tensor.is_empty()) return cpu_tensor;
  ideep::tensor cpu_tensor_view = itensor_view_from_dense(cpu_tensor);
  cpu_tensor_view.reorder_from(mkldnn_tensor);

  return cpu_tensor;
}

} // anonymous namespace

// Note [MKLDNN Convolution Memory Formats]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MKLDNN has 3 types of memory formats in convolution:
//
// In case memory format passed from PyTorch (user layout)
// differs from the internal layout MKLDNN used, a `reorder` is needed;
// otherwise when user layout is identical to internal layout,
// MKLDNN uses a memory view upon ATen tensor.
//
// 1. NCHW (default)
//  input:  NCHW(user) -> Blocked(internal)
//  weight: OIHW(user) -> Blocked(internal)
//  output: Blocked(internal) -> NCHW(user)
//
// 2. NHWC: (channels last)
//  input:  NHWC(user) -> NHWC(internal)
//  weight: OHWI(user) -> Blocked(internal)
//  output: NHWC(internal) -> NHWC(user)
//
// 3. Blocked:
//  By explicitly converting a tensor to mkldnn, e.g. `x.to_mkldnn()`,
//  blocked format will propagate between layers. Input, output and weight will
//  all be in blocked format.
//
Tensor mkldnn_convolution(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  auto output_sizes = conv_output_size(input.sizes(), weight.sizes(), padding, stride, dilation);
  auto output = at::empty({0}, input.options());
  bool is_channels_last = input.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  const ideep::tensor x = get_mkldnn_tensor(input);
  const ideep::tensor w = get_mkldnn_tensor(weight);

  ideep::tensor y;
  if (is_channels_last) {
    output.resize_(output_sizes, input.suggest_memory_format());
    y = get_mkldnn_tensor(output);
  }
  if (bias.defined()) {
    const ideep::tensor b = get_mkldnn_tensor(bias);
    ideep::convolution_forward::compute(
        x,
        w,
        b,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  } else {
    ideep::convolution_forward::compute(
        x,
        w,
        {output_sizes.cbegin(), output_sizes.cend()},
        y,
        {stride.begin(), stride.end()},
        {dilation.begin(), dilation.end()},
        {padding.begin(), padding.end()},
        {padding.begin(), padding.end()},
        groups);
  }

  if (input.is_mkldnn()) {
    return new_with_itensor_mkldnn(std::move(y), input.options());
  } else if (!is_channels_last) {
    return reorder_mkldnn_tensor(y, input.options());
  } else {
    TORCH_INTERNAL_ASSERT(y.get_desc().is_nhwc());
    return output;
  }
}

Tensor mkldnn_convolution_backward_input(
    IntArrayRef input_size, const Tensor& grad_output, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  auto grad_input = at::empty({0}, grad_output.options());
  bool is_channels_last = grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast;

  auto grad_y = get_mkldnn_tensor(grad_output);
  auto w = get_mkldnn_tensor(weight);

  ideep::tensor grad_x;
  if (is_channels_last) {
    grad_input.resize_(input_size, grad_output.suggest_memory_format());
    grad_x = get_mkldnn_tensor(grad_input);
  }
  ideep::convolution_backward_data::compute(
      grad_y,
      w,
      input_size.vec(),
      grad_x,
      stride.vec(),
      dilation.vec(),
      padding.vec(),
      padding.vec(),
      groups);

  if (!is_channels_last) {
    return reorder_mkldnn_tensor(grad_x, grad_output.options());
  } else {
    TORCH_INTERNAL_ASSERT(grad_x.get_desc().is_nhwc());
    return grad_input;
  }
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(
    IntArrayRef weight_size, const Tensor& grad_output, const Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined)
{
  const ideep::tensor grad_y = get_mkldnn_tensor(grad_output);
  const ideep::tensor x = get_mkldnn_tensor(input);

  ideep::tensor grad_w, grad_b;
  if (bias_defined) {
    ideep::convolution_backward_weights::compute(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        grad_b,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  } else {
    ideep::convolution_backward_weights::compute(
        x,
        grad_y,
        weight_size.vec(),
        grad_w,
        stride.vec(),
        dilation.vec(),
        padding.vec(),
        padding.vec(),
        groups);
  }

  // mkldnn outputs grad_weight in blocked format,
  // reorder it to plain format (OIHW or OHWI)
  auto memory_format = input.suggest_memory_format();
  auto grad_weight = reorder_mkldnn_tensor(grad_w, grad_output.options(), memory_format);
  auto grad_bias = reorder_mkldnn_tensor(grad_b, grad_output.options());

  return std::make_tuple(grad_weight, grad_bias);
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(
    const Tensor& input, const Tensor& grad_output_t, const Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask)
{
  Tensor grad_output = grad_output_t.contiguous(input.suggest_memory_format());

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}}  // namespace at::native

#endif
