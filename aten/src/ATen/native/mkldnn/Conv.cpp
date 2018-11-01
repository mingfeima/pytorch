#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

Tensor mkldnn_convolution(const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups) {
  AT_ERROR("mkldnn_convolution_forward: ATen not compiled with MKLDNN support");
}

Tensor mkldnn_convolution_backward_input(IntList input_size, const Tensor& grad_output,
    const Tensor& weight, IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_input: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(IntList weight_size,
    const Tensor& grad_output, const Tensor& input, IntList padding, IntList stride,
    IntList dilation, int64_t groups, bool bias_defined) {
  AT_ERROR("mkldnn_convolution_backward_weights: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(const Tensor& input,
    const Tensor& grad_output_t, const Tensor& weight, IntList padding, IntList stride,
    IntList dilation, int64_t groups, std::array<bool,3> output_mask) {
  AT_ERROR("mkldnn_convolution_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>
#include <ATen/mkldnn/Memory.h>
#include <ATen/mkldnn/Utils.h>

using namespace mkldnn;

namespace at { namespace native {

namespace {

constexpr int input_batch_size_dim = 0;  // also grad_input
constexpr int input_channels_dim = 1;
constexpr int output_batch_size_dim = 0;  // also grad_output
constexpr int output_channels_dim = 1;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

std::vector<int64_t> conv_output_size(
    IntList input_size, IntList weight_size,
    IntList padding, IntList stride, IntList dilation, int64_t groups)
{
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[0] = input_size[input_batch_size_dim];
  output_size[1] = weight_size[weight_output_channels_dim];
  for (size_t d = 2; d < dim; ++d) {
    auto kernel = dilation[d - 2] * (weight_size[d] - 1) + 1;
    output_size[d] = (input_size[d] + (2 * padding[d - 2])
                        - kernel) / stride[d - 2] + 1;
  }
  return output_size;
}

struct ConvolutionParams {
  int64_t dim;
  int64_t input_size[2 + max_dim];
  int64_t weight_size[2 + max_dim];
  int64_t output_size[2 + max_dim];
  int64_t padding[max_dim];
  int64_t stride[max_dim];
  int64_t dilation[max_dim];
  int64_t groups;
  bool has_bias;
};

void setConvolutionParams(ConvolutionParams* params, const Tensor& input,
    const Tensor& weight, const Tensor& output, IntList padding, IntList stride,
    IntList dilation, int64_t groups, bool has_bias) {
  memset(params, 0, sizeof(ConvolutionParams));

  params->dim = input.dim();
  for (int64_t i = 0; i < params->dim; ++i) {
    params->input_size[i] = input.size(i);
    params->weight_size[i] = weight.size(i);
    params->output_size[i] = output.size(i);
  }
  for (size_t i = 0; i < padding.size(); ++i) {
    params->padding[i] = padding[i];
    params->stride[i] = stride[i];
    params->dilation[i] = dilation[i];
  }
  params->groups = groups;
  params->has_bias = has_bias;
}

struct ConvolutionArgs {
  ConvolutionParams params;
  memory::dims input_tz;
  memory::dims weight_tz;
  memory::dims bias_tz;
  memory::dims output_tz;
  memory::dims _stride;
  memory::dims _dilation;
  memory::dims _padding;
  memory::dims _padding_r;
  memory::format format_data;
  memory::format format_weight;
  bool dilated_conv;

  ConvolutionArgs(const Tensor& input, const Tensor& weight, const Tensor& output,
      IntList padding, IntList stride, IntList dilation, int64_t groups, bool has_bias) {
    // set ConvolutionParams which is POD style, used for caching mkldnn primitives
    setConvolutionParams(&params, input, weight, output,
      padding, stride, dilation, groups, has_bias);

    if (groups != 1) weight_tz.push_back(groups);
    for (int64_t i = 0; i < input.dim(); ++i) {
      input_tz.push_back(params.input_size[i]);
      weight_tz.push_back(params.weight_size[i]);
      output_tz.push_back(params.output_size[i]);
    }
    if (groups != 1) weight_tz[weight_output_channels_dim + 1] /= groups;
    bias_tz.push_back(output.size(output_channels_dim));

    dilated_conv = false;
    for (size_t k = 0; k < padding.size(); ++k) {
      if (dilation[k] != 1) dilated_conv = true;
      _stride.push_back(stride[k]);
      _dilation.push_back(dilation[k] - 1);
      _padding.push_back(padding[k]);
      _padding_r.push_back((output.size(k + 2) - 1) * stride[k] - input.size(k + 2) +
         ((weight.size(k + 2) - 1) * dilation[k] + 1) - padding[k]);
    }

    if (input.dim() == 4) {
      format_data = memory::format::nchw;
      format_weight = (groups!= 1) ? memory::format::goihw : memory::format::oihw;
    } else {
      format_data = memory::format::ncdhw;
      format_weight = (groups!= 1) ? memory::format::goidhw : memory::format::oidhw;
    }
  }

  memory::primitive_desc input_pd() { return _primitive_md(input_tz, format_data); }
  memory::primitive_desc weight_pd() { return _primitive_md(weight_tz, format_weight); }
  memory::primitive_desc bias_pd() { return _primitive_md(bias_tz, memory::format::x); }
  memory::primitive_desc output_pd() { return _primitive_md(output_tz, format_data);}
};

convolution_forward::primitive_desc _conv_fwd_pd(const ConvolutionArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto conv_prop = prop_kind::forward; //TODO: fn_train flag?
  auto conv_algo = algorithm::convolution_direct;
  auto input_md = _generic_md(args.input_tz);
  auto weight_md = _generic_md(args.weight_tz);
  auto bias_md = _generic_md(args.bias_tz);
  auto output_md = _generic_md(args.output_tz);

  std::shared_ptr<convolution_forward::desc> _desc;
  if (args.params.has_bias) {
    if (args.dilated_conv) {
      _desc.reset(new convolution_forward::desc(conv_prop, conv_algo, input_md, weight_md,
        bias_md, output_md, args._stride, args._dilation, args._padding, args._padding_r,
        padding_kind::zero));
    } else {
      _desc.reset(new convolution_forward::desc(conv_prop, conv_algo, input_md, weight_md,
        bias_md, output_md, args._stride, args._padding, args._padding, padding_kind::zero));
    }
  } else {
    if (args.dilated_conv) {
      _desc.reset(new convolution_forward::desc(conv_prop, conv_algo, input_md, weight_md,
        output_md, args._stride, args._dilation, args._padding, args._padding_r,
        padding_kind::zero));
    } else {
      _desc.reset(new convolution_forward::desc(conv_prop, conv_algo, input_md, weight_md,
        output_md, args._stride, args._padding, args._padding, padding_kind::zero));
    }
  }

  return convolution_forward::primitive_desc(*_desc, _engine);
}

convolution_backward_data::primitive_desc _conv_bwd_data_pd(const ConvolutionArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto conv_algo = algorithm::convolution_direct;
  auto input_md = _generic_md(args.input_tz);
  auto weight_md = _generic_md(args.weight_tz);
  auto bias_md = _generic_md(args.bias_tz);
  auto output_md = _generic_md(args.output_tz);

  std::shared_ptr<convolution_backward_data::desc> _desc;
  if (args.dilated_conv) {
    _desc.reset(new convolution_backward_data::desc(conv_algo, input_md, weight_md, output_md,
      args._stride, args._dilation, args._padding, args._padding_r, padding_kind::zero));
  } else {
    _desc.reset(new convolution_backward_data::desc(conv_algo, input_md, weight_md, output_md,
      args._stride, args._padding, args._padding, padding_kind::zero));
  }

  return convolution_backward_data::primitive_desc(*_desc, _engine, _conv_fwd_pd(args));
}

convolution_backward_weights::primitive_desc _conv_bwd_weight_pd(const ConvolutionArgs& args) {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  auto conv_algo = algorithm::convolution_direct;
  auto input_md = _generic_md(args.input_tz);
  auto weight_md = _generic_md(args.weight_tz);
  auto bias_md = _generic_md(args.bias_tz);
  auto output_md = _generic_md(args.output_tz);

  std::shared_ptr<convolution_backward_weights::desc> _desc;
  if (args.params.has_bias) {
    if (args.dilated_conv) {
      _desc.reset(new convolution_backward_weights::desc(conv_algo, input_md, weight_md,
        bias_md, output_md, args._stride, args._dilation, args._padding, args._padding_r,
        padding_kind::zero));
    } else {
      _desc.reset(new convolution_backward_weights::desc(conv_algo, input_md, weight_md,
        bias_md, output_md, args._stride, args._padding, args._padding, padding_kind::zero));
    }
  } else {
    if (args.dilated_conv) {
      _desc.reset(new convolution_backward_weights::desc(conv_algo, input_md, weight_md,
        output_md, args._stride, args._dilation, args._padding, args._padding_r,
        padding_kind::zero));
    } else {
      _desc.reset(new convolution_backward_weights::desc(conv_algo, input_md, weight_md,
        output_md, args._stride, args._padding, args._padding, padding_kind::zero));
    }
  }

  return  convolution_backward_weights::primitive_desc(*_desc, _engine, _conv_fwd_pd(args));
}

struct MKLDNNConvForward : MKLDNNPrimitive<convolution_forward> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _weight;
  std::shared_ptr<memory> _bias;
  std::shared_ptr<memory> _output;

  MKLDNNConvForward() : MKLDNNPrimitive<convolution_forward>() {
    set_null_memory(_input);
    set_null_memory(_weight);
    set_null_memory(_bias);
    set_null_memory(_output);
  }

  void set(const convolution_forward::primitive_desc pd, const memory& input,
      const memory& weight, const std::shared_ptr<memory>& bias, const memory& output) {
    _input->set_data_handle(input.get_data_handle());
    _weight->set_data_handle(weight.get_data_handle());
    _output->set_data_handle(output.get_data_handle());

    if (bias != nullptr) {
      _bias->set_data_handle(bias->get_data_handle());
      if (_prim == nullptr) {
        _prim.reset(new convolution_forward(pd, primitive::at(*_input),
          primitive::at(*_weight), primitive::at(*_bias), *_output));
      }
    } else {
      if (_prim == nullptr) {
        _prim.reset(new convolution_forward(pd, primitive::at(*_input),
          primitive::at(*_weight), *_output));
      }
    }
  }
};

struct MKLDNNConvBackwardData : MKLDNNPrimitive<convolution_backward_data> {
  std::shared_ptr<memory> _grad_output;
  std::shared_ptr<memory> _weight;
  std::shared_ptr<memory> _grad_input;

  MKLDNNConvBackwardData() : MKLDNNPrimitive<convolution_backward_data>() {
    set_null_memory(_grad_output);
    set_null_memory(_weight);
    set_null_memory(_grad_input);
  }

  void set(const convolution_backward_data::primitive_desc& pd,
      const memory& grad_output, const memory& weight, const memory& grad_input) {
    _grad_output->set_data_handle(grad_output.get_data_handle());
    _weight->set_data_handle(weight.get_data_handle());
    _grad_input->set_data_handle(grad_input.get_data_handle());

    if (_prim == nullptr) {
      _prim.reset(new convolution_backward_data(pd, primitive::at(*_grad_output),
        primitive::at(*_weight), *_grad_input));
    }
  }
};

struct MKLDNNConvBackwardWeight : MKLDNNPrimitive<convolution_backward_weights> {
  std::shared_ptr<memory> _input;
  std::shared_ptr<memory> _grad_output;
  std::shared_ptr<memory> _grad_weight;
  std::shared_ptr<memory> _grad_bias;

  MKLDNNConvBackwardWeight() : MKLDNNPrimitive<convolution_backward_weights>() {
    set_null_memory(_input);
    set_null_memory(_grad_output);
    set_null_memory(_grad_weight);
    set_null_memory(_grad_bias);
  }

  void set(const convolution_backward_weights::primitive_desc& pd, const memory& input,
      const memory& grad_output, const memory& grad_weight, const std::shared_ptr<memory>& grad_bias) {
    _input->set_data_handle(input.get_data_handle());
    _grad_output->set_data_handle(grad_output.get_data_handle());
    _grad_weight->set_data_handle(grad_weight.get_data_handle());

    if (grad_bias != nullptr) {
      _grad_bias->set_data_handle(grad_bias->get_data_handle());
      if (_prim == nullptr) {
        _prim.reset(new convolution_backward_weights(pd, primitive::at(*_input),
          primitive::at(*_grad_output), *_grad_weight, *_grad_bias));
      }
    } else {
      if (_prim == nullptr) {
        _prim.reset(new convolution_backward_weights(pd, primitive::at(*_input),
          primitive::at(*_grad_output), *_grad_weight));
      }
    }
  }
};

}  // namespace

Tensor mkldnn_convolution(const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntList padding, IntList stride, IntList dilation, int64_t groups) {
  auto output = at::empty(conv_output_size(
    input.sizes(), weight.sizes(), padding, stride, dilation, groups), input.options());

  ConvolutionArgs args(input, weight, output, padding, stride, dilation, groups, bias.defined());
  auto _pd = _conv_fwd_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto weight_usr = MKLDNNMemory(args.weight_pd(), weight);
  auto output_usr = MKLDNNMemory(args.output_pd(), output);

  auto input_prv = input_usr.reorder_to(_pd.src_primitive_desc());
  auto weight_prv = weight_usr.reorder_to(_pd.weights_primitive_desc());
  auto output_prv = output_usr.create(_pd.dst_primitive_desc());

  std::shared_ptr<memory> bias_prv;
  if (bias.defined()) {
    bias_prv.reset(new memory(args.bias_pd(), bias.data_ptr()));
  }

  std::shared_ptr<MKLDNNConvForward> conv_fwd;
  static thread_local PrimitiveCache<ConvolutionParams, MKLDNNConvForward> cache;
  if (cache.find(args.params, conv_fwd)) {
    conv_fwd->set(_pd, input_prv, weight_prv, bias_prv, output_prv);
  } else {
    conv_fwd.reset(new MKLDNNConvForward());
    conv_fwd->set(_pd, input_prv, weight_prv, bias_prv, output_prv);
    cache.insert(args.params, conv_fwd);
  }
  MKLDNN_EXEC(conv_fwd->get_primitive());
  output_usr.reorder_from(output_prv);

  return output;
}

Tensor mkldnn_convolution_backward_input(IntList input_size, const Tensor& grad_output,
    const Tensor& weight, IntList padding, IntList stride, IntList dilation,
    int64_t groups, bool bias_defined) {
  auto grad_input = at::empty(input_size, grad_output.options());

  ConvolutionArgs args(grad_input, weight, grad_output, padding, stride, dilation, groups, bias_defined);
  auto _pd = _conv_bwd_data_pd(args);

  auto grad_output_usr = MKLDNNMemory(args.output_pd(), grad_output);
  auto weight_usr = MKLDNNMemory(args.weight_pd(), weight);
  auto grad_input_usr = MKLDNNMemory(args.input_pd(), grad_input);

  auto grad_output_prv = grad_output_usr.reorder_to(_pd.diff_dst_primitive_desc());
  auto weight_prv = weight_usr.reorder_to(_pd.weights_primitive_desc());
  auto grad_input_prv = grad_input_usr.create(_pd.diff_src_primitive_desc());

  std::shared_ptr<MKLDNNConvBackwardData> conv_bwd_data;
  static thread_local PrimitiveCache<ConvolutionParams, MKLDNNConvBackwardData> cache;
  if (cache.find(args.params, conv_bwd_data)) {
    conv_bwd_data->set(_pd, grad_output_prv, weight_prv, grad_input_prv);
  } else {
    conv_bwd_data.reset(new MKLDNNConvBackwardData());
    conv_bwd_data->set(_pd, grad_output_prv, weight_prv, grad_input_prv);
    cache.insert(args.params, conv_bwd_data);
  }
  MKLDNN_EXEC(conv_bwd_data->get_primitive());
  grad_input_usr.reorder_from(grad_input_prv);

  return grad_input;
}

std::tuple<Tensor, Tensor> mkldnn_convolution_backward_weights(IntList weight_size,
    const Tensor& grad_output, const Tensor& input, IntList padding, IntList stride,
    IntList dilation, int64_t groups, bool bias_defined) {
  auto grad_weight = at::empty(weight_size, grad_output.options());
  Tensor grad_bias;
  if (bias_defined) {
    grad_bias = at::empty({grad_output.size(1)}, grad_output.options());
  }

  ConvolutionArgs args(input, grad_weight, grad_output, padding, stride, dilation, groups, bias_defined);
  auto _pd = _conv_bwd_weight_pd(args);

  auto input_usr = MKLDNNMemory(args.input_pd(), input);
  auto grad_output_usr = MKLDNNMemory(args.output_pd(), grad_output);
  auto grad_weight_usr = MKLDNNMemory(args.weight_pd(), grad_weight);

  auto input_prv = input_usr.reorder_to(_pd.src_primitive_desc());
  auto grad_output_prv = grad_output_usr.reorder_to(_pd.diff_dst_primitive_desc());
  auto grad_weight_prv = grad_weight_usr.create(_pd.diff_weights_primitive_desc());

  std::shared_ptr<memory> grad_bias_prv;
  if (bias_defined) {
    grad_bias_prv.reset(new memory(args.bias_pd(), grad_bias.data_ptr()));
  }

  std::shared_ptr<MKLDNNConvBackwardWeight> conv_bwd_weight;
  static thread_local PrimitiveCache<ConvolutionParams, MKLDNNConvBackwardWeight> cache;
  if (cache.find(args.params, conv_bwd_weight)) {
    conv_bwd_weight->set(_pd, input_prv, grad_output_prv, grad_weight_prv, grad_bias_prv);
  } else {
    conv_bwd_weight.reset(new MKLDNNConvBackwardWeight());
    conv_bwd_weight->set(_pd, input_prv, grad_output_prv, grad_weight_prv, grad_bias_prv);
    cache.insert(args.params, conv_bwd_weight);
  }
  MKLDNN_EXEC(conv_bwd_weight->get_primitive());
  grad_weight_usr.reorder_from(grad_weight_prv);

  return std::tuple<Tensor, Tensor>{grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_convolution_backward(const Tensor& input,
    const Tensor& grad_output_t, const Tensor& weight, IntList padding, IntList stride,
    IntList dilation, int64_t groups, std::array<bool,3> output_mask) {
  Tensor grad_output = grad_output_t.contiguous();

  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::mkldnn_convolution_backward_input(
      input.sizes(), grad_output, weight, padding, stride, dilation, groups, output_mask[2]);
  }
  if (output_mask[1] || output_mask[2]) {
    std::tie(grad_weight, grad_bias) = at::mkldnn_convolution_backward_weights(
      weight.sizes(), grad_output, input, padding, stride, dilation, groups, output_mask[2]);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

}}  // namespace at::native

#endif
