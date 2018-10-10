#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_rnn_cell(
    const Tensor& input, TensorList weight, const Tensor& hx, const Tensor& cx) {
  throw std::runtime_error("_mkldnn_rnn_cell: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_cell_backward(
    const Tensor& input, TensorList weight, const Tensor& hx, const Tensor& cx,
    const Tensor& hy, const Tensor& cy, const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& workspace) {
  throw std::runtime_error("_mkldnn_rnn_cell_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_ENABLED()

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

constexpr int64_t ldigo_shuffle_dim = 1;
constexpr int64_t ldgoi_shuffle_dim = 0;
constexpr int64_t ldgo_shffule_dim = 0;

namespace {

  memory::desc _format_md(const memory::dims _dims, const memory::format _format) {
    return memory::desc({_dims}, memory::data_type::f32, _format);
  }

  memory::desc _generic_md(const memory::dims _dims) {
    return _format_md(_dims, memory::format::any);
  }

  memory _reorder(const memory& usr_memory, const memory::primitive_desc& pd) {
    if (usr_memory.get_primitive_desc() != memory::primitive_desc(pd)) {
      auto _memory = memory(pd);
      std::vector<memory::primitive> net;
      net.push_back(reorder(usr_memory, _memory));
      Stream::Instance().get_stream().submit(net);
      return _memory;
    }
    return usr_memory;
  }

  // NB: MKLDNN has several special requirements for RNN weight primitive
  // a) weight needs to be in ldgio format
  // b) for LSTM, mkldnn gate order is (forget, input, output, cell), different
  // from pytorch native (input, forget, cell, output)
  Tensor _shuffle_gates(const Tensor& weight, int64_t num_gates, bool forward=true) {
    auto is_bias = (weight.dim() == 1);
    auto weight_t = is_bias ? weight : weight.t();
    if (num_gates == 4) {
      auto _dim = is_bias ? ldgo_shffule_dim: forward ? ldigo_shuffle_dim : ldgoi_shuffle_dim;
      std::vector<Tensor> gates = weight_t.chunk(num_gates, _dim);
      return at::cat({gates[1], gates[0], gates[3], gates[2]}, _dim);
    }
    return weight_t.contiguous();
  }

} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> mkldnn_rnn_cell(
    const Tensor& input, TensorList weight, const Tensor& hx, const Tensor& cx) {
  // NB: MKLDNN requires to concat hx and cx for lstm
  Tensor hidden_in, hidden_out, hy, cy;
  if (cx.defined()) {
    hidden_in = at::cat({hx, cx}, 0);
    hidden_out = at::empty_like(hidden_in);
    std::vector<Tensor> hidden_arr = hidden_out.chunk(2, 0);
    hy = hidden_arr[0];
    cy = hidden_arr[1];
  } else {
    hidden_in = hx;
    hidden_out = at::empty_like(hx);
    hy = hidden_out;
    cy = at::empty({0}, hx.options()); // NB: Not allowed to return undefined tensors
  }

  auto output = at::empty_like(hy);
  auto workspace = at::empty({0}, input.options());

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto null_memory_ = null_memory(cpu_engine);

  int32_t batch_size = input.size(0);
  int32_t input_size = input.size(1);
  int32_t hidden_size = hx.size(1);

  int32_t time_step = 1;
  int32_t num_layers = 1;
  int32_t num_directions = 1;
  int32_t num_gates = (cx.defined()) ? 4 : 3;
  int32_t num_states = (cx.defined()) ? 2 : 1;

  auto format_tnc = memory::format::tnc;
  auto format_ldigo = memory::format::ldigo;
  auto format_ldgo = memory::format::ldgo;
  auto format_ldsnc = memory::format::ldsnc;

  auto weight_ih = _shuffle_gates(weight[0], num_gates);
  auto weight_hh = _shuffle_gates(weight[1], num_gates);
  auto bias = _shuffle_gates(weight[2] + weight[3], num_gates);

  auto train = true;

  auto rnn_prop = train ? prop_kind::forward_training : prop_kind::forward_inference;
  auto rnn_algo = (cx.defined()) ? algorithm::vanilla_lstm :gru_linear_before_reset;
  auto rnn_dir = rnn_direction::unidirectional_left2right;

  memory::dims input_tz = {time_step, batch_size, input_size};
  memory::dims weight_ih_tz = {num_layers, num_directions, input_size, num_gates, hidden_size};
  memory::dims weight_hh_tz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};
  memory::dims bias_tz = {num_layers, num_directions, num_gates, hidden_size};
  memory::dims hidden_tz = {num_layers, num_directions, num_states, batch_size, hidden_size};
  memory::dims output_tz = {time_step, batch_size, hidden_size};

  auto input_md = _format_md(input_tz, format_tnc);
  auto hidden_md = _generic_md(hidden_tz);
  auto weight_ih_md = _generic_md(weight_ih_tz);
  auto weight_hh_md = _generic_md(weight_hh_tz);
  auto bias_md = _generic_md(bias_tz);
  auto output_md = _format_md(output_tz, format_tnc);

  auto rnn_cell_ = rnn_cell::desc(rnn_algo);
try {
  auto rnn_forward_desc = rnn_forward::desc(rnn_prop, rnn_cell_, rnn_dir,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);

  auto rnn_forward_pd = rnn_forward::primitive_desc(rnn_forward_desc, cpu_engine);

  auto input_usr_mem = memory({input_md, cpu_engine}, input.data_ptr());
  auto hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_in.data_ptr());
  auto weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldigo), cpu_engine}, weight_ih.data_ptr());
  auto weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldigo), cpu_engine}, weight_hh.data_ptr());
  auto bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, bias.data_ptr());
  auto output_usr_mem = memory({output_md, cpu_engine}, output.data_ptr());
  auto hidden_out_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_out.data_ptr());

  auto workspace_mem = null_memory_;
  if (train) {
    auto workspace_pd = rnn_forward_pd.workspace_primitive_desc();
    auto workspace_size = workspace_pd.get_size();
    workspace.resize_(workspace_size);
    workspace_mem = memory(workspace_pd, workspace.data_ptr());
  }

  auto input_mem = _reorder(input_usr_mem, rnn_forward_pd.src_layer_primitive_desc());
  auto hidden_in_mem = _reorder(hidden_in_usr_mem, rnn_forward_pd.src_iter_primitive_desc());
  auto weight_ih_mem = _reorder(weight_ih_usr_mem, rnn_forward_pd.weights_layer_primitive_desc());
  auto weight_hh_mem = _reorder(weight_hh_usr_mem, rnn_forward_pd.weights_iter_primitive_desc());
  auto bias_mem = _reorder(bias_usr_mem, rnn_forward_pd.bias_primitive_desc());
  auto output_mem = _reorder(output_usr_mem, rnn_forward_pd.dst_layer_primitive_desc());
  auto hidden_out_mem = _reorder(hidden_out_usr_mem, rnn_forward_pd.dst_iter_primitive_desc());

  std::vector<primitive> net;
  net.push_back(rnn_forward(rnn_forward_pd, input_mem, hidden_in_mem, weight_ih_mem, weight_hh_mem,
    bias_mem, output_mem, hidden_out_mem, workspace_mem));

  if (hidden_out_mem != hidden_out_usr_mem) {
    net.push_back(reorder(hidden_out_mem, hidden_out_usr_mem));
  }

  Stream::Instance().get_stream().submit(net);

  } catch (error &e) {
    std::cerr << "message: " << e.message << std::endl;
  }
 
  return std::make_tuple(hy, cy, workspace);
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_cell_backward(
    const Tensor& input, TensorList weight, const Tensor& hx, const Tensor& cx,
    const Tensor& hy, const Tensor& cy, const Tensor& grad_hy, const Tensor& grad_cy,
    const Tensor& workspace) {
  std::cout << "hy: " << hy.defined() << std::endl;
  std::cout << "cy: " << cy.defined() << std::endl;
  std::cout << "hx: " << hx.defined() << std::endl;
  std::cout << "cx: " << cx.defined() << std::endl;
  std::cout << "grad_hy: " << grad_hy.defined() << std::endl;
  std::cout << "grad_cy: " << grad_cy.defined() << std::endl;

  auto train = true;

  AT_CHECK(train, "mkldnn_rnn_cell backward can only be called in training mode");

  // TODO: cache hidden_in, hidden_out from forward?
  // NB: MKLDNN requires to concat hx and cx for lstm
  Tensor hidden_in, hidden_out, grad_hidden_out, grad_hidden_in, grad_hx, grad_cx;
  if (cx.defined()) {
    hidden_in = at::cat({hx, cx}, 0);
    hidden_out = at::cat({hy, cy}, 0);
    std::cout << "1" << std::endl;
    grad_hidden_out = at::cat({grad_hy, grad_cy}, 0);
    std::cout << "2" << std::endl;
    grad_hidden_in = at::empty_like(grad_hidden_out);
    std::vector<Tensor> hidden_arr = grad_hidden_in.chunk(2, 0);
    grad_hx = hidden_arr[0];
    grad_cx = hidden_arr[1];
  } else {
    hidden_in = hx;
    hidden_out = hy;
    grad_hidden_out = grad_hy;
    grad_hidden_in = at::empty_like(grad_hidden_out);
    grad_hx = grad_hidden_in;
    grad_cx = at::empty({0}, hx.options()); // NB: Not allowed to return undefined tensors
  }

  // TODO: check if we need to clone this?
  auto output = hy.clone();
  auto grad_output = grad_hy.clone();
  auto grad_input = at::empty_like(input);

  auto cpu_engine = CpuEngine::Instance().get_engine();
  auto null_memory_ = null_memory(cpu_engine);

  int32_t batch_size = input.size(0);
  int32_t input_size = input.size(1);
  int32_t hidden_size = hx.size(1);

  int32_t time_step = 1;
  int32_t num_layers = 1;
  int32_t num_directions = 1;
  int32_t num_gates = (cx.defined()) ? 4 : 3;
  int32_t num_states = (cx.defined()) ? 2 : 1;

  auto format_tnc = memory::format::tnc;
  auto format_ldigo = memory::format::ldigo;
  auto format_ldgo = memory::format::ldgo;
  auto format_ldsnc = memory::format::ldsnc;

 // TODO: cache weight_ih, weight_hh, bias from forward
  // how to cache across timesteps??
  auto weight_ih = _shuffle_gates(weight[0], num_gates);
  auto weight_hh = _shuffle_gates(weight[1], num_gates);
  auto bias = _shuffle_gates(weight[2] + weight[3], num_gates);
  // NB: format ldigo, fioc
  auto grad_weight_ih = at::empty_like(weight_ih);
  auto grad_weight_hh = at::empty_like(weight_hh);
  auto grad_bias = at::empty_like(bias);

  auto rnn_prop = prop_kind::backward;
  auto rnn_algo = (cx.defined()) ? algorithm::vanilla_lstm :gru_linear_before_reset;
  auto rnn_dir = rnn_direction::unidirectional_left2right;

  memory::dims input_tz = {time_step, batch_size, input_size};
  memory::dims weight_ih_tz = {num_layers, num_directions, input_size, num_gates, hidden_size};
  memory::dims weight_hh_tz = {num_layers, num_directions, hidden_size, num_gates, hidden_size};
  memory::dims bias_tz = {num_layers, num_directions, num_gates, hidden_size};
  memory::dims hidden_tz = {num_layers, num_directions, num_states, batch_size, hidden_size};
  memory::dims output_tz = {time_step, batch_size, hidden_size};

  auto input_md = _format_md(input_tz, format_tnc);
  auto hidden_md = _generic_md(hidden_tz);
  auto weight_ih_md = _generic_md(weight_ih_tz);
  auto weight_hh_md = _generic_md(weight_hh_tz);
  auto bias_md = _generic_md(bias_tz);
  auto output_md = _format_md(output_tz, format_tnc);

  auto rnn_cell_ = rnn_cell::desc(rnn_algo);
try{
  auto rnn_backward_desc = rnn_backward::desc(rnn_prop, rnn_cell_, rnn_dir,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md,
    input_md, hidden_md, weight_ih_md, weight_hh_md, bias_md, output_md, hidden_md);

  auto rnn_backward_pd = rnn_backward::primitive_desc(rnn_backward_desc, cpu_engine);

  auto input_usr_mem = memory({input_md, cpu_engine}, input.data_ptr());
  auto hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_in.data_ptr());
  auto weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldigo), cpu_engine}, weight_ih.data_ptr());
  auto weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldigo), cpu_engine}, weight_hh.data_ptr());
  auto bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, bias.data_ptr());
  auto output_usr_mem = memory({output_md, cpu_engine}, output.data_ptr());
  auto hidden_out_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, hidden_out.data_ptr());
  auto grad_input_usr_mem = memory({input_md, cpu_engine}, grad_input.data_ptr());
  auto grad_hidden_in_usr_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, grad_hidden_in.data_ptr());
  auto grad_weight_ih_usr_mem = memory({_format_md(weight_ih_tz, format_ldigo), cpu_engine}, grad_weight_ih.data_ptr());
  auto grad_weight_hh_usr_mem = memory({_format_md(weight_hh_tz, format_ldigo), cpu_engine}, grad_weight_hh.data_ptr());
  auto grad_bias_usr_mem = memory({_format_md(bias_tz, format_ldgo), cpu_engine}, grad_bias.data_ptr());
  auto grad_output_usr_mem = memory({output_md, cpu_engine}, grad_output.data_ptr());
  auto grad_hidden_out_use_mem = memory({_format_md(hidden_tz, format_ldsnc), cpu_engine}, grad_hidden_out.data_ptr());

  auto input_mem = _reorder(input_usr_mem, rnn_backward_pd.src_layer_primitive_desc());
  auto hidden_in_mem = _reorder(hidden_in_usr_mem, rnn_backward_pd.src_iter_primitive_desc());
  auto weight_ih_mem = _reorder(weight_ih_usr_mem, rnn_backward_pd.weights_layer_primitive_desc());
  auto weight_hh_mem = _reorder(weight_hh_usr_mem, rnn_backward_pd.weights_iter_primitive_desc());
  auto bias_mem = _reorder(bias_usr_mem, rnn_backward_pd.bias_primitive_desc());
  auto output_mem = _reorder(output_usr_mem, rnn_backward_pd.dst_layer_primitive_desc());
  auto hidden_out_mem = _reorder(hidden_out_usr_mem, rnn_backward_pd.dst_iter_primitive_desc());
  auto grad_input_mem = _reorder(grad_input_usr_mem, rnn_backward_pd.diff_src_layer_primitive_desc());
  auto grad_hidden_in_mem = _reorder(grad_hidden_in_usr_mem, rnn_backward_pd.diff_src_iter_primitive_desc());
  auto grad_weight_ih_mem = _reorder(grad_weight_ih_usr_mem, rnn_backward_pd.diff_weights_layer_primitive_desc());
  auto grad_weight_hh_mem = _reorder(grad_weight_hh_usr_mem, rnn_backward_pd.diff_weights_iter_primitive_desc());
  auto grad_bias_mem = _reorder(grad_bias_usr_mem, rnn_backward_pd.diff_bias_primitive_desc());
  auto grad_output_mem = _reorder(grad_output_usr_mem, rnn_backward_pd.diff_dst_layer_primitive_desc());
  auto grad_hidden_out_mem = _reorder(grad_hidden_out_use_mem, rnn_backward_pd.diff_dst_iter_primitive_desc());
  auto workspace_mem = memory(rnn_backward_pd.workspace_primitive_desc(), workspace.data_ptr());

  std::vector<primitive> net;
  net.push_back(rnn_backward(rnn_backward_pd, input_mem, hidden_in_mem, weight_ih_mem, weight_hh_mem, bias_mem,
    output_mem, hidden_out_mem, grad_input_mem, grad_hidden_in_mem, grad_weight_ih_mem, grad_weight_hh_mem,
    grad_bias_mem, grad_output_mem, grad_hidden_out_mem, workspace_mem));

  if (grad_input_mem != grad_input_usr_mem) {
    net.push_back(reorder(grad_input_mem, grad_input_usr_mem));
  }

  if (grad_hidden_in_mem != grad_hidden_in_usr_mem) {
    net.push_back(reorder(grad_hidden_in_mem, grad_hidden_in_usr_mem));
  }

  if (grad_weight_ih_mem != grad_weight_ih_usr_mem) {
    net.push_back(reorder(grad_weight_ih_mem, grad_weight_ih_usr_mem));
  }

  if (grad_weight_hh_mem != grad_weight_hh_usr_mem) {
    net.push_back(reorder(grad_weight_hh_mem, grad_weight_hh_usr_mem));
  }

  if (grad_bias_mem != grad_bias_usr_mem) {
    net.push_back(reorder(grad_bias_mem, grad_bias_usr_mem));
  }

  Stream::Instance().get_stream().submit(net);



} catch (error &e) {
    std::cerr << "message: " << e.message << std::endl;
}

  // TODO: do gate shuffle
  std::vector<Tensor> grad_weights;
  grad_weights.emplace_back(grad_weight_ih);
  grad_weights.emplace_back(grad_weight_hh);
  grad_weights.emplace_back(grad_bias);
  grad_weights.emplace_back(grad_bias);

  return std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>>{grad_input, grad_hx, grad_cx, grad_weights};
}

}} // namespace at::native

#endif
