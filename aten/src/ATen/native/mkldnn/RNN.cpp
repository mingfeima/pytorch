#include <ATen/native/RNN.h>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/util/Exception.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

#define ONE_HIDDEN_RNN(NAME)                                                   \
std::tuple<Tensor, Tensor> NAME##_mkldnn_stub(                                 \
    const Tensor& input, const Tensor& hx, TensorList params, bool has_biases, \
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
  AT_ERROR("NAME##_mkldnn_stub: ATen not compiled with MKLDNN support");       \
}

ONE_HIDDEN_RNN(gru)
ONE_HIDDEN_RNN(rnn_tanh)
ONE_HIDDEN_RNN(rnn_relu)

std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {
  AT_ERROR("lstm_mkldnn_stub: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes) {
  AT_ERROR("mkldnn_rnn: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_EBABLED

#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>

namespace at { namespace native {

struct RNNParams {
  ideep::rnn_kind mode;
  int64_t seq_length;
  int64_t mini_batch;
  int64_t input_size;
  int64_t hidden_size;
  int64_t num_directions;
  int64_t num_layers;
  bool batch_first;
  bool train;
  IntArrayRef batch_sizes;
  int64_t num_gates;
  int64_t num_bias_gates;

  RNNParams(const Tensor& input, IntArrayRef batch_sizes_,
      int64_t mode_, int64_t hidden_size_, int64_t num_layers_,
      bool bidirectional, bool batch_first_, bool train_) {
    mode = static_cast<ideep::rnn_kind>(mode_);
    batch_first = batch_first_;
    if (batch_first) {
      seq_length = input.size(1);
      mini_batch = input.size(0);
    } else {
      seq_length = input.size(0);
      mini_batch = input.size(1);
    }
    input_size = input.size(2);
    hidden_size = hidden_size_;
    num_directions = bidirectional ? 2 : 1;
    num_layers = num_layers_;
    train = train_;
    batch_sizes = batch_sizes_;
    if (mode == ideep::rnn_kind::LSTM) {
      num_gates = 4;
      num_bias_gates = 4;
    } else if (mode == ideep::rnn_kind::GRU) {
      num_gates = 3;
      num_bias_gates = 4;
    } else {
      // RNN_RELU; RNN_TANH
      num_gates = 1;
      num_bias_gates = 1;
    }
  }

  bool is_input_packed() const {
    return batch_sizes.size() != 0;
  }

  // mkldnn memory descriptors
  using format = ideep::format_tag;
  using desc = ideep::tensor::desc;
  using dtype = ideep::tensor::data_type;
  desc src_layer_desc(int64_t _input_size) const {
    return {{seq_length, mini_batch, _input_size}, dtype::f32, format::tnc};
  }
  desc src_iter_desc() const {
    return {{1, 1, mini_batch, hidden_size}, dtype::f32, format::ldnc};
  }
  // logical size described as ldigo
  desc weights_layer_desc(int64_t _input_size) const {
    return {{1, 1, _input_size, num_gates, hidden_size}, dtype::f32, format::ldgoi};
  }
  desc weights_iter_desc() const {
    return {{1, 1, hidden_size, num_gates, hidden_size}, dtype::f32, format::ldgoi};
  }
  desc bias_desc() const {
    return {{1, 1, num_bias_gates, hidden_size}, dtype::f32, format::ldgo};
  }
  desc dst_layer_desc() const {
    return {{seq_length, mini_batch, hidden_size}, dtype::f32, format::tnc};
  }
  desc dst_iter_desc() const {
    return {{1, 1, mini_batch, hidden_size}, dtype::f32, format::ldnc};
  }
};

std::vector<int64_t> _hidden_size(const RNNParams& rnn) {
  return {rnn.num_layers * rnn.num_directions, rnn.mini_batch, rnn.hidden_size};
}

template<bool is_single_direction>
std::vector<int64_t> _output_size(const RNNParams& rnn) {
  auto output_channels = is_single_direction ? rnn.hidden_size
                                             : rnn.hidden_size * rnn.num_directions;
  return {rnn.seq_length, rnn.mini_batch, output_channels};
}

// MKLDNN GRU gate order is different from PyTorch's which requires gates shuffle
// (let rt,zt,nt be reset, update, new gates respectively)
//
//   MKLDNN GRU weight_ih/weight_hh gates order: (zt, rt, nt)
//   PyTorch GRU weight_ih/weight_hh gates order: (rt, zt, nt)
//
// MKLDNN GRU bias has 4 gates instead of 3
//  (PyTorch GRU bias)     (MKLDNN GRU bias)
//
//  bias_ih    bias_hh          bias
//  +-----+    +-----+       +---------+
//  | rt1 |    | rt2 |       | zt1+zt2 |
//  |-----|    |-----|       |---------|
//  | zt1 |    | zt2 |       | rt1+rt2 |
//  |-----|    |-----|       |---------|
//  | nt1 |    | nt2 |       |   nt1   |
//  +-----+    +-----+       |---------|
//                           |   nt2   |
//                           +---------+
//
Tensor _shuffle_weight(const Tensor& weight, int64_t mode) {
  auto weight_t = weight.contiguous();
  if (static_cast<ideep::rnn_kind>(mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> gates = weight_t.chunk(3, /*gates*/0);
    return at::cat({gates[1], gates[0], gates[2]}, /*gates*/0);
  }
  return weight_t;
};

Tensor _shuffle_bias(const Tensor& bias_ih, const Tensor& bias_hh, int64_t mode) {
  if (static_cast<ideep::rnn_kind>(mode) == ideep::rnn_kind::GRU) {
    std::vector<Tensor> b1 = bias_ih.chunk(3, /*output_channels*/0);
    std::vector<Tensor> b2 = bias_hh.chunk(3, /*output_channels*/0);
    return at::cat({b1[1] + b2[1], b1[0] + b2[0], b1[2], b2[2]}, /*output_channels*/0);
  }
  return bias_ih + bias_hh;
};

// Create mkldnn memory view from ATen tensor
static inline ideep::tensor get_mkldnn_tensor(
    const Tensor& tensor, const ideep::tensor::desc& desc) {
  return {desc, tensor.data_ptr<float>()};
}

static inline ideep::tensor get_mkldnn_tensor(
    void* data_ptr, const ideep::tensor::desc& desc) {
  return {desc, data_ptr};
}

Tensor mkldnn_rnn_layer(Tensor& hy_, Tensor& cy_,
    const Tensor& input, TensorList weights,
    const Tensor& hx_, const Tensor& cx_, ideep::tensor& ws,
    bool reverse, const RNNParams& rnn) {
  TORCH_CHECK(weights.size() == 2 || weights.size() == 4);

  auto output_size = _output_size</*is_single_direction*/true>(rnn);
  auto output = at::empty(output_size, input.options());

  bool has_bias = weights.size() == 4;
  auto weight_ih = _shuffle_weight(weights[0], rnn.mode);
  auto weight_hh = _shuffle_weight(weights[1], rnn.mode);
  auto bias = has_bias ? _shuffle_bias(weights[2], weights[3], rnn.mode)
                       : at::zeros({rnn.num_bias_gates * rnn.hidden_size}, weight_ih.options());

  // per layer input size
  int64_t layer_input_size = input.size(2);
  auto x = get_mkldnn_tensor(input, rnn.src_layer_desc(layer_input_size));
  auto hx = get_mkldnn_tensor(hx_, rnn.src_iter_desc());
  auto w1 = get_mkldnn_tensor(weight_ih, rnn.weights_layer_desc(layer_input_size));
  auto w2 = get_mkldnn_tensor(weight_hh, rnn.weights_iter_desc());
  auto b = get_mkldnn_tensor(bias, rnn.bias_desc());
  auto y = get_mkldnn_tensor(output, rnn.dst_layer_desc());
  auto hy = get_mkldnn_tensor(hy_, rnn.dst_iter_desc());

  auto _rnn_kind = static_cast<ideep::rnn_kind>(rnn.mode);
  if (rnn.train) {
    if (_rnn_kind == ideep::rnn_kind::LSTM) {
      auto cx = get_mkldnn_tensor(cx_, rnn.src_iter_desc());
      auto cy = get_mkldnn_tensor(cy_, rnn.dst_iter_desc());
      ideep::lstm_forward::compute(x, hx, cx, w1, w2, b, y, hy, cy, ws, reverse);
    }

  } else {
    if (_rnn_kind == ideep::rnn_kind::LSTM) {
      auto cx = get_mkldnn_tensor(cx_, rnn.src_iter_desc());
      auto cy = get_mkldnn_tensor(cy_, rnn.dst_iter_desc());
      ideep::lstm_forward::compute(x, hx, cx, w1, w2, b, y, hy, cy, reverse);
    } else if (_rnn_kind == ideep::rnn_kind::GRU) {
      ideep::gru_forward_inference::compute(x, hx, w1, w2, b, y, hy, reverse);
    } else {
      TORCH_CHECK(_rnn_kind == ideep::rnn_kind::RNN_RELU || _rnn_kind == ideep::rnn_kind::RNN_TANH,
                  "mkldnn_rnn: unsuppored rnn mode: ", rnn.mode);
      ideep::rnn_forward_inference::compute(x, hx, w1, w2, b, y, hy, rnn.mode, reverse);
    }
  }

  return output;
}

// NB
void prepare_workspace(
    Tensor& workspace, std::vector<ideep::tensor>& mkldnn_workspace_arr,
    const RNNParams& rnn) {
  if (!rnn.train) { return; }
  std::cout << "workspace.numel(): " << workspace.numel() << std::endl;

  auto _rnn_kind = static_cast<ideep::rnn_kind>(rnn.mode);
  auto num_layers = rnn.num_layers;
  auto num_directions = rnn.num_directions;

  std::vector<ideep::tensor::desc> workspace_desc_arr;
  workspace_desc_arr.reserve(num_layers * num_directions);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    for (int64_t direction = 0; direction < num_directions; direction++) {
      int64_t layer_input_size = layer == 0 ? rnn.input_size
                                            : rnn.hidden_size * num_directions;
      auto reverse = (direction > 0);
      if (_rnn_kind == ideep::rnn_kind::LSTM) {
        auto desc = ideep::lstm_forward::get_workspace_desc(
            rnn.src_layer_desc(layer_input_size), rnn.src_iter_desc(), rnn.src_iter_desc(),
            rnn.weights_layer_desc(layer_input_size), rnn.weights_iter_desc(), rnn.bias_desc(),
            rnn.dst_layer_desc(), rnn.dst_iter_desc(), rnn.dst_iter_desc(), reverse);
        workspace_desc_arr.emplace_back(desc);
      }
    }
  }

  int64_t size_bytes = 0;
  for (const auto& desc : workspace_desc_arr) {
    size_bytes += desc.get_size();
  }
  if (workspace.numel() == 0) {
    workspace.resize_({size_bytes});
  } else {
    TORCH_INTERNAL_ASSERT(workspace.numel() == size_bytes, "mkldnn_rnn: workspace size mismatch");
  }
  std::cout << "workspace.numel(): " << workspace.numel() << std::endl;

  int64_t offset = 0;
  auto workspace_data = workspace.data_ptr<unsigned char>();
  mkldnn_workspace_arr.reserve(num_layers * num_directions);
  for (const auto& desc : workspace_desc_arr) {
    mkldnn_workspace_arr.emplace_back(
        get_mkldnn_tensor((void*)(workspace_data + offset), desc));
    offset += desc.get_size();
  }
}

// MKLDNN RNN integration notes:
// I. Memory Formats
//   a. mkldnn will use plain formats for input, hx/cx, output, hy/cy
//      and possibly use blocked formats for weights depending shape info.
//   b. All mkldnn memorys are created (in plain format) as views on ATen tensor,
//      the weight reorder(if any) is handed automatically inside ideep (mkldnn bridge)
//
// II. MKLDNN Primitive Mapping
//   a. mkldnn rnn primitive doesn't support training with dropout or padded input sequence.
//   b. here break a single RNN module into { num_layers * num_directions } mkldnn rnn primitives
//      for future need to cover these feature gaps.
//
//TODO: a. training with dropout
//   b. padded sequence input support
//
std::tuple<Tensor, Tensor, Tensor, Tensor> mkldnn_rnn(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes) {
  TORCH_CHECK(!train || dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn: illegal defined cx for non-LSTM RNN");
  }

  RNNParams rnn(input_, batch_sizes, mode, hidden_size, num_layers, bidirectional, batch_first, train);

  auto input = input_;
  if (batch_first && !rnn.is_input_packed()) {
    input = input.transpose(0, 1);
  }
  input = input.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.defined() ? cx_.contiguous() : Tensor();

  auto hy = at::empty(_hidden_size(rnn), hx.options());
  // NB: Not allowed to return undefined tensors
  auto cy = cx.defined() ? at::empty(_hidden_size(rnn), cx.options())
                         : at::empty({0}, hx.options());

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto workspace = at::empty({0}, input.options().dtype(kByte));
  std::vector<ideep::tensor> mkldnn_workspace_arr;
  prepare_workspace(workspace, mkldnn_workspace_arr, rnn);

  auto num_directions = rnn.num_directions;
  auto layer_input = input;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_output(num_directions);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;
      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : Tensor();
      auto layer_cy = cx.defined() ? cy[index] : at::empty({0}, input.options());
      auto layer_ws = mkldnn_workspace_arr[index];
      auto reverse = (direction > 0);
      layer_output[direction] = mkldnn_rnn_layer(
          layer_hy, layer_cy, layer_input, layer_weights, layer_hx, layer_cx, layer_ws, reverse, rnn);
    }
    layer_input = num_directions == 1 ? layer_output[0]
                                      : at::cat(layer_output, /*output_channels*/-1);
  }
  auto output = layer_input;

  if (batch_first && !rnn.is_input_packed()) {
    output.transpose_(0, 1);
  }

  return std::make_tuple(output, hy, cy, workspace);
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> mkldnn_rnn_backward(
    const Tensor& input_, TensorList weight, int64_t weight_stride0,
    const Tensor& hx_, const Tensor& cx_,
    const Tensor& output_, const Tensor hy_, const Tensor cy_,
    const Tensor& grad_output_, const Tensor& grad_hy_, const Tensor& grad_cy_,
    int64_t mode, int64_t hidden_size,
    int64_t num_layers, bool batch_first, double dropout_p,
    bool train, bool bidirectional, IntArrayRef batch_sizes,
    std::array<bool, 4> output_mask) {
  TORCH_CHECK(train, "mkldnn_rnn_backward: can only be called in training mode");
  TORCH_CHECK(dropout_p == 0.0, "mkldnn_rnn doesn't support dropout");
  TORCH_CHECK(batch_sizes.size() == 0, "mkldnn_rnn doesn't support packed input");
  if (static_cast<ideep::rnn_kind>(mode) != ideep::rnn_kind::LSTM) {
    TORCH_CHECK(!cx_.defined(), "mkldnn_rnn_backward: illegal defined cx for non-LSTM RNN");
  }
  TORCH_CHECK(cx_.defined() || !output_mask[2], "illegally required grad of cx for non-LSTM RNN");

  if (!grad_output_.defined() && !grad_hy_.defined() && !grad_cy_.defined()) {
    return std::make_tuple(Tensor(), Tensor(), Tensor(), std::vector<Tensor>(weight.size()));
  }

  RNNParams rnn(input_, batch_sizes, mode, hidden_size, num_layers, bidirectional, batch_first, train);
  auto input = input_;
  auto output = output_;

  auto memory_format = at::MemoryFormat::Contiguous;
  auto grad_output = grad_output_.defined() ? grad_output_ : at::zeros_like(output, memory_format);
  auto grad_hy = grad_hy_.defined() ? grad_hy_ : at::zeros_like(hx_, memory_format);
  auto grad_cy = cx_.defined() ? (grad_cy_.defined() ? grad_cy_ : at::zeros_like(cx_, memory_format)) : grad_cy_;

  if (batch_first && !rnn.is_input_packed()) {
    input = input.transpose(0, 1);
    output = output.transpose(0, 1);
    grad_output = grad_output.transpose(0, 1);
  }
  input = input.contiguous();
  output = output.contiguous();
  grad_output = grad_output.contiguous();

  auto hx = hx_.contiguous();
  auto cx = cx_.defined() ? cx_.contiguous() : Tensor();
  auto hy = hy_.contiguous();
  auto cy = cy_.defined() ? cy_.contiguous() : Tensor();
  grad_hy = grad_hy.contiguous();
  grad_cy = grad_cy.contiguous();

  auto grad_input = at::empty_like(input, memory_format);
  auto grad_hx = at::empty_like(hx, memory_format);
  auto grad_cx = cx_.defined() ? at::empty_like(cx, memory_format) : Tensor();

  std::vector<Tensor> grad_weight;
  grad_weight.reserve(weight.size());
  for (const auto& w : weight) {
    grad_weight.emplace_back(at::empty_like(w, memory_format));
  }

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};
  MatrixRef<Tensor> grad_weights{grad_weight, static_cast<size_t>(weight_stride0)};

  std::cout << "mkldnn_rnn_backward" << std::endl;




  return std::make_tuple(grad_input, grad_hx, grad_cx, grad_weight);
}

////////////////////////////////////////////////////////////////////////////////
//// MKLDNN dispatch for the generic RNN ops (at::lstm, at::gru, ...)
////////////////////////////////////////////////////////////////////////////////

namespace {

// Helpers for working with different hidden types.
std::tuple<Tensor, Tensor> unpack_hidden(const Tensor& hidden) {
  return std::make_tuple(hidden, at::Tensor{});
}

std::tuple<Tensor, Tensor> unpack_hidden(const std::tuple<Tensor, Tensor>& hidden) {
  return hidden;
}

template<typename hidden_type>
hidden_type pack_hidden(const Tensor& hx, const Tensor& cx) {
  static_assert(std::is_same<hidden_type, void>::value, "pack_hidden not implemented for this type");
  AT_ERROR("NOT IMPLEMENTED");
}

template<>
Tensor pack_hidden<Tensor>(const Tensor& hx, const Tensor& cx) {
  AT_ASSERT(cx.numel() == 0);
  return hx;
}

template<>
std::tuple<Tensor, Tensor> pack_hidden<std::tuple<Tensor, Tensor>>(const Tensor& hx, const Tensor& cx) {
  return std::make_tuple(hx, cx);
}

template<typename hidden_type>
std::pair<Tensor, hidden_type> mkldnn_impl(
    const Tensor& input, const hidden_type& hidden,
    TensorList params, bool has_biases, ideep::rnn_kind mode,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  Tensor hx, cx;
  std::tie(hx, cx) = unpack_hidden(hidden);
  int64_t hidden_size = hx.size(2);

  // mkldnn_output = std::tuple<output, hy, cy, workspace>
  auto mkldnn_output = at::mkldnn_rnn(
      input, params, has_biases ? 4 : 2,
      hx, cx, static_cast<int>(mode), hidden_size, num_layers, batch_first, dropout_p,
      train, bidirectional, /*batch_sizes*/{});

  return {std::get<0>(mkldnn_output),
          pack_hidden<hidden_type>(std::get<1>(mkldnn_output), std::get<2>(mkldnn_output))};
}

} // anonymous namespace

#define ONE_HIDDEN_RNN(NAME, MODE)                                             \
std::tuple<Tensor, Tensor> NAME##_mkldnn_stub(                                 \
    const Tensor& input, const Tensor& hx, TensorList params, bool has_biases, \
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) { \
                                                                               \
  auto result = mkldnn_impl(input, hx, params, has_biases,                     \
      MODE, num_layers, dropout_p, train, bidirectional, batch_first);         \
  auto output = result.first;                                                  \
  auto hy = result.second;                                                     \
                                                                               \
  return std::make_tuple(output, hy);                                          \
}

ONE_HIDDEN_RNN(gru, ideep::rnn_kind::GRU)
ONE_HIDDEN_RNN(rnn_tanh, ideep::rnn_kind::RNN_TANH)
ONE_HIDDEN_RNN(rnn_relu, ideep::rnn_kind::RNN_RELU)

std::tuple<Tensor, Tensor, Tensor> lstm_mkldnn_stub(
    const Tensor& input, TensorList hx, TensorList params, bool has_biases,
    int64_t num_layers, double dropout_p, bool train, bool bidirectional, bool batch_first) {

  auto result = mkldnn_impl(input, std::make_tuple(hx[0], hx[1]), params, has_biases,
      ideep::rnn_kind::LSTM, num_layers, dropout_p, train, bidirectional, batch_first);
  auto output = result.first;
  auto hy = std::get<0>(result.second);
  auto cy = std::get<1>(result.second);

  return std::make_tuple(output, hy, cy);
}

}} // namespace at::native

#endif // AT_MKLDNN_EBABLED
