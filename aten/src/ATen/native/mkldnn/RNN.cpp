#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Error.h>
#include <ATen/MatrixRef.h>
#include <ATen/NativeFunctions.h>

#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor> _mkldnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes) {
  throw std::runtime_error("_mkldnn_rnn: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _mkldnn_rnn_backward(
    const Tensor& input,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r,
    const Tensor& grad_hy_r, const Tensor& grad_cy_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& reserve, std::array<bool, 4> output_mask) {
  throw std::runtime_error("_mkldnn_rnn_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#else // AT_MKLDNN_ENABLED()

namespace at { namespace native {

namespace {

  typedef enum
  {
    MKLDNN_RNN_RELU = 0, // Stock RNN with ReLu activation
    MKLDNN_RNN_TANH = 1, // Stock RNN with tanh activation
    MKLDNN_LSTM = 2,     // LSTM with no peephole connections
    MKLDNN_GRU = 3       // GRU with cuDNN flavor
  } mkldnnRNNMode_t;

  // RNNDescriptor
  struct RNNDescriptorParams {
    int64_t hidden_size;
    int64_t num_layers;
    bool bidirectional;
    mkldnnRNNMode_t mode;

    int64_t num_directions() const {
      return bidirectional ? 2 : 1;
    }

    void set_mode(int64_t fn_mode) {
      switch (fn_mode) {
        case MKLDNN_RNN_RELU:
          mode = MKLDNN_RNN_RELU;
          break;
        case MKLDNN_RNN_TANH:
          mode = MKLDNN_RNN_TANH;
          break;
        case MKLDNN_LSTM:
          mode = MKLDNN_LSTM;
          break;
        case MKLDNN_GRU:
          mode = MKLDNN_GRU;
          break;
        default:
        {
          std::ostringstream oss;
          oss << "unrecognized MKLDNN RNN mode " << fn_mode;
          throw std::runtime_error(oss.str());
        }
      }
    }

    void set(int64_t mode, int64_t hidden_size, int64_t num_layers, bool bidirectional) {
      this->set_mode(mode);
      this->hidden_size = hidden_size;
      this->num_layers = num_layers;
      this->bidirectional = bidirectional;
    }
  };

  // TensorDescriotor
  struct TensorDescriptorListParams {
    IntList batch_sizes;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t input_size;
    // Only valid for packed input
    int64_t batch_sizes_sum;

    bool is_input_packed() const {
      return batch_sizes.size() != 0;
    }

    void set(IntList input_sizes, IntList batch_sizes_, bool batch_first) {
      batch_sizes = batch_sizes_;
      if (is_input_packed()) {
        seq_length = batch_sizes.size();
        mini_batch = batch_sizes[0];
        batch_sizes_sum = input_sizes[0];
        input_size = input_sizes[1];
      } else {
        if (batch_first) {
          seq_length = input_sizes[1];
          mini_batch = input_sizes[0];
        } else {
          seq_length = input_sizes[0];
          mini_batch = input_sizes[1];
        }
        input_size = input_sizes[2];
        batch_sizes_sum = -1;
      }
    }
  };

  struct RNNParams {
    RNNDescriptorParams rnn;
    TensorDescriptorListParams tensors;
  };

  std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
      return {tensors.batch_sizes_sum, tensors.input_size};
    } else {
      return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
    }
  }

  std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size};
  }

  std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    if (tensors.is_input_packed()) {
      return {tensors.batch_sizes_sum, rnn.hidden_size * rnn.num_directions()};
    } else {
      return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
    }
  }

  std::vector<int64_t> _reserve_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    int64_t factor = (rnn.mode == MKLDNN_LSTM) ? 4 : 5;
    return {rnn.num_layers, rnn.num_directions(), tensors.seq_length, tensors.mini_batch, factor * rnn.hidden_size};
  }

  Tensor _rnn_layer_forward(const RNNParams& fn, const Tensor& input, const Tensor& hx, const Tensor& cx,
    const Tensor& weight_ih, const Tensor& weight_hh, const Tensor& bias_ih, const Tensor& bias_hh,
    Tensor& hy, Tensor& cy, bool reverse)
  {
    auto mode = fn.rnn.mode;
    auto seq_length = fn.tensors.seq_length;
    auto mini_batch = fn.tensors.mini_batch;
    auto hidden_size = fn.rnn.hidden_size;

    auto output = input.type().tensor({seq_length, mini_batch, hidden_size});



    throw std::runtime_error("_rnn_layer_forward");
    return output;
  }

  void _rnn_layer_backward(void) {
    std::cout << "_rnn_layer_backward" << std::endl;
  }
} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _mkldnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes)
{
  auto input = input_r;

  RNNParams fn;
  fn.rnn.set(fn_mode, fn_hidden_size, fn_num_layers, fn_bidirectional);
  fn.tensors.set(input.sizes(), fn_batch_sizes, batch_first);

  if (fn.rnn.mode != MKLDNN_LSTM) {
    if (cx.defined()) {
      throw std::runtime_error("rnn: illegal defined cx for non-LSTM RNN");
    }
  }

  auto is_input_packed = fn.tensors.is_input_packed();
  if (batch_first && !is_input_packed) {
    input = input.transpose(0, 1);
  }

  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  if (!hx.is_contiguous()) {
    throw std::runtime_error("rnn: hx is not contiguous");
  }
  if (cx.defined() && !cx.is_contiguous()) {
    throw std::runtime_error("rnn: cx is not contiguous");
  }

  // TODO unpack input
  input = input.contiguous();
  auto output = input.type().tensor(output_size);
  auto hy = hx.type().tensor(hidden_size);
  auto cy = cx.defined() ? cx.type().tensor(hidden_size) : hx.type().tensor();
  auto reserve = hx.

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions();
  auto has_bias = (weight_stride0 == 4);
  auto layer_input = input;

  for (size_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_output(num_directions);
    for (size_t direction = 0; direction < num_directions; direction++) {
      auto layer_weights = weights[layer * num_directions + direction];

      Tensor weight_ih, weight_hh, bias_ih, bias_hh;
      weight_ih = layer_weights[0];
      weight_hh = layer_weights[1];
      if (has_bias) {
        // TODO test if works
        bias_ih = layer_weights[2];
        bias_hh = layer_weights[3];
      }

      auto layer_hx = hx[layer][direction];
      auto layer_hy = hy[layer][direction];
      auto layer_cx = cx.defined() ? cx[layer][direction] : hx.type().tensor();
      auto layer_cy = cx.defined() ? cy[layer][direction] : hx.type().tensor();

      auto reverse = (direction > 0);
      layer_output[direction] = _rnn_layer_forward(fn, layer_input, layer_hx, layer_cx,
        weight_ih, weight_hh, bias_ih, bias_hh, layer_hy, layer_cy, reverse);



    }
    layer_input = at::cat(layer_output, input.dim() - 1);
  }

  //TODO unpack output and transpose!
  output = layer_input;






  Tensor reserve;
  return std::make_tuple(output, hy, cy, reserve);
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _mkldnn_rnn_backward(
    const Tensor& input,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    const Tensor& output, const Tensor& grad_output_r,
    const Tensor& grad_hy_r, const Tensor& grad_cy_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& reserve, std::array<bool, 4> output_mask)
{
  throw std::runtime_error("_mkldnn_rnn_backward: ATen not compiled with MKLDNN support");
}

}} // namespace at::native

#endif // AT_MKLDNN_ENABLED()
