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
    return {rnn.num_layers * rnn.num_directions(), tensors.seq_length, tensors.mini_batch, factor * rnn.hidden_size};
  }

  template<typename DType>
  inline DType sigm(DType x) { return 1.0f / (1.0f + std::exp(-x)); }

  template<typename DType>
  void GRUCell_forward(const Tensor& inputWeight1, const Tensor& hiddenWeight2,
    const Tensor& bias1, const Tensor& bias2, const Tensor& hx, Tensor& hy,
    Tensor& reserve, bool train)
  {
    auto has_bias = bias1.defined();

    auto hsz = hx.size(1);
    auto count = hy.numel();

    DType *inputWeight1_p = (DType*)inputWeight1.data_ptr();
    DType *hiddenWeight2_p = (DType*)hiddenWeight2.data_ptr();
    DType *bias1_p = has_bias ? (DType*)bias1.data_ptr() : NULL;
    DType *bias2_p = has_bias ? (DType*)bias2.data_ptr() : NULL;
    DType *hx_p = (DType*)hx.data_ptr();
    DType *hy_p = (DType*)hy.data_ptr();
    DType *reserve_p = train ? (DType*)reserve.data_ptr() : NULL;

    #pragma omp parallel for
    for (size_t index = 0; index < count; index++) {
      size_t offset = (index/hsz)*3*hsz+index%hsz;
      size_t offset_s = (index/hsz)*5*hsz+index%hsz;

      DType ir = inputWeight1_p[offset+0*hsz];
      DType ii = inputWeight1_p[offset+1*hsz];
      DType in = inputWeight1_p[offset+2*hsz];
      DType hr = hiddenWeight2_p[offset+0*hsz];
      DType hi = hiddenWeight2_p[offset+1*hsz];
      DType hn = hiddenWeight2_p[offset+2*hsz];

      DType hx_ = hx_p[index];
      DType *hy_ = &hy_p[index];

      DType b1r, b1i, b1n, b2r, b2i, b2n;
      b1r = has_bias ? bias1_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b1i = has_bias ? bias1_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b1n = has_bias ? bias1_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b2r = has_bias ? bias2_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b2i = has_bias ? bias2_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b2n = has_bias ? bias2_p[index%hsz+2*hsz] : static_cast<DType>(0);

      DType rg, ig, ng;
      rg = ir + hr + b1r + b2r;
      ig = ii + hi + b1i + b2i;
      rg = sigm<DType>(rg);
      ig = sigm<DType>(ig);
      ng = in + b1n + rg*(hn+b2n);
      ng = std::tanh(ng);

      *hy_ = ng + ig * (hx_-ng);

      if (train) {
        //SAVE FOR BACKWARDS
        reserve_p[offset_s+0*hsz] = rg;
        reserve_p[offset_s+1*hsz] = ig;
        reserve_p[offset_s+2*hsz] = ng;
        reserve_p[offset_s+3*hsz] = hx_;
        reserve_p[offset_s+4*hsz] = hn + b2n;
      }
    }
  }

  template<typename DType>
  void LSTMCell_forward(const Tensor& inputWeight1, const Tensor& hiddenWeight2,
    const Tensor& bias1, const Tensor& bias2, const Tensor& cx, Tensor& hy, Tensor& cy,
    Tensor& reserve, bool train)
  {
    auto has_bias = bias1.defined();

    auto hsz = cx.size(1);
    auto count = hy.numel();

    DType *inputWeight1_p = (DType*)inputWeight1.data_ptr();
    DType *hiddenWeight2_p = (DType*)hiddenWeight2.data_ptr();
    DType *bias1_p = has_bias ? (DType*)bias1.data_ptr() : NULL;
    DType *bias2_p = has_bias ? (DType*)bias2.data_ptr() : NULL;
    DType *cx_p = (DType*)cx.data_ptr();
    DType *hy_p = (DType*)hy.data_ptr();
    DType *cy_p = (DType*)cy.data_ptr();
    DType *reserve_p = train ? (DType*)reserve.data_ptr() : NULL;

    #pragma omp parallel for
    for (size_t index = 0; index < count; index++) {
      size_t offset = (index/hsz)*4*hsz+index%hsz;

      DType iig = inputWeight1_p[offset+0*hsz];
      DType ifg = inputWeight1_p[offset+1*hsz];
      DType icg = inputWeight1_p[offset+2*hsz];
      DType iog = inputWeight1_p[offset+3*hsz];

      DType hig = hiddenWeight2_p[offset+0*hsz];
      DType hfg = hiddenWeight2_p[offset+1*hsz];
      DType hcg = hiddenWeight2_p[offset+2*hsz];
      DType hog = hiddenWeight2_p[offset+3*hsz];

      DType cx_ = cx_p[index];
      DType *hy_ = &hy_p[index];
      DType *cy_ = &cy_p[index];

      DType b1i, b1f, b1c, b1o, b2i, b2f, b2c, b2o;
      b1i = has_bias ? bias1_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b1f = has_bias ? bias1_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b1c = has_bias ? bias1_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b1o = has_bias ? bias1_p[index%hsz+3*hsz] : static_cast<DType>(0);
      b2i = has_bias ? bias2_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b2f = has_bias ? bias2_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b2c = has_bias ? bias2_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b2o = has_bias ? bias2_p[index%hsz+3*hsz] : static_cast<DType>(0);

      DType ig, fg, cg, og;
      ig = iig + hig + b1i + b2i;
      fg = ifg + hfg + b1f + b2f;
      cg = icg + hcg + b1c + b2c;
      og = iog + hog + b1o + b2o;

      ig = sigm(ig);
      fg = sigm(fg);
      cg = std::tanh(cg);
      og = sigm(og);

      *cy_ = (fg * cx_) + (ig * cg);
      *hy_ = og * std::tanh(*cy_);

      if (train) {
        //SAVE FOR BACKWARDS
        reserve_p[offset+0*hsz] = ig;
        reserve_p[offset+1*hsz] = fg;
        reserve_p[offset+2*hsz] = cg;
        reserve_p[offset+3*hsz] = og;
      }
    }
  }

  Tensor _rnn_layer_forward(const RNNParams& fn, const Tensor& input, TensorList weights,
    const Tensor& hx, const Tensor& cx, Tensor& hy, Tensor& cy, Tensor& reserve,
    bool train, bool reverse)
  {
    auto has_bias = (weights.size() == 4);

    auto weight1 = weights[0];
    auto weight2 = weights[1];
    auto bias1 = has_bias ? weights[2] : Tensor();
    auto bias2 = has_bias ? weights[3] : Tensor();

    auto mode = fn.rnn.mode;
    auto seq_length = input.size(0);
    auto mini_batch = input.size(1);
    auto input_size = input.size(2);
    auto gate_size = weight2.size(0);
    auto hidden_size = weight2.size(1);

    // NB: output per layer per direction
    auto output = input.type().tensor({seq_length, mini_batch, hidden_size});
    // NB: fuse xW from each timestep
    auto inputWeight1 = input.view({-1, input_size}).matmul(weight1.t()).view({seq_length, mini_batch, gate_size});

    auto hx_t = hx;
    auto cx_t = cx;
    for (size_t t = 0; t < seq_length; t++) {
      size_t ts = (reverse) ? (seq_length-t-1) : t;
      auto inputWeight1_t = inputWeight1[ts];
      auto hiddenWeight2_t = hx_t.matmul(weight2.t());
      AT_ASSERTM(inputWeight1_t.sizes().equals(hiddenWeight2_t.sizes()),
        "input weight product and hidden weight product mismatch")
      auto hy_t = output[ts];
      auto cy_t = cx.defined() ? cx.type().zeros_like(cx) : hx.type().tensor();
      auto reserve_t = train ? reserve[ts] : hx.type().tensor();

      if (mode == MKLDNN_LSTM) {
        LSTMCell_forward<float>(inputWeight1_t, hiddenWeight2_t, bias1, bias2, cx_t, hy_t, cy_t, reserve_t, train);
      } else if (mode == MKLDNN_GRU) {
        GRUCell_forward<float>(inputWeight1_t, hiddenWeight2_t, bias1, bias2, hx_t, hy_t, reserve_t, train);
      } else {
        throw std::runtime_error("MKLDNN unsupported RNN mode");
      }
      // NB: update hidden/cell state for next time step
      hx_t = hy_t;
      cx_t = cy_t;
    }
    // NB: update hy/cy for the final time step
    hy.copy_(hx_t);
    cy.copy_(cx_t);

    return output;
  }

  void _rnn_layer_backward(void)
  {
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
  auto reserve_size = _reserve_size(fn.rnn, fn.tensors);

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
  // NB: Not allowed to return undefined tensors
  auto cy = cx.defined() ? cx.type().tensor(hidden_size) : hx.type().tensor();
  auto reserve = fn_train ? hx.type().tensor(reserve_size) : hx.type().tensor();

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions();
  auto has_bias = (weight_stride0 == 4);
  auto layer_input = input;

  for (size_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_output(num_directions);
    for (size_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;

      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : hx.type().tensor();
      auto layer_cy = cx.defined() ? cy[index] : hx.type().tensor();
      auto layer_reserve = fn_train ? reserve[index] : hx.type().tensor();

      auto reverse = (direction > 0);
      layer_output[direction] = _rnn_layer_forward(fn, layer_input, layer_weights,
         layer_hx, layer_cx, layer_hy, layer_cy, layer_reserve, fn_train, reverse);
    }
    layer_input = at::cat(layer_output, input.dim() - 1);
  }

  //TODO unpack output and transpose!
  output = layer_input;
  if (batch_first && !is_input_packed) {
    output.transpose_(0, 1);
  }

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
