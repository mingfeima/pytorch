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
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
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
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
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

  // DropoutDescriptor
  struct DropoutDescriptorParams {
    bool train;
    double dropout;

    void set(bool train, double dropout) {
      this->train = train;
      this->dropout = dropout;
    }
  };

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
    std::vector<int64_t> batch_lengths;
    int64_t seq_length;
    int64_t mini_batch;
    int64_t input_size;
    int64_t batch_sizes_sum;

    bool is_input_packed() const {
      return batch_sizes.size() != 0;
    }

    void set_batch_lengths(const IntList& batch_sizes_) {
      if (batch_sizes_.size() == 0) {
        return;
      }
      batch_lengths.resize(batch_sizes_[0]);
      for (const auto& bs : batch_sizes_) {
        for (int64_t n = 0; n < bs; n++) {
          batch_lengths[n]++;
        }
      }
    }

    void set(IntList input_sizes, IntList batch_sizes_, bool batch_first) {
      batch_sizes = batch_sizes_;
      set_batch_lengths(batch_sizes_);
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
    DropoutDescriptorParams dropout;
    RNNDescriptorParams rnn;
    TensorDescriptorListParams tensors;
  };

  // NB: unpacked input size
  std::vector<int64_t> _input_size(const TensorDescriptorListParams& tensors) {
    return {tensors.seq_length, tensors.mini_batch, tensors.input_size};
  }

  std::vector<int64_t> _hidden_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {rnn.num_layers * rnn.num_directions(), tensors.mini_batch, rnn.hidden_size};
  }

  // NB: unpacked output size
  std::vector<int64_t> _output_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    return {tensors.seq_length, tensors.mini_batch, rnn.hidden_size * rnn.num_directions()};
  }

  // NB: for LSTM, storage contains ig, fg, cg, og
  // for GRU, storage contains rg, ig, ng, hx, hn
  std::vector<int64_t> _storage_size(const RNNDescriptorParams& rnn, const TensorDescriptorListParams& tensors) {
    int64_t factor = (rnn.mode == MKLDNN_LSTM) ? 4 : 5;
    return {rnn.num_layers * rnn.num_directions(), tensors.seq_length, tensors.mini_batch, factor * rnn.hidden_size};
  }

  // NB: LSTM needs to reserve storage, immediate layer outputs, cy, dropout state
  // GRU needs to reserve storage, immediate layer outputs, dropout state
  int64_t get_num_reserve(const RNNParams& fn) {
    int64_t factor = (fn.rnn.mode == MKLDNN_LSTM) ? 4 : 5;
    int64_t num_reserve = fn.rnn.num_layers * fn.rnn.num_directions() * fn.tensors.seq_length * fn.tensors.mini_batch * factor * fn.rnn.hidden_size;
    num_reserve += (fn.rnn.num_layers-1) * fn.tensors.seq_length * fn.tensors.mini_batch * fn.rnn.hidden_size * fn.rnn.num_directions();
    num_reserve += fn.rnn.num_layers * fn.rnn.num_directions() * fn.tensors.seq_length * fn.tensors.mini_batch * fn.rnn.hidden_size;
    if (fn.dropout.dropout > 0) {
      num_reserve += (fn.rnn.num_layers-1) * fn.tensors.seq_length * fn.tensors.mini_batch * fn.rnn.hidden_size * fn.rnn.num_directions();
    }

    return num_reserve;
  }

  std::vector<Tensor> get_reserve_tensor(const RNNParams& fn, const Tensor& reserve) {
    std::vector<Tensor> tensor_arr;
    std::vector<std::vector<int64_t>> size_arr;

    auto storage_sz = _storage_size(fn.rnn, fn.tensors);
    auto output_sz = _output_size(fn.rnn, fn.tensors);
    auto cy_sz = std::vector<int64_t>{fn.rnn.num_layers * fn.rnn.num_directions(), fn.tensors.seq_length, fn.tensors.mini_batch, fn.rnn.hidden_size};
    output_sz.insert(output_sz.begin(), fn.rnn.num_layers-1);
    size_arr.push_back(storage_sz);
    size_arr.push_back(output_sz);
    size_arr.push_back((fn.rnn.mode == MKLDNN_LSTM) ? cy_sz : std::vector<int64_t>{0});
    // NB: dropout state has same size as immediate layer oupputs
    size_arr.push_back((fn.dropout.dropout > 0) ? output_sz : std::vector<int64_t>{0});

    int64_t offset = 0;
    for (auto& size: size_arr) {
      int64_t sz = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int64_t>());
      Tensor ts = (sz == 0) ? reserve.type().tensor() : reserve.type().tensor().set_(*reserve.storage(), offset, sz).view(size);
      tensor_arr.emplace_back(std::move(ts));
      offset += sz;
    }

    return tensor_arr;
  }

  Tensor _unpack_sequence(const TensorDescriptorListParams& tensors, const Tensor& packed) {
    auto batch_sizes = tensors.batch_sizes;
    auto unpacked_size = std::vector<int64_t>{tensors.seq_length, tensors.mini_batch, packed.size(1)};
    auto unpacked = packed.type().tensor(unpacked_size).zero_();

    int64_t offset = 0;
    for (int64_t t = 0; t < batch_sizes.size(); t++) {
      auto batch_size_step = batch_sizes[t];
      auto tensor_from = packed.narrow(0, offset, batch_size_step);
      auto tensor_to = unpacked[t].narrow(0, 0, batch_size_step);
      tensor_to.copy_(tensor_from.view_as(tensor_to));
      offset += batch_size_step;
    }

    return unpacked;
  }

  Tensor _pack_sequence(const TensorDescriptorListParams& tensors, const Tensor& unpacked) {
    auto batch_sizes = tensors.batch_sizes;
    auto packed_size = std::vector<int64_t>{tensors.batch_sizes_sum, unpacked.size(2)};
    auto packed = unpacked.type().tensor(packed_size).zero_();

    int64_t offset = 0;
    for (int64_t t = 0; t < batch_sizes.size(); t++) {
      auto batch_size_step = batch_sizes[t];
      auto tensor_from = unpacked[t].narrow(0, 0, batch_size_step);
      auto tensor_to = packed.narrow(0, offset, batch_size_step);
      tensor_to.copy_(tensor_from.view_as(tensor_to));
      offset += batch_size_step;
    }

    return packed;
  }

  void _get_packed_states(Tensor& tensor_to, const Tensor& tensor_from, const IntList& batch_lengths, int64_t t) {
    if (!tensor_from.defined()) {
      return;
    }
    if (!tensor_to.sizes().equals(tensor_from.sizes())) {
      throw std::runtime_error("_get_packed_states tensor size mismatch");
    }
    for (int64_t n = 0; n < tensor_from.size(0); n++) {
      if (batch_lengths[n] - 1 == t) {
        tensor_to[n].copy_(tensor_from[n]);
      }
    }
  }

  void _make_noise(Tensor& noise, double p) {
    if (p == 1) {
      noise.zero_();
    } else {
      noise.bernoulli_(1-p).div_(1-p);
    }
  }

  template<typename DType>
  inline DType sigm(DType x) { return 1.0f / (1.0f + std::exp(-x)); }

  template<typename DType>
  void GRUCell_forward(const Tensor& xw1, const Tensor& hw2,
    const Tensor& b1, const Tensor& b2, const Tensor& hx, Tensor& hy,
    Tensor& storage, bool train)
  {
    auto has_bias = b1.defined();

    auto hsz = hx.size(1);
    auto count = hy.numel();

    auto *xw1_p = xw1.data<DType>();
    auto *hw2_p = hw2.data<DType>();
    auto *b1_p = has_bias ? b1.data<DType>() : NULL;
    auto *b2_p = has_bias ? b2.data<DType>() : NULL;
    auto *hx_p = hx.data<DType>();
    auto *hy_p = hy.data<DType>();
    auto *storage_p = train ? storage.data<DType>() : NULL;

    #pragma omp parallel for
    for (int64_t index = 0; index < count; index++) {
      int64_t offset = (index/hsz)*3*hsz+index%hsz;
      int64_t offset_s = (index/hsz)*5*hsz+index%hsz;

      DType ir = xw1_p[offset+0*hsz];
      DType ii = xw1_p[offset+1*hsz];
      DType in = xw1_p[offset+2*hsz];
      DType hr = hw2_p[offset+0*hsz];
      DType hi = hw2_p[offset+1*hsz];
      DType hn = hw2_p[offset+2*hsz];

      DType hx_ = hx_p[index];
      DType *hy_ = &hy_p[index];

      DType b1r, b1i, b1n, b2r, b2i, b2n;
      b1r = has_bias ? b1_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b1i = has_bias ? b1_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b1n = has_bias ? b1_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b2r = has_bias ? b2_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b2i = has_bias ? b2_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b2n = has_bias ? b2_p[index%hsz+2*hsz] : static_cast<DType>(0);

      DType rg, ig, ng;
      rg = ir + hr + b1r + b2r;
      ig = ii + hi + b1i + b2i;
      rg = sigm<DType>(rg);
      ig = sigm<DType>(ig);
      ng = in + b1n + rg*(hn+b2n);
      ng = std::tanh(ng);

      *hy_ = ng + ig * (hx_-ng);

      if (train) {
        // NB: save for backwards
        storage_p[offset_s+0*hsz] = rg;
        storage_p[offset_s+1*hsz] = ig;
        storage_p[offset_s+2*hsz] = ng;
        storage_p[offset_s+3*hsz] = hx_;
        storage_p[offset_s+4*hsz] = hn + b2n;
      }
    }
  }

  template<typename DType>
  void GRUCell_backward(Tensor& grad_xw1, Tensor& grad_hw2,
    const Tensor& grad_y, Tensor& grad_hx, const Tensor& storage)
  {
    auto hsz = grad_y.size(1);
    auto count = grad_hx.numel();

    auto *grad_xw1_p = grad_xw1.data<DType>();
    auto *grad_hw2_p = grad_hw2.data<DType>();
    auto *grad_y_p = grad_y.data<DType>();
    auto *grad_hx_p = grad_hx.data<DType>();
    auto *storage_p = storage.data<DType>();

    #pragma omp parallel for
    for (int64_t index = 0; index < count; index++) {
      int64_t offset = (index/hsz)*3*hsz+index%hsz;
      int64_t offset_s = (index/hsz)*5*hsz+index%hsz;

      DType rg = storage_p[offset_s+0*hsz];
      DType ig = storage_p[offset_s+1*hsz];
      DType ng = storage_p[offset_s+2*hsz];
      DType hx = storage_p[offset_s+3*hsz];
      DType hn = storage_p[offset_s+4*hsz];

      DType go = grad_y_p[index];

      DType gig = go*(hx-ng)*(1-ig)*ig;
      DType ghx = go*ig;
      DType gin = go*(1-ig)*(1-ng*ng);
      DType ghn = gin*rg;
      DType grg = gin*hn*(1-rg)*rg;

      grad_xw1_p[offset+0*hsz] = grg;
      grad_xw1_p[offset+1*hsz] = gig;
      grad_xw1_p[offset+2*hsz] = gin;

      grad_hw2_p[offset+0*hsz] = grg;
      grad_hw2_p[offset+1*hsz] = gig;
      grad_hw2_p[offset+2*hsz] = ghn;
      grad_hx_p[index] = ghx;
    }
  }

  template<typename DType>
  void LSTMCell_forward(const Tensor& xw1, const Tensor& hw2,
    const Tensor& b1, const Tensor& b2, const Tensor& cx, Tensor& hy, Tensor& cy,
    Tensor& storage, bool train)
  {
    auto has_bias = b1.defined();

    auto hsz = cx.size(1);
    auto count = hy.numel();

    auto *xw1_p = xw1.data<DType>();
    auto *hw2_p = hw2.data<DType>();
    auto *b1_p = has_bias ? b1.data<DType>() : NULL;
    auto *b2_p = has_bias ? b2.data<DType>() : NULL;
    auto *cx_p = cx.data<DType>();
    auto *hy_p = hy.data<DType>();
    auto *cy_p = cy.data<DType>();
    auto *storage_p = train ? storage.data<DType>() : NULL;

    #pragma omp parallel for
    for (int64_t index = 0; index < count; index++) {
      int64_t offset = (index/hsz)*4*hsz+index%hsz;

      DType iig = xw1_p[offset+0*hsz];
      DType ifg = xw1_p[offset+1*hsz];
      DType icg = xw1_p[offset+2*hsz];
      DType iog = xw1_p[offset+3*hsz];

      DType hig = hw2_p[offset+0*hsz];
      DType hfg = hw2_p[offset+1*hsz];
      DType hcg = hw2_p[offset+2*hsz];
      DType hog = hw2_p[offset+3*hsz];

      DType cx_ = cx_p[index];
      DType *hy_ = &hy_p[index];
      DType *cy_ = &cy_p[index];

      DType b1i, b1f, b1c, b1o, b2i, b2f, b2c, b2o;
      b1i = has_bias ? b1_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b1f = has_bias ? b1_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b1c = has_bias ? b1_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b1o = has_bias ? b1_p[index%hsz+3*hsz] : static_cast<DType>(0);
      b2i = has_bias ? b2_p[index%hsz+0*hsz] : static_cast<DType>(0);
      b2f = has_bias ? b2_p[index%hsz+1*hsz] : static_cast<DType>(0);
      b2c = has_bias ? b2_p[index%hsz+2*hsz] : static_cast<DType>(0);
      b2o = has_bias ? b2_p[index%hsz+3*hsz] : static_cast<DType>(0);

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
        // NB: save for backwards
        storage_p[offset+0*hsz] = ig;
        storage_p[offset+1*hsz] = fg;
        storage_p[offset+2*hsz] = cg;
        storage_p[offset+3*hsz] = og;
      }
    }
  }

  template <typename DType>
  void LSTMCell_backward(Tensor& grad_xw1, const Tensor& cx, const Tensor& cy,
    const Tensor& grad_y, const Tensor& grad_cy, Tensor& grad_cx, const Tensor& storage)
  {
    auto hsz = grad_y.size(1);
    auto count = grad_cx.numel();

    auto storage_p = storage.data<DType>();
    auto grad_xw1_p = grad_xw1.data<DType>();
    auto cx_p = cx.data<DType>();
    auto cy_p = cy.data<DType>();
    auto grad_y_p = grad_y.data<DType>();
    auto grad_cy_p = grad_cy.data<DType>();
    auto grad_cx_p = grad_cx.data<DType>();

    #pragma omp parallel for
    for (int64_t index = 0; index < count; index++) {
      size_t offset = (index/hsz)*4*hsz+index%hsz;

      DType ig = storage_p[offset+0*hsz];
      DType fg = storage_p[offset+1*hsz];
      DType cg = storage_p[offset+2*hsz];
      DType og = storage_p[offset+3*hsz];

      DType *ih = &grad_xw1_p[offset+0*hsz];
      DType *fh = &grad_xw1_p[offset+1*hsz];
      DType *ch = &grad_xw1_p[offset+2*hsz];
      DType *oh = &grad_xw1_p[offset+3*hsz];

      DType cx_ = cx_p[index];
      DType cy_ = cy_p[index];

      DType *gi = &grad_cx_p[index];

      DType go = grad_y_p[index];
      DType goc = grad_cy_p[index];

      DType gcx = std::tanh(cy_);

      DType gog = go * gcx;
      gcx = go * og * (1 - gcx*gcx) + goc;

      DType gig = gcx * cg;
      DType gfg = gcx * cx_;
      DType gcg = gcx * ig;

      gcx = gcx * fg;

      gig = gig * (1-ig) * ig;
      gfg = gfg * (1-fg) * fg;
      gcg = gcg * (1-cg*cg);
      gog = gog * (1-og) * og;

      *ih = gig;
      *fh = gfg;
      *ch = gcg;
      *oh = gog;

      *gi = gcx;
    }
  }

  Tensor _rnn_layer_forward(const RNNParams& fn, const Tensor& x, TensorList weights,
    const Tensor& hx, const Tensor& cx, Tensor& hy, Tensor& cy, Tensor& storage, Tensor& res_cy,
    bool train, bool reverse)
  {
    auto has_bias = (weights.size() == 4);
    auto is_input_packed = fn.tensors.is_input_packed();
    auto batch_lengths = fn.tensors.batch_lengths;
    auto mode = fn.rnn.mode;

    auto w1 = weights[0];
    auto w2 = weights[1];
    auto b1 = has_bias ? weights[2] : Tensor();
    auto b2 = has_bias ? weights[3] : Tensor();

    auto seq_length = x.size(0);
    auto mini_batch = x.size(1);
    auto input_size = x.size(2);
    auto gate_size = w2.size(0);
    auto hidden_size = w2.size(1);

    // NB: output per layer per direction
    auto y = x.type().tensor({seq_length, mini_batch, hidden_size});
    // NB: fuse x*w from each timestep
    auto xw1 = x.view({-1, input_size}).matmul(w1.t()).view({seq_length, mini_batch, gate_size});

    auto hx_t = hx;
    auto cx_t = cx;
    for (int64_t t = 0; t < seq_length; t++) {
      int64_t ts = reverse ? (seq_length-t-1) : t;
      // NB: Suppose time sequences are (0 refers to padding)
      //   Sequence 1: A B C D
      //   Sequence 2: E F 0 0
      //   Sequence 3: G 0 0 0
      // for packed input
      //   forward direction of bidirectional RNN, hx/cx is AEG and hy/cy will be DFG
      //   backward direction of bidirectional RNN, hx/cx is DFG and hy/cy will be AEG
      // for padded input
      //   forward direction of bidirectional RNN, hx/cx is AEG and hy/cy will be D00
      //   backward direction of bidirectional RNN, hx/cx is D00 and hy/cy will be AEG
      if (is_input_packed and reverse) {
        _get_packed_states(hx_t, hx, batch_lengths, ts);
        _get_packed_states(cx_t, cx, batch_lengths, ts);
      }
      auto xw1_t = xw1[ts];
      auto hw2_t = hx_t.matmul(w2.t());
      AT_ASSERTM(xw1_t.sizes().equals(hw2_t.sizes()),
        "input weight product and hidden weight product mismatch")
      auto hy_t = y[ts];
      auto cy_t = cx.defined() ? cx.type().zeros_like(cx) : Tensor();
      auto storage_t = train ? storage[ts] : hx.type().tensor();
      auto res_cy_t = (train && cx.defined()) ? res_cy[ts] : Tensor();

      if (mode == MKLDNN_LSTM) {
        LSTMCell_forward<float>(xw1_t, hw2_t, b1, b2, cx_t, hy_t, cy_t, storage_t, train);
        if (train) {
          res_cy_t.copy_(cy_t);
        }
      } else if (mode == MKLDNN_GRU) {
        GRUCell_forward<float>(xw1_t, hw2_t, b1, b2, hx_t, hy_t, storage_t, train);
      } else {
        throw std::runtime_error("MKLDNN unsupported RNN mode");
      }

      if (is_input_packed and !reverse) {
        _get_packed_states(hy, hy_t, batch_lengths, ts);
        _get_packed_states(cy, cy_t, batch_lengths, ts);
      }
      // NB: update hidden/cell state for next time step
      hx_t = hy_t;
      cx_t = cy_t;
    }
    // NB: update hy/cy at the final time step
    if (!is_input_packed || (is_input_packed && reverse)) {
      hy.copy_(hx_t);
      if (cx.defined()) {
        cy.copy_(cx_t);
      }
    }

    return y;
  }

  void _rnn_layer_backward(const RNNParams& fn, const Tensor& x, TensorList weights,
    const Tensor& hx, const Tensor& cx, const Tensor& y, const Tensor& grad_y,
    const Tensor& grad_hy, const Tensor& grad_cy, const Tensor& storage, const Tensor& res_cy,
    Tensor& grad_x, Tensor& grad_hx, Tensor& grad_cx, TensorList grad_weights,
    bool reverse, std::array<bool, 4> output_mask)
  {
    auto has_bias = (weights.size() == 4);
    auto is_input_packed = fn.tensors.is_input_packed();
    auto batch_sizes = fn.tensors.batch_sizes;
    auto batch_lengths = fn.tensors.batch_lengths;
    auto mode = fn.rnn.mode;

    auto w1 = weights[0];
    auto w2 = weights[1];
    auto grad_w1 = grad_weights[0];
    auto grad_w2 = grad_weights[1];
    auto grad_b1 = has_bias ? grad_weights[2] : hx.type().tensor();
    auto grad_b2 = has_bias ? grad_weights[3] : hx.type().tensor();

    auto seq_length = x.size(0);
    auto mini_batch = x.size(1);
    auto input_size = x.size(2);
    auto gate_size = w2.size(0);
    auto hidden_size = w2.size(1);

    auto grad_xw1 = x.type().tensor({seq_length, mini_batch, gate_size});

    // NB: can't use grad_hy_t as a signature of grad_hy here,
    // as we mutates data of grad_hy_t
    auto grad_hy_t = grad_hy.clone();
    auto grad_cy_t = grad_cy;
    for (int64_t t = 0; t < seq_length; t++) {
      int64_t ts = reverse ? t : (seq_length-t-1);

      if (is_input_packed and !reverse) {
        _get_packed_states(grad_hy_t, grad_hy, batch_lengths, ts);
        _get_packed_states(grad_cy_t, grad_cy, batch_lengths, ts);
      }
      auto x_t = x[ts];
      auto hx_t = reverse ? ((ts == seq_length-1) ? hx : y[ts+1])
                          : ((ts == 0) ? hx : y[ts-1]);
      if (is_input_packed and reverse) {
        _get_packed_states(hx_t, hx, batch_lengths, ts);
      }
      grad_hy_t += grad_y[ts];
      auto storage_t = storage[ts];

      auto grad_xw1_t = grad_xw1[ts];
      auto grad_hw2_t = hx.type().tensor({mini_batch, gate_size});
      auto grad_hx_t = hx.type().zeros_like(hx);

      Tensor cx_t, cy_t, grad_cx_t;
      if (mode == MKLDNN_LSTM) {
        cx_t = reverse ? ((ts == seq_length-1) ? cx : res_cy[ts+1])
                       : ((ts == 0) ? cx : res_cy[ts-1]);
        cy_t = res_cy[ts];
        if (is_input_packed and reverse) {
          _get_packed_states(cx_t, cx, batch_lengths, ts);
        }
        grad_cx_t = cx.type().tensor({mini_batch, hidden_size});
        LSTMCell_backward<float>(grad_xw1_t, cx_t, cy_t, grad_hy_t, grad_cy_t, grad_cx_t, storage_t);
        grad_hw2_t = grad_xw1_t;
      }
      else if (mode == MKLDNN_GRU) {
        GRUCell_backward<float>(grad_xw1_t, grad_hw2_t, grad_hy_t, grad_hx_t, storage_t);
      } else {
        throw std::runtime_error("MKLDNN unsupported RNN mode");
      }

      if (is_input_packed) {
        auto bs = batch_sizes[ts];
        if (bs < mini_batch) {
          grad_xw1_t.narrow(0, bs, mini_batch - bs).zero_();
          grad_hw2_t.narrow(0, bs, mini_batch - bs).zero_();
        }
      }
      // NB: grad_x from bidirectional direction should be accumulated
      grad_hx_t.addmm_(grad_hw2_t, w2);
      if (output_mask[3]) {
        grad_w1.addmm_(grad_xw1_t.t(), x_t);
        grad_w2.addmm_(grad_hw2_t.t(), hx_t);
        grad_b1 += grad_xw1_t.sum(0);
        grad_b2 += grad_hw2_t.sum(0);
      }

      if (is_input_packed and reverse) {
        _get_packed_states(grad_hx, grad_hx_t, batch_lengths, ts);
        _get_packed_states(grad_cx, grad_cx_t, batch_lengths, ts);
      }
      // NB: update grad_hy/grad_cy for next time step
      grad_hy_t = grad_hx_t;
      grad_cy_t = grad_cx_t;
    }

    // NB: fuse grad_xw1*w1 from each time step
    auto grad_x_view = grad_x.view({-1, input_size});
    grad_x_view.addmm_(grad_xw1.view({-1, gate_size}), w1);

    // NB: update grad_hx at the final time step
    if (!is_input_packed || (is_input_packed && !reverse)) {
      grad_hx.copy_(grad_hy_t);
      if (grad_cy.defined()) {
        grad_cx.copy_(grad_cy_t);
      }
    }
  }
} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor, Tensor> _mkldnn_rnn(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes)
{
  auto input = input_r;

  RNNParams fn;
  fn.dropout.set(fn_train, fn_dropout);
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
  if (fn_dropout < 0 || fn_dropout > 1) {
    throw std::runtime_error("rnn: dropout probability has to be between 0 and 1");
  }

  auto x = input.contiguous();
  if (is_input_packed) {
    x = _unpack_sequence(fn.tensors, x);
  }
  auto output = input.type().tensor(output_size);
  auto y = output;
  auto hy = hx.type().tensor(hidden_size);
  // NB: Not allowed to return undefined tensors
  auto cy = cx.defined() ? cx.type().tensor(hidden_size) : hx.type().tensor();

  Tensor reserve, res_storage, res_y, res_cy, res_dropout;
  if (fn_train) {
    auto num_reserve = get_num_reserve(fn);
    reserve = hx.type().tensor(num_reserve).zero_();
    auto res_arr = get_reserve_tensor(fn, reserve);
    res_storage = res_arr[0];
    res_y = res_arr[1];
    res_cy = res_arr[2];
    res_dropout = res_arr[3];
  }

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions();

  auto layer_x = x;
  for (int64_t layer = 0; layer < num_layers; layer++) {
    std::vector<Tensor> layer_y(num_directions);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;

      auto layer_weights = weights[index];
      auto layer_hx = hx[index];
      auto layer_hy = hy[index];
      auto layer_cx = cx.defined() ? cx[index] : Tensor();
      auto layer_cy = cx.defined() ? cy[index] : hx.type().tensor();
      auto layer_storage = fn_train ? res_storage[index] : hx.type().tensor();
      auto layer_res_cy = (fn_train && cx.defined()) ? res_cy[index] : hx.type().tensor();

      auto reverse = (direction > 0);
      layer_y[direction] = _rnn_layer_forward(fn, layer_x, layer_weights,
         layer_hx, layer_cx, layer_hy, layer_cy, layer_storage, layer_res_cy, fn_train, reverse);
    }
    layer_x = at::cat(layer_y, x.dim() - 1);
    // NB: save immediate layer outputs for backward
    if (fn_train && (layer != num_layers - 1)) {
      res_y[layer].copy_(layer_x);
    }
    if (fn_train && (layer != num_layers - 1) && (fn_dropout > 0)) {
      auto noise = res_dropout[layer];
      _make_noise(noise, fn_dropout);
      layer_x.mul_(noise);
    }
  }

  y = layer_x;
  if (is_input_packed) {
    y = _pack_sequence(fn.tensors, y);
  }
  if (batch_first && !is_input_packed) {
    y.transpose_(0, 1);
  }

  return std::make_tuple(y, hy, cy, reserve);
}

std::tuple<Tensor, Tensor, Tensor, std::vector<Tensor>> _mkldnn_rnn_backward(
    const Tensor& input_r,
    TensorList weight, int64_t weight_stride0,
    const Tensor& hx, const Tensor& cx,
    const Tensor& output_r, const Tensor& grad_output_r,
    const Tensor& grad_hy_r, const Tensor& grad_cy_r,
    int64_t fn_mode, int64_t fn_hidden_size,
    int64_t fn_num_layers, bool batch_first, double fn_dropout,
    bool fn_train, bool fn_bidirectional, IntList fn_batch_sizes,
    const Tensor& reserve, std::array<bool, 4> output_mask)
{
  auto input = input_r;
  auto output = output_r;
  auto grad_output = grad_output_r.defined() ? grad_output_r : output.type().zeros_like(output);
  auto grad_hy = grad_hy_r.defined() ? grad_hy_r : hx.type().zeros_like(hx);
  auto grad_cy = cx.defined() ? (grad_cy_r.defined() ? grad_cy_r : cx.type().zeros_like(cx)) : grad_cy_r;

  RNNParams fn;
  fn.dropout.set(fn_train, fn_dropout);
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
    grad_output = grad_output.transpose(0, 1);
    output = output.transpose(0, 1);
  }

  auto input_size = _input_size(fn.tensors);
  auto hidden_size = _hidden_size(fn.rnn, fn.tensors);
  auto output_size = _output_size(fn.rnn, fn.tensors);

  if (!hx.is_contiguous()) {
    throw std::runtime_error("rnn: hx is not contiguous");
  }
  if (cx.defined() && !cx.is_contiguous()) {
    throw std::runtime_error("rnn: cx is not contiguous");
  }
  if (!cx.defined() && output_mask[2]) {
    throw std::runtime_error("illegally required grad of cx for non-LSTM RNN");
  }
  if (!fn_train) {
    throw std::runtime_error("backward_input can only be called in training mode");
  }
  if (fn_dropout < 0 || fn_dropout > 1) {
    throw std::runtime_error("rnn: dropout probability has to be between 0 and 1");
  }

  auto x = input.contiguous();
  auto y = output;
  auto grad_y = grad_output.contiguous();
  if (is_input_packed) {
    x = _unpack_sequence(fn.tensors, x);
    y = _unpack_sequence(fn.tensors, y);
    grad_y = _unpack_sequence(fn.tensors, grad_y);
  }
  grad_hy = grad_hy.contiguous().view(hidden_size);
  grad_cy =  grad_cy.defined() ? grad_cy.contiguous().view(hidden_size) : Tensor();
  auto grad_x = input.type().tensor(input_size);
  auto grad_hx = hx.type().tensor(hidden_size);
  auto grad_cx = cx.defined() ? cx.type().tensor(hidden_size) : Tensor();

  Tensor res_storage, res_y, res_cy, res_dropout;
  if (fn_train) {
    auto res_arr = get_reserve_tensor(fn, reserve);
    res_storage = res_arr[0];
    res_y = res_arr[1];
    res_cy = res_arr[2];
    res_dropout = res_arr[3];
  }

  MatrixRef<Tensor> weights{weight, static_cast<size_t>(weight_stride0)};

  std::vector<Tensor> grad_weight_arr;
  grad_weight_arr.reserve(weights.numel());
  for (const auto& w: weight) {
    grad_weight_arr.emplace_back(w.type().zeros_like(w));
  }
  MatrixRef<Tensor> grad_weights{grad_weight_arr, static_cast<size_t>(weight_stride0)};

  auto num_layers = fn.rnn.num_layers;
  auto num_directions = fn.rnn.num_directions();

  auto layer_grad_output = grad_y;
  for (int64_t layer = num_layers-1; layer >= 0; layer--) {
    if ((layer != num_layers-1) && (fn_dropout > 0)) {
      layer_grad_output.mul_(res_dropout[layer]);
    }
    auto layer_x = (layer == 0) ? x : res_y[layer-1];
    auto layer_grad_x = layer_x.type().zeros_like(layer_x);
    auto layer_output = (layer == num_layers-1) ? y : res_y[layer];
    std::vector<Tensor> layer_ys = layer_output.chunk(num_directions, layer_output.dim()-1);
    std::vector<Tensor> layer_grad_ys = layer_grad_output.chunk(num_directions, layer_grad_output.dim()-1);
    for (int64_t direction = 0; direction < num_directions; direction++) {
      auto index = layer * num_directions + direction;

      auto layer_weights = weights[index];
      auto layer_grad_weights = grad_weights[index];
      auto layer_hx = hx[index];
      auto layer_cx = cx.defined() ? cx[index] : Tensor();
      auto layer_grad_hy = grad_hy[index];
      auto layer_grad_cy = grad_cy.defined() ? grad_cy[index] : Tensor();
      auto layer_grad_hx = grad_hx[index];
      auto layer_grad_cx = cx.defined() ? grad_cx[index] : hx.type().tensor();
      auto layer_y = layer_ys[direction].contiguous();
      auto layer_grad_y = layer_grad_ys[direction].contiguous();
      auto layer_storage = res_storage[index];
      auto layer_res_cy = (cx.defined()) ? res_cy[index] : hx.type().tensor();

      auto reverse = (direction > 0);
      _rnn_layer_backward(fn, layer_x, layer_weights, layer_hx, layer_cx, layer_y, layer_grad_y, layer_grad_hy, layer_grad_cy, layer_storage, layer_res_cy,
        layer_grad_x, layer_grad_hx, layer_grad_cx, layer_grad_weights, reverse, output_mask);
    }
    // NB: update grad_output for next layer
    layer_grad_output = layer_grad_x;
  }

  grad_x = layer_grad_output;
  if (is_input_packed) {
    grad_x = _pack_sequence(fn.tensors, grad_x);
  }
  if (batch_first && !is_input_packed) {
    grad_x = grad_x.transpose_(0, 1);
  }

  return std::make_tuple(grad_x, grad_hx, grad_cx, grad_weight_arr);
}

}} // namespace at::native

#endif // AT_MKLDNN_ENABLED()
