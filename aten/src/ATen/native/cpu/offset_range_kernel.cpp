#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/cpu/offset_range_kernel.h>

namespace at { namespace native {

namespace {


template <typename scalar_t>
static inline void arange_vec(scalar_t* out, int64_t size) {
  using Vec = vec256::Vec256<scalar_t>;

  Vec out_vec = Vec::arange(scalar_t(0), scalar_t(1));
  Vec step_vec = Vec(scalar_t(Vec::size()));

  int64_t d = 0;
  for (; d < size - (size % Vec::size()); d += Vec::size()) {
    out_vec.store(out + d);
    out_vec += step_vec;
  }
  if (size - d > 0) {
    out_vec.store(out + d, size - d);
  }
}

template <typename scalar_t>
void cpu_offset_range(Tensor& output, const Tensor& input) {
  scalar_t* input_data = input.data_ptr<scalar_t>();

  int64_t numel = input.numel();
  int64_t grain_size = 512;

  int64_t num_threads = at::get_num_threads();
  int64_t size_per_thread[num_threads];
  int64_t offset_per_thread[num_threads];
  std::fill_n(&size_per_thread[0], num_threads, int64_t(0));

  // Path I: accumulate num of elements per thread
  //
  // [note]: from performance aspect the expectation is 'outpu_size >> input_size'
  //   so Path I won't be the bottleneck and I didn't vectorize it.
  //
  //   If output_size ~= input_size but still the numel is big,
  //   Path I also need to be vectorized.
  at::parallel_for(/*begin*/1, /*end*/numel, grain_size, [&](int64_t begin, int64_t end) {
    int64_t tid = at::get_thread_num();

    int64_t sum = 0;
    for (int64_t i = begin; i < end; i++) {
      sum += input_data[i] - input_data[i - 1];
    }
    size_per_thread[tid] = sum;
  });

  // caculate the total num of elements (output size)
  // and offset index per thread
  int64_t output_size{0}, offset{0};
  for (int64_t t = 0; t < num_threads; t++) {
    output_size += size_per_thread[t];
    offset_per_thread[t] = offset;
    offset += size_per_thread[t];
  }

  TORCH_CHECK(output_size == int64_t(input_data[numel - 1] - input_data[0]),
      "expect the input sequence is ascending!")

  // allocate output buffer
  output.resize_({output_size});
  scalar_t* output_data = output.data_ptr<scalar_t>();

  // Path II: update with 'arange' for each segment
  at::parallel_for(/*begin*/1, /*end*/numel, grain_size, [&](int64_t begin, int64_t end) {
    int64_t tid = at::get_thread_num();

    // local pointer
    scalar_t* output_ptr = output_data + offset_per_thread[tid];

    // local output indexing
    int64_t local_offset = 0;
    for (int64_t i = begin; i < end; i++) {
      int64_t size = input_data[i] - input_data[i - 1];
      arange_vec(output_ptr + local_offset, size);
      local_offset += size;
    }
  });
}

void offset_range_kernel_impl(Tensor& output, const Tensor& input) {
  if (input.scalar_type() == ScalarType::Long) {
    cpu_offset_range<int64_t>(output, input);
  } else if (input.scalar_type() == ScalarType::Int) {
    cpu_offset_range<int32_t>(output, input);
  } else {
    TORCH_CHECK(false, "expect int32_t or int64_t input");
  }
}

} // anonymous namespace

REGISTER_DISPATCH(offset_range_kernel, &offset_range_kernel_impl);

}} // at::native
