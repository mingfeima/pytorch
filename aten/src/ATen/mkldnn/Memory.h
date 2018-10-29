#pragma once

#include <ATen/ATen.h>

#include <mkldnn.hpp>
#include "Runtime.h"

using namespace mkldnn;

namespace at { namespace native {

inline memory::desc _format_md(const memory::dims& _dims, const memory::format& _format) {
  return memory::desc({_dims}, memory::data_type::f32, _format);
}

inline memory::desc _generic_md(const memory::dims& _dims) {
  return _format_md(_dims, memory::format::any);
}

inline memory::primitive_desc _primitive_md(const memory::dims& _dims, const memory::format& _format) {
  auto _engine = MKLDNNEngine::Instance().get_engine(); 
  return memory::primitive_desc(_format_md(_dims, _format), _engine);
}

struct MKLDNNMemory {
  memory _memory;
  memory::primitive_desc _pdesc;

  MKLDNNMemory(const memory::primitive_desc& pd, const Tensor& tensor)
    : _memory(pd, tensor.data_ptr()), _pdesc(pd) {
  }

  // reorder from user layout to primitive layout, e.g. input, weight
  memory reorder_to(const memory::primitive_desc& pd) const {
    auto mem = _memory;
    if (pd != _pdesc) {
      mem = memory(pd);
      MKLDNN_EXEC(reorder(_memory, mem));
    }
    return mem;
  }

  // create mkldnn memory with primitive layout, e.g.output
  memory create(const memory::primitive_desc& pd) const {
    return (pd != _pdesc) ? memory(pd) : _memory;
  }

  // reorder from primitive layout to user layout, e.g. output
  void reorder_from(const memory& mem) {
    if (mem.get_primitive_desc() != _pdesc) {
      MKLDNN_EXEC(reorder(mem, _memory));
    }    
  }
};

}}  // namespace at::native
