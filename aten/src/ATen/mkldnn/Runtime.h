#pragma once

#include <mkldnn.hpp>

using namespace mkldnn;

namespace at { namespace native {

// MKLDNNEngine singleton
struct MKLDNNEngine {
  static MKLDNNEngine& Instance() {
    static MKLDNNEngine myInstance;
    return myInstance;
  }
  engine& get_engine() {
    return _engine;
  }
  MKLDNNEngine(MKLDNNEngine const&) = delete;
  MKLDNNEngine& operator=(MKLDNNEngine const&) = delete;

protected:
  MKLDNNEngine():_engine(engine::cpu, 0) {}
  ~MKLDNNEngine() {}

private:
  engine _engine;
};

// MKLDNNStream singleton
struct MKLDNNStream {
  static MKLDNNStream& Instance() {
    static thread_local MKLDNNStream myInstance;
    return myInstance;
  };

  void commit(const primitive& _primitive) {
    net.push_back(_primitive);
  }

  void submit() {
    if (!net.empty()) {
      stream(stream::kind::eager).submit(net).wait();
      net.clear();
    }
  }

private:
  std::vector<primitive> net;
};

#define MKLDNN_EXEC(_primitive)                                   \
  std::vector<mkldnn::primitive> net;                             \
  net.push_back(_primitive);                                      \
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait(); \

template<typename prim_t>
struct MKLDNNPrimitive {
  std::shared_ptr<prim_t> _prim;
  MKLDNNPrimitive() : _prim(nullptr) {}
  const prim_t& get_primitive() const { return *_prim; }
};

}}  // namespace at::native
