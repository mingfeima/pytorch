#pragma once

#include "ATen/native/utils/ParamsHash.h"

namespace at { namespace native {

template <typename key_t, typename prim_t>
struct PrimitiveCache {
  using prim_handle_t = std::shared_ptr<prim_t>;
  std::unordered_map<key_t, prim_handle_t, ParamsHash<key_t>, ParamsEqual<key_t>> map;

  bool find(const key_t& params, prim_handle_t& results) {
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    results = it->second;
    return true;
  }

  void insert(const key_t& params, const prim_handle_t& results) {
    map.insert(std::pair<key_t, prim_handle_t>(params, results));
  }
};

}}  // namespace at::native
