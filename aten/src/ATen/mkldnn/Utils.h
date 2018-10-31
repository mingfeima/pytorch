#pragma once

#include "ATen/native/utils/ParamsHash.h"

namespace at { namespace native {

template <typename key_t, typename handle_t>
struct PrimitiveCache {
  std::unordered_map<key_t, handle_t, ParamsHash<key_t>, ParamsEqual<key_t>> map;

  bool find(const key_t& params, handle_t& results) {
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    results = it->second;
    return true;
  }

  void insert(const key_t& params, const handle_t& results) {
    map.insert(std::pair<key_t, handle_t>(params, results));
  }
};

}}  // namespace at::native
