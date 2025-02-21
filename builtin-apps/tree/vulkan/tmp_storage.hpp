#pragma once

#include "../../base_appdata.hpp"

namespace tree::vulkan {

struct TmpStorage : BaseAppData {
  TmpStorage(std::pmr::memory_resource* mr, const size_t n_input)
      : BaseAppData(mr),
        u_contributes(n_input, mr),
        u_out_idx(n_input, mr),
        u_sums(n_input, mr),
        u_prefix_sums(n_input, mr) {}

  // for remove duplicates
  UsmVector<uint32_t> u_contributes;
  UsmVector<uint32_t> u_out_idx;

  // for prefix sum
  UsmVector<uint32_t> u_sums;
  UsmVector<uint32_t> u_prefix_sums;
};

}  // namespace tree::vulkan
