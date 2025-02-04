#include "tree_appdata.hpp"

#include <algorithm>
#include <random>

namespace tree {

AppData::AppData(std::pmr::memory_resource* mr, const size_t n_input)
    : BaseAppData(mr),
      n_input(n_input),
      u_input_points(n_input, mr),
      u_morton_keys(n_input, mr),
      u_morton_keys_alt(n_input, mr),
      u_edge_count(n_input, mr),
      u_edge_offset(n_input, mr),
      u_brt_prefix_n(n_input, mr),
      u_brt_has_leaf_left(n_input, mr),
      u_brt_has_leaf_right(n_input, mr),
      u_brt_left_child(n_input, mr),
      u_brt_parents(n_input, mr),
      u_oct_children(n_input * 8 * kMemoryRatio, mr),
      u_oct_corner(n_input * kMemoryRatio, mr),
      u_oct_cell_size(n_input * kMemoryRatio, mr),
      u_oct_child_node_mask(n_input * kMemoryRatio, mr),
      u_oct_child_leaf_mask(n_input * kMemoryRatio, mr) {
  std::mt19937 gen(114514);
  std::uniform_real_distribution dis(kMinCoord, kMinCoord + kRange);

  // generate input points
  std::ranges::generate(u_input_points, [&]() {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

}  // namespace tree
