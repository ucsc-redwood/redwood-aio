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
      brt(n_input, mr),
      oct(n_input * k_memory_ratio, mr) {
  constexpr auto seed = 114514;

  std::mt19937 gen(seed);
  std::uniform_real_distribution dis(min_coord, min_coord + range);

  std::ranges::generate(u_input_points, [&]() {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

AppData::RadixTree::RadixTree(const size_t n_nodes,
                              std::pmr::memory_resource* mr)
    : u_prefix_n(n_nodes, mr),
      u_has_leaf_left(n_nodes, mr),
      u_has_leaf_right(n_nodes, mr),
      u_left_child(n_nodes, mr),
      u_parents(n_nodes, mr) {}

AppData::Octree::Octree(const size_t n_nodes, std::pmr::memory_resource* mr)
    : u_children(n_nodes * 8, mr),
      u_corner(n_nodes, mr),
      u_cell_size(n_nodes, mr),
      u_child_node_mask(n_nodes, mr),
      u_child_leaf_mask(n_nodes, mr) {}

}  // namespace tree
