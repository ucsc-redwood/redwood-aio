#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <stdexcept>

#include "../base_appdata.hpp"

namespace tree {

constexpr auto kMemoryRatio = 0.6f;
constexpr auto kDefaultInputSize = 640 * 480;
constexpr auto kMinCoord = 0.0f;
constexpr auto kRange = 1024.0f;

struct AppData final : BaseAppData {
  explicit AppData(std::pmr::memory_resource* mr,
                   const size_t n_input = kDefaultInputSize);

  ~AppData() override = default;

  // --------------------------------------------------------------------------
  // Essential data
  // --------------------------------------------------------------------------
  const uint32_t n_input;
  uint32_t n_unique = std::numeric_limits<uint32_t>::max();
  uint32_t n_brt_nodes = std::numeric_limits<uint32_t>::max();
  uint32_t n_octree_nodes = std::numeric_limits<uint32_t>::max();

  // n_input
  UsmVector<glm::vec4> u_input_points;
  UsmVector<uint32_t> u_morton_keys;
  UsmVector<uint32_t> u_morton_keys_alt;
  UsmVector<int32_t> u_edge_count;
  UsmVector<int32_t> u_edge_offset;

  UsmVector<uint8_t> u_brt_prefix_n;
  UsmVector<uint8_t> u_brt_has_leaf_left;
  UsmVector<uint8_t> u_brt_has_leaf_right;
  UsmVector<int32_t> u_brt_left_child;
  UsmVector<int32_t> u_brt_parents;

  // int (*u_children)[8]; note, this is 8x more
  UsmVector<int32_t> u_oct_children;

  // everything else is size of 'n_octree_nodes'
  // but for simplicity, we allocate the 0.6 times of input size
  // 60% memory is an empirical value
  UsmVector<glm::vec4> u_oct_corner;
  UsmVector<float> u_oct_cell_size;
  UsmVector<int32_t> u_oct_child_node_mask;
  UsmVector<int32_t> u_oct_child_leaf_mask;

  // --------------------------------------------------------------------------
  // Getters
  // --------------------------------------------------------------------------

  [[nodiscard]] uint32_t get_n_input() const { return n_input; }

  [[nodiscard]] uint32_t get_n_unique() const {
    if (n_unique == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("n_unique is not set");
    }

    return n_unique;
  }

  [[nodiscard]] uint32_t get_n_brt_nodes() const {
    return this->get_n_unique() - 1;
  }

  [[nodiscard]] uint32_t get_n_octree_nodes() const {
    if (n_octree_nodes == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("n_octree_nodes is not set");
    }

    return n_octree_nodes;
  }

  void set_n_unique(const uint32_t n_unique) { this->n_unique = n_unique; }

  void set_n_brt_nodes(const uint32_t n_brt_nodes) {
    this->n_brt_nodes = n_brt_nodes;
  }

  void set_n_octree_nodes(const uint32_t n_octree_nodes) {
    this->n_octree_nodes = n_octree_nodes;
  }
};

}  // namespace tree
