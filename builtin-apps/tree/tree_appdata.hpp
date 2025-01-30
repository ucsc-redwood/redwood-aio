#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <stdexcept>

#include "../base_appdata.hpp"

namespace tree {

constexpr auto k_memory_ratio = 0.6f;
constexpr auto kDefaultInputSize = 640 * 480;  // ~300k points

struct AppData final : BaseAppData {
  static constexpr auto min_coord = 0.0f;
  static constexpr auto range = 1024.0f;

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

  // should have size 'n_brt_nodes', but for simplicity, we allocate the same
  // size as the input buffer 'n_input'
  UsmVector<int32_t> u_edge_count;
  UsmVector<int32_t> u_edge_offset;

  struct RadixTree {
    explicit RadixTree(const size_t n_nodes, std::pmr::memory_resource* mr);

    UsmVector<uint8_t> u_prefix_n;
    UsmVector<uint8_t> u_has_leaf_left;
    UsmVector<uint8_t> u_has_leaf_right;
    UsmVector<int32_t> u_left_child;
    UsmVector<int32_t> u_parents;
  } brt;

  struct Octree {
    explicit Octree(const size_t n_nodes, std::pmr::memory_resource* mr);

    // int (*u_children)[8]; note, this is 8x more
    UsmVector<int32_t> u_children;

    // everything else is size of 'n_octree_nodes'
    // but for simplicity, we allocate the 0.6 times of input size
    // 60% memory is an empirical value
    UsmVector<glm::vec4> u_corner;
    UsmVector<float> u_cell_size;
    UsmVector<int32_t> u_child_node_mask;
    UsmVector<int32_t> u_child_leaf_mask;
  } oct;

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

  void set_n_unique(const uint32_t n_unique) { this->n_unique = n_unique; }

  void set_n_brt_nodes(const uint32_t n_brt_nodes) {
    this->n_brt_nodes = n_brt_nodes;
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

  void set_n_octree_nodes(const uint32_t n_octree_nodes) {
    this->n_octree_nodes = n_octree_nodes;
  }

  // The Idea is.

  // after calling sort, 'u_morton_keys_alt' is sorted
  // after calling move_dups, 'u_morton_keys' is sorted and unique
  // there's no way to check if you called, so I trust you call them in order

  [[nodiscard]] uint32_t* get_sorted_morton_keys() {
    return u_morton_keys_alt.data();
  }

  [[nodiscard]] const uint32_t* get_sorted_morton_keys() const {
    return u_morton_keys_alt.data();
  }

  [[nodiscard]] uint32_t* get_unique_morton_keys() {
    return u_morton_keys.data();
  }

  [[nodiscard]] const uint32_t* get_unique_morton_keys() const {
    return u_morton_keys.data();
  }
};

}  // namespace tree
