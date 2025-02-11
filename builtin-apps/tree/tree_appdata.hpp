#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <stdexcept>

#include "../base_appdata.hpp"

namespace tree {

// From empirical observation, 60% memory is a good ratio for octree
constexpr auto kMemoryRatio = 0.6f;

// Default problem size, other sizes are
constexpr auto kDefaultInputSize = 640 * 480;
constexpr auto kMinCoord = 0.0f;
constexpr auto kRange = 1024.0f;

// clang-format off
// Data structure for managing buffers in the octree construction pipeline.
//
// Buffer Allocation Summary:
// -----------------------------------------------------------------------------------------------
// | Stage | Buffer Name                  | Allocated Size               | Real Data Used          |
// |-------|------------------------------|------------------------------|-------------------------|
// | 1     | u_input_points_s0            | n_input                      | n_input                 |
// | 1     | u_morton_keys_s1             | n_input                      | n_input                 |
// | 2     | u_morton_keys_sorted_s2      | n_input                      | n_input                 |
// | 3     | u_morton_keys_unique_s3      | n_input                      | n_unique                |
// | 4     | u_brt_prefix_n_s4            | n_input                      | n_brt_nodes             |
// | 4     | u_brt_has_leaf_left_s4       | n_input                      | n_brt_nodes             |
// | 4     | u_brt_has_leaf_right_s4      | n_input                      | n_brt_nodes             |
// | 4     | u_brt_left_child_s4          | n_input                      | n_brt_nodes             |
// | 4     | u_brt_parents_s4             | n_input                      | n_brt_nodes             |
// | 5     | u_edge_count_s5              | n_input                      | n_brt_nodes             |
// | 6     | u_edge_offset_s6             | n_input                      | n_brt_nodes             |
// | 7     | u_oct_corner_s7              | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_cell_size_s7           | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_child_node_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_child_leaf_mask_s7     | n_input * 0.6f               | n_octree_nodes          |
// | 7     | u_oct_children_s7            | 8 * n_input * 0.6f           | 8 * n_octree_nodes      |
// ------------------------------------------------------------------------------------------------
// Notes:
// - `n_input` is the total number of input points.
// - `n_unique` is the number of unique Morton keys (`≤ n_input`).
// - `n_brt_nodes` is the number of Binary Radix Tree nodes (`= n_unique - 1`).
// - `n_octree_nodes` is determined after Stage 6, usually ~50% of `n_input`.
// - `u_oct_children_s7` is 8× larger because each octree node can have up to 8 children.
// clang-format on

struct AppData final : BaseAppData {
  explicit AppData(std::pmr::memory_resource* mr, const size_t n_input = kDefaultInputSize);

  ~AppData() override = default;

  // --------------------------------------------------------------------------
  // Essential data
  // --------------------------------------------------------------------------
  const uint32_t n_input;
  uint32_t n_unique = std::numeric_limits<uint32_t>::max();
  uint32_t n_brt_nodes = std::numeric_limits<uint32_t>::max();
  uint32_t n_octree_nodes = std::numeric_limits<uint32_t>::max();

  // --------------------------------------------------------------------------
  // Stage 1: xyz -> morton
  // --------------------------------------------------------------------------
  UsmVector<glm::vec4> u_input_points_s0;
  UsmVector<uint32_t> u_morton_keys_s1;

  // --------------------------------------------------------------------------
  // Stage 2: morton -> sorted morton
  // --------------------------------------------------------------------------
  UsmVector<uint32_t> u_morton_keys_sorted_s2;

  // --------------------------------------------------------------------------
  // Stage 3: sorted morton -> unique morton
  // --------------------------------------------------------------------------
  UsmVector<uint32_t> u_morton_keys_unique_s3;

  // --------------------------------------------------------------------------
  // Stage 4: unique morton -> Binary Radix Tree (BRT)
  // --------------------------------------------------------------------------
  UsmVector<uint8_t> u_brt_prefix_n_s4;
  UsmVector<uint8_t> u_brt_has_leaf_left_s4;
  UsmVector<uint8_t> u_brt_has_leaf_right_s4;
  UsmVector<int32_t> u_brt_left_child_s4;
  UsmVector<int32_t> u_brt_parents_s4;

  // --------------------------------------------------------------------------
  // Stage 5: BRT -> edge count
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_edge_count_s5;

  // --------------------------------------------------------------------------
  // Stage 6: edge count -> edge offset
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_edge_offset_s6;

  // --------------------------------------------------------------------------
  // Stage 7: Build Octree
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_oct_children_s7;  // 8 * sizeof
  UsmVector<glm::vec4> u_oct_corner_s7;
  UsmVector<float> u_oct_cell_size_s7;
  UsmVector<int32_t> u_oct_child_node_mask_s7;
  UsmVector<int32_t> u_oct_child_leaf_mask_s7;

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

  [[nodiscard]] uint32_t get_n_brt_nodes() const { return this->get_n_unique() - 1; }

  [[nodiscard]] uint32_t get_n_octree_nodes() const {
    if (n_octree_nodes == std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("n_octree_nodes is not set");
    }

    return n_octree_nodes;
  }

  void set_n_unique(const uint32_t n_unique) { this->n_unique = n_unique; }

  void set_n_brt_nodes(const uint32_t n_brt_nodes) { this->n_brt_nodes = n_brt_nodes; }

  void set_n_octree_nodes(const uint32_t n_octree_nodes) { this->n_octree_nodes = n_octree_nodes; }
};

}  // namespace tree
