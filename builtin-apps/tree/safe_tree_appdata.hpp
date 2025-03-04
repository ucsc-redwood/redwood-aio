#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <stdexcept>

#include "../base_appdata.hpp"
#include "tree_appdata.hpp"

namespace tree {

struct SafeAppData final : BaseAppData {
  explicit SafeAppData(std::pmr::memory_resource* mr);

  ~SafeAppData() override = default;

  // --------------------------------------------------------------------------
  // Essential data
  // --------------------------------------------------------------------------
  const uint32_t n_input;
  const uint32_t n_unique;
  const uint32_t n_brt_nodes;
  const uint32_t n_octree_nodes;

  // --------------------------------------------------------------------------
  // Stage 1: xyz -> morton
  // --------------------------------------------------------------------------
  UsmVector<glm::vec4> u_input_points_s0;
  UsmVector<uint32_t> u_morton_keys_s1;

  UsmVector<uint32_t> u_morton_keys_s1_out;

  // --------------------------------------------------------------------------
  // Stage 2: morton -> sorted morton
  // --------------------------------------------------------------------------
  UsmVector<uint32_t> u_morton_keys_sorted_s2;

  UsmVector<uint32_t> u_morton_keys_sorted_s2_out;

  // --------------------------------------------------------------------------
  // Stage 3: sorted morton -> unique morton
  // --------------------------------------------------------------------------
  UsmVector<uint32_t> u_morton_keys_unique_s3;

  UsmVector<uint32_t> u_morton_keys_unique_s3_out;

  // --------------------------------------------------------------------------
  // Stage 4: unique morton -> Binary Radix Tree (BRT)
  // --------------------------------------------------------------------------
  UsmVector<uint8_t> u_brt_prefix_n_s4;
  UsmVector<uint8_t> u_brt_has_leaf_left_s4;
  UsmVector<uint8_t> u_brt_has_leaf_right_s4;
  UsmVector<int32_t> u_brt_left_child_s4;
  UsmVector<int32_t> u_brt_parents_s4;

  UsmVector<uint8_t> u_brt_prefix_n_s4_out;
  UsmVector<uint8_t> u_brt_has_leaf_left_s4_out;
  UsmVector<uint8_t> u_brt_has_leaf_right_s4_out;
  UsmVector<int32_t> u_brt_left_child_s4_out;
  UsmVector<int32_t> u_brt_parents_s4_out;

  // --------------------------------------------------------------------------
  // Stage 5: BRT -> edge count
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_edge_count_s5;

  UsmVector<int32_t> u_edge_count_s5_out;

  // --------------------------------------------------------------------------
  // Stage 6: edge count -> edge offset
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_edge_offset_s6;

  UsmVector<int32_t> u_edge_offset_s6_out;

  // --------------------------------------------------------------------------
  // Stage 7: Build Octree
  // --------------------------------------------------------------------------
  UsmVector<int32_t> u_oct_children_s7;  // 8 * sizeof
  UsmVector<glm::vec4> u_oct_corner_s7;
  UsmVector<float> u_oct_cell_size_s7;
  UsmVector<int32_t> u_oct_child_node_mask_s7;
  UsmVector<int32_t> u_oct_child_leaf_mask_s7;

  UsmVector<int32_t> u_oct_children_s7_out;
  UsmVector<glm::vec4> u_oct_corner_s7_out;
  UsmVector<float> u_oct_cell_size_s7_out;
  UsmVector<int32_t> u_oct_child_node_mask_s7_out;
  UsmVector<int32_t> u_oct_child_leaf_mask_s7_out;

  // --------------------------------------------------------------------------
  // Getters / Setters
  // --------------------------------------------------------------------------

  [[nodiscard]] uint32_t get_n_input() const { return n_input; }

  [[nodiscard]] uint32_t get_n_unique() const { return n_unique; }

  [[nodiscard]] uint32_t get_n_brt_nodes() const { return n_brt_nodes; }

  [[nodiscard]] uint32_t get_n_octree_nodes() const { return n_octree_nodes; }

  void set_n_unique([[maybe_unused]] const uint32_t n_unique) {  // No-op
  }

  void set_n_brt_nodes([[maybe_unused]] const uint32_t n_brt_nodes) {  // No-op
  }

  void set_n_octree_nodes([[maybe_unused]] const uint32_t n_octree_nodes) {  // No-op
  }
};

class HostTreeManager {
 public:
  static HostTreeManager& getInstance() {
    static HostTreeManager instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  HostTreeManager(const HostTreeManager&) = delete;
  HostTreeManager& operator=(const HostTreeManager&) = delete;

  // Initialize the tree data
  void initialize();

  // Get the AppData
  tree::AppData* getAppData() { return appdata_.get(); }

 private:
  HostTreeManager() = default;
  std::unique_ptr<tree::AppData> appdata_;
};

}  // namespace tree
