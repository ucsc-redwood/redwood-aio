#pragma once

#include <gtest/gtest.h>

#include <algorithm>

#include "builtin-apps/tree/tree_appdata.hpp"

namespace test_tree {

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

inline void verify_stage_1(tree::AppData &appdata) {
  // EXPECT_TRUE(std::ranges::all_of(appdata.u_morton_keys_s1, [](auto x) { return x != 0; }));

  // check if all morton codes are non-zero
  for (uint32_t i = 0; i < appdata.get_n_input(); ++i) {
    // std::cout << "morton[" << i << "] = " << appdata.u_morton_keys_s1[i] << std::endl;
    EXPECT_NE(appdata.u_morton_keys_s1[i], 0);
  }
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

inline void verify_stage_2(tree::AppData &appdata) {
  EXPECT_TRUE(std::ranges::all_of(appdata.u_morton_keys_sorted_s2, [](auto x) { return x != 0; }));
  // EXPECT_TRUE(std::ranges::is_sorted(appdata.u_morton_keys_sorted_s2));

  for (uint32_t i = 0; i < appdata.get_n_input() - 1; ++i) {
    EXPECT_LE(appdata.u_morton_keys_sorted_s2[i], appdata.u_morton_keys_sorted_s2[i + 1]);
  }
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

inline void verify_stage_3(tree::AppData &appdata) {
  std::vector<uint32_t> expected_unique_morton_keys(appdata.get_n_input());

  const auto last = std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
                                     appdata.u_morton_keys_sorted_s2.data() + appdata.get_n_input(),
                                     expected_unique_morton_keys.data());
  const auto num_unique = std::distance(expected_unique_morton_keys.data(), last);

  EXPECT_EQ(appdata.get_n_unique(), num_unique);
  EXPECT_EQ(appdata.get_n_brt_nodes(), num_unique - 1);

  for (uint32_t i = 0; i < num_unique; ++i) {
    EXPECT_EQ(appdata.u_morton_keys_unique_s3[i], expected_unique_morton_keys[i]);
  }
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

inline void verify_stage_4(tree::AppData &appdata) {
  // Verify BRT parents are non-zero
  ASSERT_GT(appdata.get_n_brt_nodes(), 0);

  ASSERT_EQ(appdata.u_brt_parents_s4[0], 0);
  for (uint32_t i = 1; i < appdata.get_n_brt_nodes(); ++i) {
    EXPECT_GE(appdata.u_brt_parents_s4[i], 0);
  }
}

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

inline void verify_stage_5(tree::AppData &appdata) {
  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata.u_edge_count_s5, [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);
}

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

inline void verify_stage_6(tree::AppData &appdata) {
  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata.u_edge_offset_s6, [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);

  // Verify edge offsets are monotonically increasing
  for (uint32_t i = 1; i < appdata.get_n_brt_nodes(); ++i) {
    EXPECT_LE(appdata.u_edge_offset_s6[i - 1], appdata.u_edge_offset_s6[i]);
  }
}

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

inline void verify_stage_7(tree::AppData &appdata) {
  // Verify not all are zeros
  bool all_zeros =
      std::ranges::all_of(appdata.u_oct_child_node_mask_s7, [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);

  // Verify octree nodes
  ASSERT_GT(appdata.get_n_octree_nodes(), 0);
  for (uint32_t i = 0; i < appdata.get_n_octree_nodes(); ++i) {
    EXPECT_GE(appdata.u_oct_child_node_mask_s7[i], 0);
    EXPECT_LE(appdata.u_oct_child_node_mask_s7[i], 255);  // 8-bit mask
  }
}

}  // namespace test_tree
