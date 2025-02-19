#pragma once

#include <gtest/gtest.h>

#include "builtin-apps/tree/tree_appdata.hpp"

namespace test_tree {

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

inline void verify_stage_1(tree::AppData &appdata) {
  // Verify results are non-zero
  for (auto i = 0u; i < appdata.get_n_input(); i++) {
    EXPECT_NE(appdata.u_morton_keys_s1[i], 0) << "Morton code at index " << i << " is zero";
  }
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

inline void verify_stage_2(tree::AppData &appdata) {
  // Verify results are non-zero and sorted
  for (auto i = 0u; i < appdata.get_n_input(); i++) {
    EXPECT_NE(appdata.u_morton_keys_sorted_s2[i], 0)
        << "Sorted Morton code at index " << i << " is zero";

    if (i > 0) {
      EXPECT_LT(appdata.u_morton_keys_sorted_s2[i], appdata.u_morton_keys_sorted_s2[i - 1])
          << "Sorted Morton code at index " << i << " is not less than the previous code";
    }
  }
}

}  // namespace test_tree
