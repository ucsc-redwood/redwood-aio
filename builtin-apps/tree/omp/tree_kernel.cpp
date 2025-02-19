#include "tree_kernel.hpp"

#include <numeric>

#include "func_brt.hpp"
#include "func_edge.hpp"
#include "func_morton.hpp"
#include "func_octree.hpp"
#include "func_sort.hpp"

namespace tree {

namespace omp {

// ----------------------------------------------------------------------------
// Stage 1 (xyz -> morton)
// ----------------------------------------------------------------------------

void process_stage_1(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  const int start = 0;
  const int end = appdata.get_n_input();

#pragma omp for
  for (int i = start; i < end; ++i) {
    appdata.u_morton_keys_s1[i] =
        xyz_to_morton32(appdata.u_input_points_s0[i], tree::kMinCoord, tree::kRange);
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (morton -> sorted morton)
// ----------------------------------------------------------------------------

void process_stage_2(tree::AppData &appdata, TempStorage &temp_storage) {
  const auto num_threads = omp_get_num_threads();

  bucket_sort(appdata.u_morton_keys_s1.data(),
              appdata.u_morton_keys_sorted_s2.data(),
              temp_storage.global_n_elem,
              temp_storage.global_starting_position,
              temp_storage.buckets,
              appdata.get_n_input(),
              num_threads,
              num_threads);

#pragma omp barrier
  // by this point, 'u_morton_keys_sorted_s2' is sorted
}

// ----------------------------------------------------------------------------
// Stage 3 (sorted morton -> unique morton)
// ----------------------------------------------------------------------------

void process_stage_3(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  const auto last = std::unique_copy(appdata.u_morton_keys_sorted_s2.data(),
                                     appdata.u_morton_keys_sorted_s2.data() + appdata.get_n_input(),
                                     appdata.u_morton_keys_unique_s3.data());
  const auto n_unique = std::distance(appdata.u_morton_keys_unique_s3.data(), last);

  appdata.set_n_unique(n_unique);
  appdata.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (unique morton -> brt)
// ----------------------------------------------------------------------------

void process_stage_4(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  const int start = 0;
  const int end = appdata.get_n_unique();

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_radix_tree_i(i,
                         appdata.get_n_brt_nodes(),
                         appdata.u_morton_keys_unique_s3.data(),
                         appdata.u_brt_prefix_n_s4.data(),
                         appdata.u_brt_has_leaf_left_s4.data(),
                         appdata.u_brt_has_leaf_right_s4.data(),
                         appdata.u_brt_left_child_s4.data(),
                         appdata.u_brt_parents_s4.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 5 (brt -> edge count)
// ----------------------------------------------------------------------------

void process_stage_5(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  for (int i = start; i < end; ++i) {
    process_edge_count_i(i,
                         appdata.u_brt_prefix_n_s4.data(),
                         appdata.u_brt_parents_s4.data(),
                         appdata.u_edge_count_s5.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 6 (edge count -> edge offset)
// ----------------------------------------------------------------------------

void process_stage_6(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  const int start = 0;
  const int end = appdata.get_n_brt_nodes();

  std::partial_sum(appdata.u_edge_count_s5.data() + start,
                   appdata.u_edge_count_s5.data() + end,
                   appdata.u_edge_offset_s6.data() + start);

  // num_octree node is the result of the partial sum
  const auto num_octree_nodes = appdata.u_edge_offset_s6[end - 1];

  appdata.set_n_octree_nodes(num_octree_nodes);
}
// ----------------------------------------------------------------------------
// Stage 7 (everything -> octree)
// ----------------------------------------------------------------------------

void process_stage_7(tree::AppData &appdata, [[maybe_unused]] TempStorage &temp_storage) {
  // note: 1 here, skipping root
  const int start = 1;
  const int end = appdata.get_n_octree_nodes();

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_oct_node(i,
                     reinterpret_cast<int(*)[8]>(appdata.u_oct_children_s7.data()),
                     appdata.u_oct_corner_s7.data(),
                     appdata.u_oct_cell_size_s7.data(),
                     appdata.u_oct_child_node_mask_s7.data(),
                     appdata.u_edge_offset_s6.data(),
                     appdata.u_edge_count_s5.data(),
                     appdata.u_morton_keys_unique_s3.data(),
                     appdata.u_brt_prefix_n_s4.data(),
                     appdata.u_brt_parents_s4.data(),
                     tree::kMinCoord,
                     tree::kRange);
  }
}

}  // namespace omp

}  // namespace tree
