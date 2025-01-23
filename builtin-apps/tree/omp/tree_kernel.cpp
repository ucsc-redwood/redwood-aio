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

void process_stage_1(tree::AppData &app_data) {
  const int start = 0;
  const int end = app_data.get_n_input();

#pragma omp for
  for (int i = start; i < end; ++i) {
    app_data.u_morton_keys[i] = xyz_to_morton32(
        app_data.u_input_points[i], app_data.min_coord, app_data.range);
  }
}

// ----------------------------------------------------------------------------
// Stage 2 (morton -> sorted morton)
// ----------------------------------------------------------------------------

namespace v2 {
void process_stage_2(tree::AppData &app_data, v2::TempStorage &temp_storage) {
  const auto num_threads = omp_get_num_threads();

  v2::bucket_sort(app_data.u_morton_keys.data(),
                  app_data.u_morton_keys_alt.data(),
                  temp_storage.global_n_elem,
                  temp_storage.global_starting_position,
                  temp_storage.buckets,
                  app_data.get_n_input(),
                  num_threads,
                  num_threads);
}
}  // namespace v2

// ----------------------------------------------------------------------------
// Stage 3 (sorted morton -> unique morton)
// ----------------------------------------------------------------------------

void process_stage_3(tree::AppData &app_data) {
  const auto last =
      std::unique_copy(app_data.u_morton_keys.data(),
                       app_data.u_morton_keys.data() + app_data.get_n_input(),
                       app_data.u_morton_keys_alt.data());
  const auto n_unique = std::distance(app_data.u_morton_keys_alt.data(), last);

  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (unique morton -> brt)
// ----------------------------------------------------------------------------

void process_stage_4(tree::AppData &app_data) {
  const int start = 0;
  const int end = app_data.get_n_unique();

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_radix_tree_i(i,
                         app_data.get_n_brt_nodes(),
                         app_data.get_unique_morton_keys(),
                         app_data.brt.u_prefix_n.data(),
                         app_data.brt.u_has_leaf_left.data(),
                         app_data.brt.u_has_leaf_right.data(),
                         app_data.brt.u_left_child.data(),
                         app_data.brt.u_parents.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 5 (brt -> edge count)
// ----------------------------------------------------------------------------

void process_stage_5(tree::AppData &app_data) {
  const int start = 0;
  const int end = app_data.get_n_brt_nodes();

  for (int i = start; i < end; ++i) {
    process_edge_count_i(i,
                         app_data.brt.u_prefix_n.data(),
                         app_data.brt.u_parents.data(),
                         app_data.u_edge_count.data());
  }
}

// ----------------------------------------------------------------------------
// Stage 6 (edge count -> edge offset)
// ----------------------------------------------------------------------------

void process_stage_6(tree::AppData &app_data) {
  const int start = 0;
  const int end = app_data.get_n_brt_nodes();

  std::partial_sum(app_data.u_edge_count.data() + start,
                   app_data.u_edge_count.data() + end,
                   app_data.u_edge_offset.data() + start);

  // num_octree node is the result of the partial sum
  const auto num_octree_nodes = app_data.u_edge_offset[end - 1];

  app_data.set_n_octree_nodes(num_octree_nodes);
}
// ----------------------------------------------------------------------------
// Stage 7 (everything -> octree)
// ----------------------------------------------------------------------------

void process_stage_7(tree::AppData &app_data) {
  // note: 1 here, skipping root
  const int start = 1;
  const int end = app_data.get_n_octree_nodes();

#pragma omp for
  for (int i = start; i < end; ++i) {
    process_oct_node(
        i,
        reinterpret_cast<int(*)[8]>(app_data.oct.u_children.data()),
        app_data.oct.u_corner.data(),
        app_data.oct.u_cell_size.data(),
        app_data.oct.u_child_node_mask.data(),
        app_data.u_edge_offset.data(),
        app_data.u_edge_count.data(),
        app_data.get_unique_morton_keys(),
        app_data.brt.u_prefix_n.data(),
        app_data.brt.u_parents.data(),
        app_data.min_coord,
        app_data.range);
  }
}

}  // namespace omp

}  // namespace tree
