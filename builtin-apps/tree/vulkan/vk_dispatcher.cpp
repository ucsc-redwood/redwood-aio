
#include "vk_dispatcher.hpp"

#include <cstdint>
#include <numeric>
#include <random>

#include "../../app.hpp"

namespace tree {

namespace vulkan {

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.sequence()) {
  spdlog::info("Singleton instance created.");
}

// ----------------------------------------------------------------------------
// Stage 1 (Input -> Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_1(tree::AppData &app_data_ref) {
  struct PushConstants {
    uint32_t n;
    float min_coord;
    float range;
  };

  static auto morton_algo =
      engine
          .algorithm(
              "tree_morton.comp",
              {
                  engine.get_buffer(app_data_ref.u_input_points_s0.data()),
                  engine.get_buffer(app_data_ref.u_morton_keys_s1.data()),
              })
          ->set_push_constants<PushConstants>({
              .n = static_cast<uint32_t>(app_data_ref.get_n_input()),
              .min_coord = tree::kMinCoord,
              .range = tree::kRange,
          })
          ->build();

  seq->record_commands(morton_algo.get(), app_data_ref.get_n_input());

  seq->launch_kernel_async();
  seq->sync();

  //   std::iota(app_data_ref.u_morton_keys_s1.data(),
  //             app_data_ref.u_morton_keys_s1.data() +
  //             app_data_ref.get_n_input(), 0);

  //   std::mt19937 g(42);
  //   std::ranges::shuffle(app_data_ref.u_morton_keys_s1, g);
}

// ----------------------------------------------------------------------------
// Stage 2 (Morton -> Sorted Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_2(tree::AppData &app_data_ref) {
  struct PushConstants {
    uint32_t g_num_elements;
  };

  static auto shader_name = "tmp_single_radixsort_warp" +
                            std::to_string(get_vulkan_warp_size()) + ".comp";

  auto algo =
      engine
          .algorithm(
              shader_name,
              {
                  engine.get_buffer(app_data_ref.u_morton_keys_s1.data()),
                  engine.get_buffer(
                      app_data_ref.u_morton_keys_sorted_s2.data()),
              })
          ->set_push_constants<PushConstants>({
              .g_num_elements = app_data_ref.get_n_input(),
          })
          ->build();

  seq->record_commands_with_blocks(algo.get(), 1);

  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_3(tree::AppData &app_data_ref,
                                TmpStorage &tmp_storage) {
  const auto last = std::unique_copy(
      app_data_ref.u_morton_keys_sorted_s2.data(),
      app_data_ref.u_morton_keys_sorted_s2.data() + app_data_ref.get_n_input(),
      app_data_ref.u_morton_keys_unique_s3.data());
  const auto n_unique =
      std::distance(app_data_ref.u_morton_keys_unique_s3.data(), last);

  app_data_ref.set_n_unique(n_unique);
  app_data_ref.set_n_brt_nodes(n_unique - 1);
}

// ----------------------------------------------------------------------------
// Stage 4 (Unique Sorted Morton -> Radix Tree)
// ----------------------------------------------------------------------------

void Singleton::process_stage_4(tree::AppData &app_data_ref) {
  const int32_t n = app_data_ref.get_n_unique();

  auto build_radix_tree_algo =
      engine
          .algorithm(
              "tree_build_radix_tree.comp",
              {
                  engine.get_buffer(
                      app_data_ref.u_morton_keys_unique_s3.data()),
                  engine.get_buffer(app_data_ref.u_brt_prefix_n_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_has_leaf_left_s4.data()),
                  engine.get_buffer(
                      app_data_ref.u_brt_has_leaf_right_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_left_child_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_parents_s4.data()),
              })
          ->set_push_constants<InputSizePushConstantsSigned>({
              .n = n,
          })
          ->build();

  seq->record_commands(build_radix_tree_algo.get(), n);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 5 (Radix Tree -> Edge Count)
// ----------------------------------------------------------------------------

void Singleton::process_stage_5(tree::AppData &app_data_ref) {
  auto edge_count_algo =
      engine
          .algorithm(
              "tree_edge_count.comp",
              {
                  engine.get_buffer(app_data_ref.u_brt_prefix_n_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_parents_s4.data()),
                  engine.get_buffer(app_data_ref.u_edge_count_s5.data()),
              })
          ->set_push_constants<InputSizePushConstantsUnsigned>({
              .n = app_data_ref.get_n_brt_nodes(),
          })
          ->build();

  seq->record_commands(edge_count_algo.get(), app_data_ref.get_n_brt_nodes());
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 6 (Edge Count -> Edge Offset, prefix sum)
// ----------------------------------------------------------------------------

void Singleton::process_stage_6(tree::AppData &app_data_ref) {
  const int start = 0;
  const int end = app_data_ref.get_n_brt_nodes();

  std::partial_sum(app_data_ref.u_edge_count_s5.data() + start,
                   app_data_ref.u_edge_count_s5.data() + end,
                   app_data_ref.u_edge_offset_s6.data() + start);

  // num_octree node is the result of the partial sum
  const auto num_octree_nodes = app_data_ref.u_edge_offset_s6[end - 1];

  app_data_ref.set_n_octree_nodes(num_octree_nodes);
}

//----------------------------------------------------------------------------
// Stage 7 (Edge Offset -> Octree)
//----------------------------------------------------------------------------

void Singleton::process_stage_7(tree::AppData &app_data_ref) {
  struct OctreePushConstants {
    float min_coord;
    float range;
    int32_t n_brt_nodes;
  };

  auto build_octree_algo =
      engine
          .algorithm(
              "tree_build_octree.comp",
              {
                  engine.get_buffer(app_data_ref.u_oct_children_s7.data()),
                  engine.get_buffer(app_data_ref.u_oct_corner_s7.data()),
                  engine.get_buffer(app_data_ref.u_oct_cell_size_s7.data()),
                  engine.get_buffer(
                      app_data_ref.u_oct_child_node_mask_s7.data()),
                  engine.get_buffer(
                      app_data_ref.u_oct_child_leaf_mask_s7.data()),
                  engine.get_buffer(app_data_ref.u_edge_offset_s6.data()),
                  engine.get_buffer(app_data_ref.u_edge_count_s5.data()),
                  engine.get_buffer(
                      app_data_ref.u_morton_keys_unique_s3.data()),
                  engine.get_buffer(app_data_ref.u_brt_prefix_n_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_parents_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_left_child_s4.data()),
                  engine.get_buffer(app_data_ref.u_brt_has_leaf_left_s4.data()),
                  engine.get_buffer(
                      app_data_ref.u_brt_has_leaf_right_s4.data()),
              })
          ->set_push_constants<OctreePushConstants>({
              .min_coord = tree::kMinCoord,
              .range = tree::kRange,
              .n_brt_nodes =
                  static_cast<int32_t>(app_data_ref.get_n_brt_nodes()),
          })
          ->build();

  seq->record_commands(build_octree_algo.get(),
                       app_data_ref.get_n_octree_nodes());
  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan

}  // namespace tree