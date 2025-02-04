
#include "vk_dispatcher.hpp"

namespace tree {

namespace vulkan {

// ----------------------------------------------------------------------------
// Singleton Constructor
// ----------------------------------------------------------------------------

// //
// --------------------------------------------------------------------------
// // Stage 3
// //
// --------------------------------------------------------------------------

// struct FindDupsPushConstants {
//   int32_t n;
// };

// struct MoveDupsPushConstants {
//   uint32_t n;
// };

// //
// --------------------------------------------------------------------------
// // Stage 4
// //
// --------------------------------------------------------------------------

// struct BuildTreePushConstants {
//   int32_t n;
// };

// //
// --------------------------------------------------------------------------
// // Stage 5
// //
// --------------------------------------------------------------------------

// struct EdgeCountPushConstants {
//   int32_t n_brt_nodes;
// };

// //
// --------------------------------------------------------------------------
// // Stage 6
// //
// --------------------------------------------------------------------------

// struct PrefixSumPushConstants {
//   uint32_t inputSize;
// };

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.sequence()) {
  spdlog::info("Singleton instance created.");

  // tmp
  tree::AppData app_data(engine.get_mr());
  auto tmp_storage =
      ::vulkan::TmpStorage(engine.get_mr(), app_data.get_n_input());

  // --------------------------------------------------------------------------
  // Morton
  // --------------------------------------------------------------------------

  auto morton_algo =
      engine
          .algorithm("tree_morton.comp",
                     {
                         engine.get_buffer(app_data.u_input_points.data()),
                         engine.get_buffer(app_data.get_unsorted_morton_keys()),
                     })
          ->set_push_constants<MortonPushConstants>({
              .n = static_cast<uint32_t>(app_data.get_n_input()),
              .min_coord = tree::kMinCoord,
              .range = tree::kRange,
          })
          ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  // --------------------------------------------------------------------------
  // Radix Sort
  // --------------------------------------------------------------------------

  std::string shader_name =
      "tmp_single_radixsort_warp" + std::to_string(get_warp_size()) + ".comp";

  auto radix_sort_algo =
      engine
          .algorithm(shader_name,
                     {
                         // In
                         engine.get_buffer(app_data.get_unsorted_morton_keys()),
                         // Out
                         engine.get_buffer(app_data.get_sorted_morton_keys()),
                     })
          ->set_push_constants<InputSizePushConstantsUnsigned>({
              .n = static_cast<uint32_t>(app_data.get_n_input()),
          })
          ->build();

  cached_algorithms.try_emplace("radix_sort", std::move(radix_sort_algo));

  // --------------------------------------------------------------------------
  // Find Dups
  // --------------------------------------------------------------------------

  auto find_dups_algo =
      engine
          .algorithm("tree_find_dups.comp",
                     {
                         // In
                         engine.get_buffer(app_data.get_sorted_morton_keys()),
                         // Out
                         engine.get_buffer(tmp_storage.u_contributes.data()),
                     })
          ->set_push_constants<InputSizePushConstantsSigned>({
              .n = static_cast<int32_t>(app_data.get_n_input()),
          })
          ->build();

  cached_algorithms.try_emplace("find_dups", std::move(find_dups_algo));

  // --------------------------------------------------------------------------
  // Move Dups
  // --------------------------------------------------------------------------

  auto move_dups_algo =
      engine
          .algorithm(
              "tree_move_dups.comp",
              {

                  // OutIdx
                  engine.get_buffer(tmp_storage.u_out_idx.data()),

                  // InKeys
                  engine.get_buffer(app_data.get_sorted_morton_keys()),

                  // OutKeys
                  engine.get_buffer(app_data.get_sorted_unique_morton_keys()),
              })
          ->set_push_constants<InputSizePushConstantsUnsigned>({
              .n = static_cast<uint32_t>(app_data.get_n_input()),
          })
          ->build();

  cached_algorithms.try_emplace("move_dups", std::move(move_dups_algo));

  // --------------------------------------------------------------------------
  // Build Radix Tree
  // --------------------------------------------------------------------------

  auto build_radix_tree_algo =
      engine
          .algorithm(
              "tree_build_radix_tree.comp",
              {
                  engine.get_buffer(app_data.get_sorted_unique_morton_keys()),
                  engine.get_buffer(app_data.u_brt_prefix_n.data()),
                  engine.get_buffer(app_data.u_brt_has_leaf_left.data()),
                  engine.get_buffer(app_data.u_brt_has_leaf_right.data()),
                  engine.get_buffer(app_data.u_brt_left_child.data()),
                  engine.get_buffer(app_data.u_brt_parents.data()),
              })
          ->place_holder_push_constants<InputSizePushConstantsSigned>()
          ->build();

  cached_algorithms.try_emplace("build_radix_tree",
                                std::move(build_radix_tree_algo));

  // --------------------------------------------------------------------------
  // Edge Count
  // --------------------------------------------------------------------------

  auto edge_count_algo =
      engine
          .algorithm("tree_edge_count.comp",
                     {
                         engine.get_buffer(app_data.u_brt_prefix_n.data()),
                         engine.get_buffer(app_data.u_brt_parents.data()),
                         engine.get_buffer(app_data.u_edge_count.data()),
                     })
          ->place_holder_push_constants<InputSizePushConstantsSigned>()
          ->build();

  cached_algorithms.try_emplace("edge_count", std::move(edge_count_algo));

  // --------------------------------------------------------------------------
  // Prefix Sum
  // --------------------------------------------------------------------------

  auto prefix_sum_algo =
      engine
          .algorithm("tree_naive_prefix_sum.comp",
                     {
                         engine.get_buffer(app_data.u_edge_count.data()),
                         engine.get_buffer(app_data.u_edge_offset.data()),
                     })
          ->place_holder_push_constants<InputSizePushConstantsUnsigned>()
          ->build();

  cached_algorithms.try_emplace("prefix_sum", std::move(prefix_sum_algo));

  // --------------------------------------------------------------------------
  // Build Octree
  // --------------------------------------------------------------------------

  auto build_octree_algo =
      engine
          .algorithm(
              "tree_build_octree.comp",
              {
                  engine.get_buffer(app_data.u_oct_children.data()),
                  engine.get_buffer(app_data.u_oct_corner.data()),
                  engine.get_buffer(app_data.u_oct_cell_size.data()),
                  engine.get_buffer(app_data.u_oct_child_node_mask.data()),
                  engine.get_buffer(app_data.u_oct_child_leaf_mask.data()),
                  engine.get_buffer(app_data.u_edge_offset.data()),
                  engine.get_buffer(app_data.u_edge_count.data()),
                  engine.get_buffer(app_data.get_sorted_unique_morton_keys()),
                  engine.get_buffer(app_data.u_brt_prefix_n.data()),
                  engine.get_buffer(app_data.u_brt_parents.data()),
                  engine.get_buffer(app_data.u_brt_left_child.data()),
                  engine.get_buffer(app_data.u_brt_has_leaf_left.data()),
                  engine.get_buffer(app_data.u_brt_has_leaf_right.data()),
              })
          ->place_holder_push_constants<OctreePushConstants>()
          ->build();

  cached_algorithms.try_emplace("build_octree", std::move(build_octree_algo));
}

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

void Singleton::process_stage_1(tree::AppData &app_data_ref) {
  const int total_iterations = app_data_ref.get_n_input();

  auto algo = cached_algorithms.at("morton").get();

  seq->record_commands(algo, total_iterations);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

void Singleton::process_stage_2(tree::AppData &app_data_ref) {
  const uint32_t n = app_data_ref.get_n_input();

  auto algo = cached_algorithms.at("radix_sort").get();

  algo->update_push_constants(InputSizePushConstantsUnsigned{
      .n = n,
  });

  seq->record_commands_with_blocks(algo, 1);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

void Singleton::process_stage_3(tree::AppData &app_data_ref,
                                ::vulkan::TmpStorage &tmp_storage) {
  const uint32_t n = app_data_ref.get_n_input();

  auto find_dups = cached_algorithms.at("find_dups").get();

  auto prefix_sum = cached_algorithms.at("prefix_sum").get();

  auto move_dups = cached_algorithms.at("move_dups").get();

  find_dups->update_push_constants(InputSizePushConstantsSigned{
      .n = static_cast<int32_t>(n),
  });

  prefix_sum->update_push_constants(InputSizePushConstantsUnsigned{
      .n = n,
  });
  prefix_sum->update_descriptor_sets({
      engine.get_buffer(tmp_storage.u_contributes.data()),
      engine.get_buffer(tmp_storage.u_out_idx.data()),
  });

  move_dups->update_push_constants(InputSizePushConstantsUnsigned{
      .n = n,
  });

  seq->record_commands(find_dups, n);
  seq->launch_kernel_async();
  seq->sync();

  // print 10 u_contributes
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_contributes[{}] = {}", i, tmp_storage.u_contributes[i]);
  }

  seq->record_commands_with_blocks(prefix_sum, 1);
  seq->launch_kernel_async();
  seq->sync();

  // print 10 u_out_idx
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_out_idx[{}] = {}", i, tmp_storage.u_out_idx[i]);
  }

  seq->record_commands(move_dups, n);
  seq->launch_kernel_async();
  seq->sync();

  const auto n_unique = tmp_storage.u_out_idx[n - 1] + 1;
  app_data_ref.set_n_unique(n_unique);
  app_data_ref.set_n_brt_nodes(n_unique - 1);

  // print 10 u_out_idx
  for (auto i = 0; i < 10; ++i) {
    spdlog::trace("u_out_idx[{}] = {}", i, tmp_storage.u_out_idx[i]);
  }
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

void Singleton::process_stage_4(tree::AppData &app_data_ref) {
  const int32_t n = app_data_ref.get_n_unique();

  auto build_radix_tree = cached_algorithms.at("build_radix_tree").get();

  build_radix_tree->update_push_constants(InputSizePushConstantsSigned{
      .n = n,
  });

  seq->record_commands(build_radix_tree, n);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

void Singleton::process_stage_5(tree::AppData &app_data_ref) {
  auto edge_count = cached_algorithms.at("edge_count").get();

  const int32_t n = app_data_ref.get_n_brt_nodes();

  edge_count->update_push_constants(InputSizePushConstantsSigned{
      .n = n,
  });

  seq->record_commands(edge_count, n);
  seq->launch_kernel_async();
  seq->sync();
}

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

void Singleton::process_stage_6(tree::AppData &app_data_ref) {
  auto prefix_sum = cached_algorithms.at("prefix_sum").get();

  const uint32_t n = app_data_ref.get_n_brt_nodes();

  prefix_sum->update_descriptor_sets({
      engine.get_buffer(app_data_ref.u_edge_count.data()),
      engine.get_buffer(app_data_ref.u_edge_offset.data()),
  });

  prefix_sum->update_push_constants(InputSizePushConstantsUnsigned{
      .n = n,
  });

  seq->record_commands_with_blocks(prefix_sum, 1);
  seq->launch_kernel_async();
  seq->sync();

  //   const auto n_octree_nodes =
  //   data.u_edge_offset->at(data.get_n_brt_nodes());

  //   spdlog::info("n_octree_nodes: {}", n_octree_nodes);
  //   data.set_n_octree_nodes(n_octree_nodes);

  const auto n_octree_nodes = app_data_ref.u_edge_offset[n - 1] + 1;
  app_data_ref.set_n_octree_nodes(n_octree_nodes);
}

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

void Singleton::process_stage_7(tree::AppData &app_data_ref) {
  auto build_octree = cached_algorithms.at("build_octree").get();

  const int32_t n = app_data_ref.get_n_brt_nodes();

  build_octree->update_push_constants(OctreePushConstants{
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
      .n_brt_nodes = n,
  });

  seq->record_commands(build_octree, n);
  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan

}  // namespace tree