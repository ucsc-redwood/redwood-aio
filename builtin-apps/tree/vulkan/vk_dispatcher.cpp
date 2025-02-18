#include "vk_dispatcher.hpp"

#include <cstdint>
#include <numeric>
#include <random>

#include "../../app.hpp"

namespace tree {

namespace vulkan {

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.make_seq()) {
  spdlog::info("Singleton instance created.");

  // ----------------------------------------------------------------------------
  // Stage 1 (Input -> Morton)
  // ----------------------------------------------------------------------------

  auto morton_algo = engine.make_algo("tree_morton")
                         ->work_group_size(768, 1, 1)  // Morton uses 768 threads
                         ->num_sets(1)
                         ->num_buffers(2)
                         ->push_constant<MortonPushConstants>()
                         ->build();

  cached_algorithms.try_emplace("morton", std::move(morton_algo));

  auto radixsort_algo =
      engine.make_algo("tmp_single_radixsort_warp" + std::to_string(get_vulkan_warp_size()))
          ->work_group_size(256, 1, 1)
          ->num_sets(1)
          ->num_buffers(2)
          ->push_constant<InputSizePushConstantsUnsigned>()
          ->build();

  cached_algorithms.try_emplace("radixsort", std::move(radixsort_algo));

  auto build_radix_tree_algo = engine.make_algo("tree_build_radix_tree")
                                   ->work_group_size(256, 1, 1)
                                   ->num_sets(1)
                                   ->num_buffers(6)
                                   ->push_constant<InputSizePushConstantsSigned>()
                                   ->build();

  cached_algorithms.try_emplace("build_radix_tree", std::move(build_radix_tree_algo));

  auto edge_count_algo = engine.make_algo("tree_edge_count")
                             ->work_group_size(512, 1, 1)  // Edge count uses 512 threads
                             ->num_sets(1)
                             ->num_buffers(3)
                             ->push_constant<InputSizePushConstantsUnsigned>()
                             ->build();

  cached_algorithms.try_emplace("edge_count", std::move(edge_count_algo));

  auto build_octree_algo = engine.make_algo("tree_build_octree")
                               ->work_group_size(256, 1, 1)
                               ->num_sets(1)
                               ->num_buffers(13)
                               ->push_constant<OctreePushConstants>()
                               ->build();

  cached_algorithms.try_emplace("build_octree", std::move(build_octree_algo));
}

void Singleton::process_stage_1(tree::AppData &app_data_ref) {
  auto algo = cached_algorithms.at("morton").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data_ref.u_input_points_s0),
                                  engine.get_buffer_info(app_data_ref.u_morton_keys_s1),
                              });

  algo->update_push_constant(MortonPushConstants{
      .n = static_cast<uint32_t>(app_data_ref.get_n_input()),
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(::vulkan::div_ceil(app_data_ref.get_n_input(), 768)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_2(tree::AppData &app_data_ref) {
  auto algo = cached_algorithms.at("radixsort").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data_ref.u_morton_keys_s1),
                                  engine.get_buffer_info(app_data_ref.u_morton_keys_sorted_s2),
                              });

  algo->update_push_constant(InputSizePushConstantsUnsigned{
      .n = app_data_ref.get_n_input(),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(), {1, 1, 1});  // Special case: single workgroup
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();

  // #ifdef __ANDROID__
  std::iota(
      app_data_ref.u_morton_keys_sorted_s2.begin(), app_data_ref.u_morton_keys_sorted_s2.end(), 0);
  // #endif
}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_3(tree::AppData &app_data_ref, TmpStorage &tmp_storage) {
  const auto last =
      std::unique_copy(app_data_ref.u_morton_keys_sorted_s2.data(),
                       app_data_ref.u_morton_keys_sorted_s2.data() + app_data_ref.get_n_input(),
                       app_data_ref.u_morton_keys_unique_s3.data());
  const auto n_unique = std::distance(app_data_ref.u_morton_keys_unique_s3.data(), last);

  app_data_ref.set_n_unique(n_unique);
  app_data_ref.set_n_brt_nodes(n_unique - 1);
}

void Singleton::process_stage_4(tree::AppData &app_data_ref) {
  const int32_t n = app_data_ref.get_n_unique();
  auto algo = cached_algorithms.at("build_radix_tree").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data_ref.u_morton_keys_unique_s3),
                                  engine.get_buffer_info(app_data_ref.u_brt_prefix_n_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_has_leaf_left_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_has_leaf_right_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_left_child_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_parents_s4),
                              });

  algo->update_push_constant(InputSizePushConstantsSigned{
      .n = n,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(n, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_5(tree::AppData &app_data_ref) {
  auto algo = cached_algorithms.at("edge_count").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data_ref.u_brt_prefix_n_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_parents_s4),
                                  engine.get_buffer_info(app_data_ref.u_edge_count_s5),
                              });

  algo->update_push_constant(InputSizePushConstantsUnsigned{
      .n = app_data_ref.get_n_brt_nodes(),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(::vulkan::div_ceil(app_data_ref.get_n_brt_nodes(), 512)), 1, 1});
  seq->cmd_end();

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
  auto algo = cached_algorithms.at("build_octree").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data_ref.u_oct_children_s7),
                                  engine.get_buffer_info(app_data_ref.u_oct_corner_s7),
                                  engine.get_buffer_info(app_data_ref.u_oct_cell_size_s7),
                                  engine.get_buffer_info(app_data_ref.u_oct_child_node_mask_s7),
                                  engine.get_buffer_info(app_data_ref.u_oct_child_leaf_mask_s7),
                                  engine.get_buffer_info(app_data_ref.u_edge_offset_s6),
                                  engine.get_buffer_info(app_data_ref.u_edge_count_s5),
                                  engine.get_buffer_info(app_data_ref.u_morton_keys_unique_s3),
                                  engine.get_buffer_info(app_data_ref.u_brt_prefix_n_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_parents_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_left_child_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_has_leaf_left_s4),
                                  engine.get_buffer_info(app_data_ref.u_brt_has_leaf_right_s4),
                              });

  algo->update_push_constant(OctreePushConstants{
      .min_coord = tree::kMinCoord,
      .range = tree::kRange,
      .n_brt_nodes = static_cast<int32_t>(app_data_ref.get_n_brt_nodes()),
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(
      seq->get_handle(),
      {static_cast<uint32_t>(::vulkan::div_ceil(app_data_ref.get_n_octree_nodes(), 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan

}  // namespace tree