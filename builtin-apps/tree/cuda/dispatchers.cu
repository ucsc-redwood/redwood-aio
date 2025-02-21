#include <spdlog/spdlog.h>

#include <cub/cub.cuh>
#include <cub/util_math.cuh>

#include "../../common/cuda/helpers.cuh"
#include "01_morton.cuh"
#include "04_radix_tree.cuh"
#include "05_edge_count.cuh"
#include "07_octree.cuh"
#include "kernel.cuh"

namespace tree::cuda {

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

cub::CachingDeviceAllocator g_allocator(true);  // Caching allocator for device memory

TempStorage::TempStorage() {
  CubDebugExit(cudaMallocManaged(&u_num_selected_out, sizeof(uint32_t)));
}

TempStorage::~TempStorage() {
  if (u_num_selected_out) {
    CubDebugExit(cudaFree(u_num_selected_out));
  }

  if (sort.d_temp_storage) {
    CubDebugExit(cudaFree(sort.d_temp_storage));
  }

  if (unique.d_temp_storage) {
    CubDebugExit(cudaFree(unique.d_temp_storage));
  }

  if (prefix_sum.d_temp_storage) {
    CubDebugExit(cudaFree(prefix_sum.d_temp_storage));
  }
}

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void process_stage_1(AppData &app_data, [[maybe_unused]] TempStorage &tmp) {
  constexpr auto block_size = 256;
  const auto grid_size = cub::DivideAndRoundUp(app_data.get_n_input(), block_size);
  constexpr auto s_mem = 0;

  ::cuda::kernels::k_ComputeMortonCode<<<grid_size, block_size, s_mem>>>(
      app_data.u_input_points_s0.data(),
      app_data.u_morton_keys_s1.data(),
      app_data.get_n_input(),
      tree::kMinCoord,
      tree::kRange);
}

// ----------------------------------------------------------------------------
// Stage 2 (sort) (morton code -> sorted morton code)
// ----------------------------------------------------------------------------

void process_stage_2(AppData &app_data, TempStorage &tmp) {
  uint32_t *d_keys_in = app_data.u_morton_keys_s1.data();
  uint32_t *d_keys_out = app_data.u_morton_keys_sorted_s2.data();
  uint32_t num_items = app_data.get_n_input();

  // Get temporary storage size

  cub::DeviceRadixSort::SortKeys(
      tmp.sort.d_temp_storage, tmp.sort.temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  CubDebugExit(g_allocator.DeviceAllocate(&tmp.sort.d_temp_storage, tmp.sort.temp_storage_bytes));

  // Sort data
  cub::DeviceRadixSort::SortKeys(
      tmp.sort.d_temp_storage, tmp.sort.temp_storage_bytes, d_keys_in, d_keys_out, num_items);
}

// ----------------------------------------------------------------------------
// Stage 3 (unique) (sorted morton code -> unique sorted morton code)
// ----------------------------------------------------------------------------

void process_stage_3(AppData &app_data, TempStorage &tmp) {
  uint32_t *d_in = app_data.u_morton_keys_sorted_s2.data();
  uint32_t *d_out = app_data.u_morton_keys_unique_s3.data();
  uint32_t num_items = app_data.get_n_input();

  // Allocate temporary storage
  CubDebugExit(cub::DeviceSelect::Unique(tmp.unique.d_temp_storage,
                                         tmp.unique.temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         tmp.u_num_selected_out,
                                         num_items));

  CubDebugExit(
      g_allocator.DeviceAllocate(&tmp.unique.d_temp_storage, tmp.unique.temp_storage_bytes));

  // Run
  CubDebugExit(cub::DeviceSelect::Unique(tmp.unique.d_temp_storage,
                                         tmp.unique.temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         tmp.u_num_selected_out,
                                         num_items));

  CubDebugExit(cudaDeviceSynchronize());

  // -------- host --------------
  const auto n_unique = tmp.u_num_selected_out[0];
  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);
  // ----------------------------
}

// ----------------------------------------------------------------------------
// Stage 4 (build tree) (unique sorted morton code -> tree nodes)
// ----------------------------------------------------------------------------

void process_stage_4(AppData &app_data, [[maybe_unused]] TempStorage &tmp) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_BuildRadixTree<<<gridDim, blockDim, sharedMem>>>(
      app_data.get_n_unique(),
      app_data.u_morton_keys_unique_s3.data(),
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_has_leaf_left_s4.data(),
      app_data.u_brt_has_leaf_right_s4.data(),
      app_data.u_brt_left_child_s4.data(),
      app_data.u_brt_parents_s4.data());
}

// ----------------------------------------------------------------------------
// Stage 5 (edge count) (tree nodes -> edge count)
// ----------------------------------------------------------------------------

void process_stage_5(AppData &app_data, [[maybe_unused]] TempStorage &tmp) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_EdgeCount<<<gridDim, blockDim, sharedMem>>>(app_data.u_brt_prefix_n_s4.data(),
                                                                 app_data.u_brt_parents_s4.data(),
                                                                 app_data.u_edge_count_s5.data(),
                                                                 app_data.get_n_brt_nodes());
}

// ----------------------------------------------------------------------------
// Stage 6 (edge offset) (edge count -> edge offset)
// ----------------------------------------------------------------------------

void process_stage_6(AppData &app_data, TempStorage &tmp) {
  cub::DeviceScan::InclusiveSum(tmp.prefix_sum.d_temp_storage,
                                tmp.prefix_sum.temp_storage_bytes,
                                app_data.u_edge_count_s5.data(),
                                app_data.u_edge_offset_s6.data(),
                                app_data.get_n_brt_nodes());

  CubDebugExit(g_allocator.DeviceAllocate(&tmp.prefix_sum.d_temp_storage,
                                          tmp.prefix_sum.temp_storage_bytes));

  // Perform prefix sum (inclusive scan)
  cub::DeviceScan::InclusiveSum(tmp.prefix_sum.d_temp_storage,
                                tmp.prefix_sum.temp_storage_bytes,
                                app_data.u_edge_count_s5.data(),
                                app_data.u_edge_offset_s6.data(),
                                app_data.get_n_brt_nodes());

  CubDebugExit(cudaDeviceSynchronize());

  // -------- host --------------
  app_data.set_n_octree_nodes(app_data.u_edge_offset_s6[app_data.get_n_brt_nodes() - 1]);
  // ----------------------------
}

// ----------------------------------------------------------------------------
// Stage 7 (octree) (everything above -> octree)
// ----------------------------------------------------------------------------

void process_stage_7(AppData &app_data, [[maybe_unused]] TempStorage &tmp) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_MakeOctNodes<<<gridDim, blockDim, sharedMem>>>(
      reinterpret_cast<int(*)[8]>(app_data.u_oct_children_s7.data()),
      app_data.u_oct_corner_s7.data(),
      app_data.u_oct_cell_size_s7.data(),
      app_data.u_oct_child_node_mask_s7.data(),
      app_data.u_edge_offset_s6.data(),
      app_data.u_edge_count_s5.data(),
      app_data.u_morton_keys_unique_s3.data(),
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_parents_s4.data(),
      tree::kMinCoord,
      tree::kRange,
      app_data.get_n_brt_nodes());
}

}  // namespace tree::cuda
