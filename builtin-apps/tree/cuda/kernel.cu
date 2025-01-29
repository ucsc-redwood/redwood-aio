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

// Caching allocator for device memory
cub::CachingDeviceAllocator g_allocator(true);

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void process_stage_1(AppData &app_data) {
  constexpr auto block_size = 256;
  const auto grid_size =
      cub::DivideAndRoundUp(app_data.get_n_input(), block_size);
  constexpr auto s_mem = 0;

  ::cuda::kernels::k_ComputeMortonCode<<<grid_size, block_size, s_mem>>>(
      app_data.u_input_points.data(),
      app_data.u_morton_keys.data(),
      app_data.get_n_input(),
      app_data.min_coord,
      app_data.range);
}

// ----------------------------------------------------------------------------
// Stage 2 (sort) (morton code -> sorted morton code)
// ----------------------------------------------------------------------------

void process_stage_2(AppData &app_data) {
  uint32_t *d_keys_in = app_data.u_morton_keys.data();
  uint32_t *d_keys_out = app_data.u_morton_keys_alt.data();
  uint32_t num_items = app_data.get_n_input();

  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Get temporary storage size
  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Sort data
  cub::DeviceRadixSort::SortKeys(
      d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);

  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

// ----------------------------------------------------------------------------
// Stage 3 (unique) (sorted morton code -> unique sorted morton code)
// ----------------------------------------------------------------------------

void process_stage_3(AppData &app_data) {
  uint32_t *d_in = app_data.u_morton_keys_alt.data();
  uint32_t *d_out = app_data.u_morton_keys.data();
  uint32_t num_items = app_data.get_n_input();

  uint32_t *u_num_selected_out = nullptr;
  CUDA_CHECK(cudaMallocManaged(&u_num_selected_out, sizeof(uint32_t)));

  // Allocate temporary storage
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CubDebugExit(cub::DeviceSelect::Unique(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         u_num_selected_out,
                                         num_items));
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Run
  CubDebugExit(cub::DeviceSelect::Unique(d_temp_storage,
                                         temp_storage_bytes,
                                         d_in,
                                         d_out,
                                         u_num_selected_out,
                                         num_items));

  CUDA_CHECK(cudaDeviceSynchronize());

  const auto n_unique = u_num_selected_out[0];
  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);

  CUDA_CHECK(cudaFree(u_num_selected_out));
  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

// ----------------------------------------------------------------------------
// Stage 4 (build tree) (unique sorted morton code -> tree nodes)
// ----------------------------------------------------------------------------

void process_stage_4(AppData &app_data) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_BuildRadixTree<<<gridDim, blockDim, sharedMem>>>(
      app_data.get_n_unique(),
      app_data.get_unique_morton_keys(),
      app_data.brt.u_prefix_n.data(),
      app_data.brt.u_has_leaf_left.data(),
      app_data.brt.u_has_leaf_right.data(),
      app_data.brt.u_left_child.data(),
      app_data.brt.u_parents.data());
}

// ----------------------------------------------------------------------------
// Stage 5 (edge count) (tree nodes -> edge count)
// ----------------------------------------------------------------------------

void process_stage_5(AppData &app_data) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_EdgeCount<<<gridDim, blockDim, sharedMem>>>(
      app_data.brt.u_prefix_n.data(),
      app_data.brt.u_parents.data(),
      app_data.u_edge_count.data(),
      app_data.get_n_brt_nodes());
}

// ----------------------------------------------------------------------------
// Stage 6 (edge offset) (edge count -> edge offset)
// ----------------------------------------------------------------------------

void process_stage_6(AppData &app_data) {
  // Temporary storage for CUB
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Determine temporary storage size
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                app_data.u_edge_count.data(),
                                app_data.u_edge_offset.data(),
                                app_data.get_n_brt_nodes());
  CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

  // Perform prefix sum (inclusive scan)
  cub::DeviceScan::InclusiveSum(d_temp_storage,
                                temp_storage_bytes,
                                app_data.u_edge_count.data(),
                                app_data.u_edge_offset.data(),
                                app_data.get_n_brt_nodes());

  // Sync
  CUDA_CHECK(cudaDeviceSynchronize());

  // num oct is the result of last of prefix sum
  app_data.set_n_octree_nodes(
      app_data.u_edge_offset[app_data.get_n_brt_nodes() - 1]);

  CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
}

// ----------------------------------------------------------------------------
// Stage 7 (octree) (everything above -> octree)
// ----------------------------------------------------------------------------

void process_stage_7(AppData &app_data) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_MakeOctNodes<<<gridDim, blockDim, sharedMem>>>(
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
      app_data.range,
      app_data.get_n_brt_nodes());
}

}  // namespace tree::cuda
