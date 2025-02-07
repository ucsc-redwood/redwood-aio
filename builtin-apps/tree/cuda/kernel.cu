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

// Global temporary storage for different CUB operations
void *g_sort_temp_storage = nullptr;
size_t g_sort_temp_bytes = 0;

void *g_unique_temp_storage = nullptr;
size_t g_unique_temp_bytes = 0;
uint32_t *g_num_selected_out = nullptr;

void *g_scan_temp_storage = nullptr;
size_t g_scan_temp_bytes = 0;

void cleanup() {
  if (g_sort_temp_storage) {
    CUDA_CHECK(cudaFree(g_sort_temp_storage));
  }
  if (g_unique_temp_storage) {
    CUDA_CHECK(cudaFree(g_unique_temp_storage));
  }
  if (g_scan_temp_storage) {
    CUDA_CHECK(cudaFree(g_scan_temp_storage));
  }
  if (g_num_selected_out) {
    CUDA_CHECK(cudaFree(g_num_selected_out));
  }
}

void warmup(AppData &app_data) {
  process_stage_1(app_data);
  process_stage_2(app_data);
  process_stage_3(app_data);
  process_stage_4(app_data);
  process_stage_5(app_data);
  process_stage_6(app_data);
  process_stage_7(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void process_stage_1(AppData &app_data) {
  constexpr auto block_size = 256;
  const auto grid_size =
      cub::DivideAndRoundUp(app_data.get_n_input(), block_size);
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

void process_stage_2(AppData &app_data) {
  uint32_t *d_keys_in = app_data.u_morton_keys_s1.data();
  uint32_t *d_keys_out = app_data.u_morton_keys_sorted_s2.data();
  uint32_t num_items = app_data.get_n_input();

  // Get temporary storage size
  if (g_sort_temp_storage == nullptr) {
    cub::DeviceRadixSort::SortKeys(g_sort_temp_storage,
                                   g_sort_temp_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   num_items);
    CUDA_CHECK(cudaMalloc(&g_sort_temp_storage, g_sort_temp_bytes));
  }

  // Sort data
  cub::DeviceRadixSort::SortKeys(
      g_sort_temp_storage, g_sort_temp_bytes, d_keys_in, d_keys_out, num_items);
}

// ----------------------------------------------------------------------------
// Stage 3 (unique) (sorted morton code -> unique sorted morton code)
// ----------------------------------------------------------------------------

void process_stage_3(AppData &app_data) {
  uint32_t *d_in = app_data.u_morton_keys_sorted_s2.data();
  uint32_t *d_out = app_data.u_morton_keys_unique_s3.data();
  uint32_t num_items = app_data.get_n_input();

  if (g_num_selected_out == nullptr) {
    CUDA_CHECK(cudaMallocManaged(&g_num_selected_out, sizeof(uint32_t)));

    // Allocate temporary storage
    CubDebugExit(cub::DeviceSelect::Unique(g_unique_temp_storage,
                                           g_unique_temp_bytes,
                                           d_in,
                                           d_out,
                                           g_num_selected_out,
                                           num_items));
    CUDA_CHECK(cudaMalloc(&g_unique_temp_storage, g_unique_temp_bytes));
  }

  // Run
  CubDebugExit(cub::DeviceSelect::Unique(g_unique_temp_storage,
                                         g_unique_temp_bytes,
                                         d_in,
                                         d_out,
                                         g_num_selected_out,
                                         num_items));

  CUDA_CHECK(cudaDeviceSynchronize());

  const auto n_unique = g_num_selected_out[0];
  app_data.set_n_unique(n_unique);
  app_data.set_n_brt_nodes(n_unique - 1);
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

void process_stage_5(AppData &app_data) {
  constexpr auto gridDim = 16;
  constexpr auto blockDim = 512;
  constexpr auto sharedMem = 0;

  ::cuda::kernels::k_EdgeCount<<<gridDim, blockDim, sharedMem>>>(
      app_data.u_brt_prefix_n_s4.data(),
      app_data.u_brt_parents_s4.data(),
      app_data.u_edge_count_s5.data(),
      app_data.get_n_brt_nodes());
}

// ----------------------------------------------------------------------------
// Stage 6 (edge offset) (edge count -> edge offset)
// ----------------------------------------------------------------------------

void process_stage_6(AppData &app_data) {
  if (g_scan_temp_storage == nullptr) {
    cub::DeviceScan::InclusiveSum(g_scan_temp_storage,
                                  g_scan_temp_bytes,
                                  app_data.u_edge_count_s5.data(),
                                  app_data.u_edge_offset_s6.data(),
                                  app_data.get_n_brt_nodes());
    CUDA_CHECK(cudaMalloc(&g_scan_temp_storage, g_scan_temp_bytes));
  }

  // Perform prefix sum (inclusive scan)
  cub::DeviceScan::InclusiveSum(g_scan_temp_storage,
                                g_scan_temp_bytes,
                                app_data.u_edge_count_s5.data(),
                                app_data.u_edge_offset_s6.data(),
                                app_data.get_n_brt_nodes());

  CUDA_CHECK(cudaDeviceSynchronize());

  app_data.set_n_octree_nodes(
      app_data.u_edge_offset_s6[app_data.get_n_brt_nodes() - 1]);
}

// ----------------------------------------------------------------------------
// Stage 7 (octree) (everything above -> octree)
// ----------------------------------------------------------------------------

void process_stage_7(AppData &app_data) {
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
