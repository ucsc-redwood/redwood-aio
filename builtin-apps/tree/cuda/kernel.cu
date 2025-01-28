#include <spdlog/spdlog.h>

#include <cub/util_math.cuh>
#include <numeric>

#include "01_morton.cuh"
#include "02_sort.cuh"
#include "03_unique.cuh"
#include "04_radix_tree.cuh"
#include "05_edge_count.cuh"
#include "06_prefix_sum.cuh"
#include "07_octree.cuh"
#include "kernel.cuh"

namespace tree::cuda {

// ----------------------------------------------------------------------------
// Stage 1 (input -> morton code)
// ----------------------------------------------------------------------------

void process_stage_1(AppData &app_data) {
  static constexpr auto block_size = 256;
  const auto grid_size = div_up(app_data.get_n_input(), block_size);
  constexpr auto s_mem = 0;

  spdlog::debug(
      "CUDA kernel 'compute_morton_code', n = {}, threads = {}, blocks = {}, "
      "stream: {}",
      app_data.get_n_input(),
      block_size,
      grid_size,
      reinterpret_cast<void *>(stream));

  ::cuda::kernels::k_ComputeMortonCode<<<grid_size, block_size, s_mem>>>(
      app_data.u_input_points.data(),
      app_data.u_morton_keys.data(),
      app_data.get_n_input(),
      app_data.min_coord,
      app_data.range);
}

void process_stage_2(AppData &app_data) {}

void process_stage_3(AppData &app_data) {}

void process_stage_4(AppData &app_data) {}

void process_stage_5(AppData &app_data) {}

void process_stage_6(AppData &app_data) {}

void process_stage_7(AppData &app_data) {}

void device_sync();

}  // namespace tree::cuda
