#include <omp.h>
#include <spdlog/spdlog.h>

#include <thread>

#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "tree/cuda/kernel.cuh"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

int main() {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData appdata(&mr);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; ++i) {
    tree::cuda::process_stage_1(appdata);
    tree::cuda::process_stage_2(appdata);
    tree::cuda::process_stage_3(appdata);
    tree::cuda::process_stage_4(appdata);
    tree::cuda::process_stage_5(appdata);
    tree::cuda::process_stage_6(appdata);
    tree::cuda::process_stage_7(appdata);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto avg_duration = duration.count() / 10.0;
  spdlog::info("Average duration per iteration: {}ms", avg_duration);

  // assert();
  auto is_sorted = std::ranges::is_sorted(appdata.u_morton_keys_alt);
  spdlog::info("is_sorted: {}", (is_sorted ? "true" : "false"));

  // print first 10 elements
  for (int i = 0; i < 10; ++i) {
    spdlog::info("{}", appdata.u_morton_keys[i]);
  }

  spdlog::info("n_unique: {}", appdata.get_n_unique());
  spdlog::info("n_brt_nodes: {}", appdata.get_n_brt_nodes());
  spdlog::info("n_octree_nodes: {}", appdata.get_n_octree_nodes());

  spdlog::info("Done");

  tree::cuda::cleanup();

  return 0;
}