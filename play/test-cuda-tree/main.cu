#include <omp.h>
#include <spdlog/spdlog.h>

#include <thread>

#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

int main() {
  auto mr = cuda::CudaMemoryResource();
  auto appdata = new tree::AppData(&mr);

  const auto n_threads = std::thread::hardware_concurrency();
  tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);

#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, temp_storage);
    tree::omp::process_stage_3(*appdata);
    tree::omp::process_stage_4(*appdata);
    tree::omp::process_stage_5(*appdata);
    tree::omp::process_stage_6(*appdata);
    tree::omp::process_stage_7(*appdata);
  }

  spdlog::info("Done");

  delete appdata;
  return 0;
}