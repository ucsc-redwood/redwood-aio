#include <spdlog/spdlog.h>

#include <algorithm>
#include <memory_resource>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../builtin-apps/tree/omp/tree_kernel.hpp"
#include "../../builtin-apps/tree/tree_appdata.hpp"
#include "../../pipe/app.hpp"

void run_normal(const int n_threads) {
  auto mr = std::pmr::new_delete_resource();

  tree::AppData app_data(mr);
  tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);

  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(n_threads)
  {
    tree::omp::process_stage_1(app_data);
    tree::omp::v2::process_stage_2(app_data, temp_storage);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "duration: " << duration.count() << "ms" << std::endl;

  //   std::cout << "n_octree_nodes: " << app_data.get_n_octree_nodes() <<
  //   std::endl;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  for (int i = 1; i <= 8; ++i) {
    run_normal(i);
  }

  return 0;
}