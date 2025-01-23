#include <spdlog/spdlog.h>

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

#pragma omp parallel num_threads(n_threads)
  {
    tree::omp::process_stage_1(app_data);
    tree::omp::v2::process_stage_2(app_data, temp_storage);
    tree::omp::process_stage_3(app_data);
    tree::omp::process_stage_4(app_data);
    tree::omp::process_stage_5(app_data);
    tree::omp::process_stage_6(app_data);
    tree::omp::process_stage_7(app_data);
  }

  std::cout << "done" << std::endl;
  std::cout << "n_octree_nodes: " << app_data.get_n_octree_nodes() << std::endl;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  run_normal(4);

  return 0;
}