#include <omp.h>

#include <affinity.hpp>
#include <array>
#include <cifar_dense_kernel.hpp>
#include <cifar_sparse_kernel.hpp>

#include "arg_max.hpp"

std::array<int, 6> core_ids = {0, 1, 2, 3, 4, 5};

void run_sparse(cifar_sparse::AppData& appdata, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    auto tid = omp_get_thread_num();
    auto core_id = core_ids[tid % core_ids.size()];

    bind_thread_to_core(core_id);

    cifar_sparse::omp::process_stage_1(appdata);
    cifar_sparse::omp::process_stage_2(appdata);
    cifar_sparse::omp::process_stage_3(appdata);
    cifar_sparse::omp::process_stage_4(appdata);
    cifar_sparse::omp::process_stage_5(appdata);
    cifar_sparse::omp::process_stage_6(appdata);
    cifar_sparse::omp::process_stage_7(appdata);
    cifar_sparse::omp::process_stage_8(appdata);
    cifar_sparse::omp::process_stage_9(appdata);
  }
}

void run_dense(cifar_dense::AppData& appdata, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    cifar_dense::omp::process_stage_1(appdata);
    cifar_dense::omp::process_stage_2(appdata);
    cifar_dense::omp::process_stage_3(appdata);
    cifar_dense::omp::process_stage_4(appdata);
    cifar_dense::omp::process_stage_5(appdata);
    cifar_dense::omp::process_stage_6(appdata);
    cifar_dense::omp::process_stage_7(appdata);
    cifar_dense::omp::process_stage_8(appdata);
    cifar_dense::omp::process_stage_9(appdata);
  }
}

int main() {
  auto mr = std::pmr::new_delete_resource();

  for (int n_threads = 1; n_threads <= 6; ++n_threads) {
    std::cout << "\nRunning with " << n_threads << " threads:" << std::endl;

    {
      cifar_dense::AppData appdata(mr);

      const auto start = omp_get_wtime();
      run_dense(appdata, n_threads);
      const auto end = omp_get_wtime();

      std::cout << "Dense execution time: " << (end - start) * 1000 << " ms"
                << std::endl;
      print_prediction(arg_max(appdata.u_linear_out.data()));
    }

    {
      cifar_sparse::AppData appdata(mr);

      const auto start = omp_get_wtime();
      run_sparse(appdata, n_threads);
      const auto end = omp_get_wtime();

      std::cout << "Sparse execution time: " << (end - start) * 1000 << " ms"
                << std::endl;
      print_prediction(arg_max(appdata.u_linear_output.data()));
    }
  }

  return 0;
}
