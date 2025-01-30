#include <omp.h>

#include "arg_max.hpp"
#include "dense_appdata.hpp"
#include "omp/dense_kernel.hpp"

int main(int argc, char* argv[]) {
  bool use_omp = false;
  if (argc > 1) {
    std::string_view arg(argv[1]);
    use_omp = (arg == "true" || arg == "1");
  }

  auto mr = std::pmr::get_default_resource();
  cifar_dense::AppData appdata(mr);

  if (use_omp) {
#pragma omp parallel
    for (int i = 0; i < 10; ++i) {
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

  } else {
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

  auto arg_max_index = arg_max(appdata.u_linear_out.data());
  print_prediction(arg_max_index);

  return 0;
}