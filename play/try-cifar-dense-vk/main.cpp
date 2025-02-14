#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/arg_max.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"

void run_vk() {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  cifar_dense::AppData app_data(mr);

  cifar_dense::vulkan::Singleton::getInstance().process_stage_1(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_2(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_3(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_4(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_5(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_6(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_7(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_8(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_9(app_data);

  const auto max_index = arg_max(app_data.u_linear_out.data());

  std::cout << "VK: ";
  print_prediction(max_index);
}

void run_omp() {
  auto mr = std::pmr::new_delete_resource();
  cifar_dense::AppData app_data(mr);

  cifar_dense::omp::process_stage_1(app_data);
  cifar_dense::omp::process_stage_2(app_data);
  cifar_dense::omp::process_stage_3(app_data);
  cifar_dense::omp::process_stage_4(app_data);
  cifar_dense::omp::process_stage_5(app_data);
  cifar_dense::omp::process_stage_6(app_data);
  cifar_dense::omp::process_stage_7(app_data);
  cifar_dense::omp::process_stage_8(app_data);
  cifar_dense::omp::process_stage_9(app_data);

  const auto max_index = arg_max(app_data.u_linear_out.data());

  std::cout << "OMP: ";
  print_prediction(max_index);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  run_vk();
  run_omp();

  return 0;
}