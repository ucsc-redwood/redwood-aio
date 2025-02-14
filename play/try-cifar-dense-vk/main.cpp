#include "builtin-apps/app.hpp"

// cifar-dense
#include "builtin-apps/cifar-dense/arg_max.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"

// cifar-sparse
#include "builtin-apps/cifar-sparse/arg_max.hpp"
#include "builtin-apps/cifar-sparse/omp/sparse_kernel.hpp"
#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"
#include "builtin-apps/cifar-sparse/vulkan/vk_dispatcher.hpp"

void run_dense_vk() {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  auto app_data = std::make_unique<cifar_dense::AppData>(mr);

  cifar_dense::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_3(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_8(*app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_9(*app_data);

  const auto max_index = cifar_dense::arg_max(app_data->u_linear_out.data());

  std::cout << "Dense VK: ";
  cifar_dense::print_prediction(max_index);

  app_data.reset();
}

void run_dense_omp() {
  auto mr = std::pmr::new_delete_resource();
  auto app_data = std::make_unique<cifar_dense::AppData>(mr);

  cifar_dense::omp::process_stage_1(*app_data);
  cifar_dense::omp::process_stage_2(*app_data);
  cifar_dense::omp::process_stage_3(*app_data);
  cifar_dense::omp::process_stage_4(*app_data);
  cifar_dense::omp::process_stage_5(*app_data);
  cifar_dense::omp::process_stage_6(*app_data);
  cifar_dense::omp::process_stage_7(*app_data);
  cifar_dense::omp::process_stage_8(*app_data);
  cifar_dense::omp::process_stage_9(*app_data);

  const auto max_index = cifar_dense::arg_max(app_data->u_linear_out.data());

  std::cout << "Dense OMP: ";
  cifar_dense::print_prediction(max_index);

  app_data.reset();
}

void run_sparse_vk() {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();
  auto app_data = std::make_unique<cifar_sparse::AppData>(mr);

  cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(*app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(*app_data);

  const auto max_index = cifar_sparse::arg_max(app_data->u_linear_output.data());

  std::cout << "Sparse VK: ";
  cifar_sparse::print_prediction(max_index);

  app_data.reset();
}

void run_sparse_omp() {
  auto mr = std::pmr::new_delete_resource();
  auto app_data = std::make_unique<cifar_sparse::AppData>(mr);

  cifar_sparse::omp::process_stage_1(*app_data);
  cifar_sparse::omp::process_stage_2(*app_data);
  cifar_sparse::omp::process_stage_3(*app_data);
  cifar_sparse::omp::process_stage_4(*app_data);
  cifar_sparse::omp::process_stage_5(*app_data);
  cifar_sparse::omp::process_stage_6(*app_data);
  cifar_sparse::omp::process_stage_7(*app_data);
  cifar_sparse::omp::process_stage_8(*app_data);
  cifar_sparse::omp::process_stage_9(*app_data);

  const auto max_index = cifar_sparse::arg_max(app_data->u_linear_output.data());

  std::cout << "Sparse OMP: ";
  cifar_sparse::print_prediction(max_index);

  app_data.reset();
}

int main(int argc, char** argv) {
   
  // run_dense_vk();
  run_dense_omp();

  run_sparse_vk();
  run_sparse_omp();

  return 0;
}