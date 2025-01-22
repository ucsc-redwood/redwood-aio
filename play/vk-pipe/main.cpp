#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../../builtin-apps/cifar-dense/arg_max.hpp"
#include "../../builtin-apps/cifar-dense/dense_appdata.hpp"
#include "../../builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"
#include "../../builtin-apps/common/vulkan/engine.hpp"
#include "../../pipe/affinity.hpp"
#include "../../pipe/app.hpp"
#include "../../pipe/conf.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  //   run_2_stages();
  spdlog::set_level(spdlog::level::trace);

  auto mr = vulkan::Singleton::getInstance().get_mr();

  cifar_dense::AppData app_data(mr);

  vulkan::Singleton::getInstance().process_stage_1(app_data);
  vulkan::Singleton::getInstance().process_stage_2(app_data);
  vulkan::Singleton::getInstance().process_stage_3(app_data);
  vulkan::Singleton::getInstance().process_stage_4(app_data);
  vulkan::Singleton::getInstance().process_stage_5(app_data);
  vulkan::Singleton::getInstance().process_stage_6(app_data);
  vulkan::Singleton::getInstance().process_stage_7(app_data);
  vulkan::Singleton::getInstance().process_stage_8(app_data);
  vulkan::Singleton::getInstance().process_stage_9(app_data);

  auto arg_max_index = arg_max(app_data.u_linear_out.data());
  print_prediction(arg_max_index);

  return 0;
}
