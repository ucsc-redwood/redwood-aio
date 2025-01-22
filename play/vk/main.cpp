#include <spdlog/spdlog.h>

#include "../../builtin-apps/cifar-dense/arg_max.hpp"
#include "../../builtin-apps/cifar-dense/dense_appdata.hpp"
#include "../../builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"
#include "../../builtin-apps/common/vulkan/engine.hpp"

int main() {
  vulkan::Engine engine;

  // set spdlog to trace level
  spdlog::set_level(spdlog::level::trace);

  auto mr = engine.get_mr();
  cifar_dense::AppData appdata(mr);

  vulkan::Dispatcher dispatcher(engine, appdata);

  auto seq = engine.sequence();

  dispatcher.run_stage1(seq.get());
  dispatcher.run_stage2(seq.get());
  dispatcher.run_stage3(seq.get());
  dispatcher.run_stage4(seq.get());
  dispatcher.run_stage5(seq.get());
  dispatcher.run_stage6(seq.get());
  dispatcher.run_stage7(seq.get());
  dispatcher.run_stage8(seq.get());
  dispatcher.run_stage9(seq.get());

  auto arg_max_index = arg_max(appdata.u_linear_out.data());
  print_prediction(arg_max_index);

  return 0;
}