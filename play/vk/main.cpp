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

  return 0;
}