#include <spdlog/spdlog.h>

#include "../common/vulkan/engine.hpp"
#include "arg_max.hpp"
#include "dense_appdata.hpp"
#include "vulkan/vk_dispatcher.hpp"

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