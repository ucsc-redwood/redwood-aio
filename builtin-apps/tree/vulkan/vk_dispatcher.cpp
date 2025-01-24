
#include "vk_dispatcher.hpp"

namespace tree {

namespace vulkan {

// ----------------------------------------------------------------------------
// Singleton Constructor
// ----------------------------------------------------------------------------

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.sequence()) {
  spdlog::info("Singleton instance created.");

  // tmp
  tree::AppData app_data(engine.get_mr());
}

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

void Singleton::process_stage_1(cifar_sparse::AppData &app_data) {}

}  // namespace vulkan

}  // namespace tree