
#include "vk_dispatcher.hpp"

#include <cstdint>
#include <numeric>
#include <random>

#include "../../app.hpp"

namespace tree {

namespace vulkan {

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.sequence()) {
  spdlog::info("Singleton instance created.");
}

// ----------------------------------------------------------------------------
// Stage 1 (Input -> Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_1(tree::AppData &app_data_ref) {}

// ----------------------------------------------------------------------------
// Stage 2 (Morton -> Sorted Morton)
// ----------------------------------------------------------------------------
struct Ps {
  uint32_t n;
};

void Singleton::process_stage_2(tree::AppData &app_data_ref) {}

// ----------------------------------------------------------------------------
// Stage 3 (Sorted Morton -> Unique Sorted Morton)
// ----------------------------------------------------------------------------

void Singleton::process_stage_3(tree::AppData &app_data_ref,
                                TmpStorage &tmp_storage) {}

// ----------------------------------------------------------------------------
// Stage 4 (Unique Sorted Morton -> Radix Tree)
// ----------------------------------------------------------------------------

void Singleton::process_stage_4(tree::AppData &app_data_ref) {}

// ----------------------------------------------------------------------------
// Stage 5 (Radix Tree -> Edge Count)
// ----------------------------------------------------------------------------

void Singleton::process_stage_5(tree::AppData &app_data_ref) {}

// ----------------------------------------------------------------------------
// Stage 6 (Edge Count -> Edge Offset, prefix sum)
// ----------------------------------------------------------------------------

void Singleton::process_stage_6(tree::AppData &app_data_ref) {}

//----------------------------------------------------------------------------
// Stage 7 (Edge Offset -> Octree)
//----------------------------------------------------------------------------

void Singleton::process_stage_7(tree::AppData &app_data_ref) {}

}  // namespace vulkan

}  // namespace tree