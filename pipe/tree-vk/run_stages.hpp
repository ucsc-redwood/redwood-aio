#pragma once

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/func_sort.hpp"
#include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "builtin-apps/tree/vulkan/vk_dispatcher.hpp"

template <int Start, int End>
concept ValidStageRange = Start >= 1 && End <= 9 && Start <= End;

/**
 * @brief Runs stages of the CIFAR dense network on specified processor cores with OpenMP
 * parallelization
 *
 * @tparam start_stage First stage to execute (must be >= 1)
 * @tparam end_stage Last stage to execute (must be <= 9)
 * @tparam processor_type Type of processor core to run on (kLittleCore, kMediumCore, or kBigCore)
 * @tparam num_threads Number of OpenMP threads to use
 * @param app_data Pointer to application data containing network state
 *
 * This template function executes the specified range of network stages using OpenMP
 * parallelization. It binds threads to the appropriate processor cores based on processor_type and
 * runs the stages in sequence using compile-time unrolling.
 */
template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
  requires ValidStageRange<start_stage, end_stage>
void run_stages(tree::AppData* app_data, tree::omp::TempStorage& temp_storage) {
#pragma omp parallel num_threads(num_threads)
  {
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    } else {
      assert(false);
    }

    // Generate a compile-time sequence for the range [start_stage, end_stage]
    [&temp_storage]<std::size_t... I>(std::index_sequence<I...>, tree::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((tree::omp::run_stage<start_stage + I>(data, temp_storage)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
  }
}

/**
 * @brief Runs stages of the CIFAR dense network on GPU using Vulkan
 *
 * @tparam start_stage First stage to execute (must be >= 1)
 * @tparam end_stage Last stage to execute (must be <= 9)
 * @param app_data Pointer to application data containing network state
 *
 * This template function executes the specified range of network stages on the GPU using Vulkan.
 * The stages are run in sequence using compile-time unrolling.
 */
template <int start_stage, int end_stage>
  requires ValidStageRange<start_stage, end_stage>
void run_gpu_stages(tree::AppData* app_data, tree::vulkan::TmpStorage& temp_storage) {
  // Generate a compile-time sequence for the range [start_stage, end_stage]
  [&temp_storage]<std::size_t... I>(std::index_sequence<I...>, tree::AppData& data) {
    ((tree::vulkan::Singleton::getInstance().run_stage<start_stage + I>(data, temp_storage)), ...);
  }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
}
