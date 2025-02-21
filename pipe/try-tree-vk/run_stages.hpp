#pragma once

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// New Design
// ---------------------------------------------------------------------

template <int Stage>
concept ValidStage = (Stage >= 1) && (Stage <= 7);

template <int Start, int End>
concept ValidStageRange = ValidStage<Start> && ValidStage<End> && (Start <= End);

// Helper function that unfolds the stage calls.
template <int Start, int... Is>
void run_cpu_stages_impl(tree::AppData* app_data,
                         tree::omp::TmpStorage* tmp_storage,
                         std::integer_sequence<int, Is...>) {
  // Expand the calls: run_stage<Start + 0>(), run_stage<Start + 1>(), ...
  (tree::omp::run_stage<Start + Is>(*app_data, *tmp_storage), ...);
}

// Main interface
template <int Start, int End, ProcessorType processor_type, int n_threads>
  requires ValidStageRange<Start, End>
void run_cpu_stages(Task& task) {
  // Bind to the selected cores
  if constexpr (processor_type == ProcessorType::kLittleCore) {
    bind_thread_to_cores(g_little_cores);
  } else if constexpr (processor_type == ProcessorType::kMediumCore) {
    bind_thread_to_cores(g_medium_cores);
  } else if constexpr (processor_type == ProcessorType::kBigCore) {
    bind_thread_to_cores(g_big_cores);
  }

#pragma omp parallel num_threads(n_threads)
  {
    // Generate the sequence [0, 1, 2, ..., (End-Start)]
    // and expand the calls.
    run_cpu_stages_impl<Start>(
        task.app_data, task.omp_tmp_storage, std::make_integer_sequence<int, End - Start + 1>{});
  }
}

// Helper function that unfolds the stage calls.
template <int Start, int... Is>
void run_gpu_stages_impl(tree::AppData* app_data,
                         tree::vulkan::TmpStorage* tmp_storage,
                         std::integer_sequence<int, Is...>) {
  // Expand the calls: run_stage<Start + 0>(), run_stage<Start + 1>(), ...
  (tree::vulkan::Singleton::getInstance().run_stage<Start + Is>(*app_data, *tmp_storage), ...);
}

// Main interface
template <int Start, int End>
  requires ValidStageRange<Start, End>
void run_gpu_stages(Task& task) {
  // Generate the sequence [0, 1, 2, ..., (End-Start)]
  // and expand the calls.
  run_gpu_stages_impl<Start>(
      task.app_data, task.vulkan_tmp_storage, std::make_integer_sequence<int, End - Start + 1>{});
}

// ---------------------------------------------------------------------
// Old working design
// ---------------------------------------------------------------------

// template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
//   requires(start_stage <= end_stage) && (start_stage >= 1) && (end_stage <= 7)
// void run_stages(tree::AppData* app_data, tree::omp::TmpStorage* omp_tmp_storage) {
// #pragma omp parallel num_threads(num_threads)
//   {
//     // Bind to core if needed:
//     if constexpr (processor_type == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if constexpr (processor_type == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if constexpr (processor_type == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

//     // Generate a compile-time sequence for the range [start_stage, end_stage]
//     []<std::size_t... I>(
//         std::index_sequence<I...>, tree::AppData& data, tree::omp::TmpStorage& omp_tmp_storage) {
//       // Each I is offset by (start_stage - 1)
//       ((tree::omp::run_stage<start_stage + I>(data, omp_tmp_storage)), ...);
//     }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data, *omp_tmp_storage);
//   }
// }

// /**
//  * @brief Runs stages of the CIFAR dense network on GPU using Vulkan
//  *
//  * @tparam start_stage First stage to execute (must be >= 1)
//  * @tparam end_stage Last stage to execute (must be <= 9)
//  * @param app_data Pointer to application data containing network state
//  *
//  * This template function executes the specified range of network stages on the GPU using Vulkan.
//  * The stages are run in sequence using compile-time unrolling.
//  */
// template <int start_stage, int end_stage>
//   requires(start_stage <= end_stage) && (start_stage >= 1) && (end_stage <= 7)
// void run_gpu_stages(tree::AppData* app_data, tree::vulkan::TmpStorage* vulkan_tmp_storage) {
//   // Generate a compile-time sequence for the range [start_stage, end_stage]
//   []<std::size_t... I>(std::index_sequence<I...>,
//                        tree::AppData& data,
//                        tree::vulkan::TmpStorage& vulkan_tmp_storage) {
//     ((tree::vulkan::Singleton::getInstance().run_stage<start_stage + I>(data,
//     vulkan_tmp_storage)),
//      ...);
//   }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data, *vulkan_tmp_storage);
// }
