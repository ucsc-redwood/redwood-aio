#pragma once

#include "../concepts.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/cuda/dispatchers.cuh"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// CPU stages
// ---------------------------------------------------------------------

template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
  requires ValidStageRange<start_stage, end_stage> && ValidProcessorType<processor_type>
void run_cpu_stages(Task& task) {
#pragma omp parallel num_threads(num_threads)
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    }

    // Generate a compile-time sequence for the range [start_stage, end_stage]
    []<std::size_t... I>(
        std::index_sequence<I...>, tree::AppData& data, tree::omp::TmpStorage& tmp_storage) {
      // Each I is offset by (start_stage - 1)
      ((tree::omp::run_stage<start_stage + I>(data, tmp_storage)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{},
      *task.app_data,
      *task.omp_tmp_storage);
  }
}

// ---------------------------------------------------------------------
// GPU stages
// ---------------------------------------------------------------------

template <int start_stage, int end_stage>
  requires ValidStageRange<start_stage, end_stage>
void run_gpu_stages(Task& task) {
  // Generate a compile-time sequence for the range [start_stage, end_stage]
  []<std::size_t... I>(
      std::index_sequence<I...>, tree::AppData& data, tree::cuda::TempStorage& tmp_storage) {
    ((tree::cuda::run_stage<start_stage + I>(data, tmp_storage)), ...);
  }(std::make_index_sequence<end_stage - start_stage + 1>{},
    *task.app_data,
    *task.cuda_tmp_storage);
}

// tree::cuda::TempStorage tmp_storage;
// tree::cuda::TempStorage tmp_storage;

// template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
//   requires ValidStageRange<start_stage, end_stage>
// void run_cpu_stages(Task& task) {
// #pragma omp parallel num_threads(num_threads)
//   {
//     if constexpr (processor_type == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if constexpr (processor_type == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if constexpr (processor_type == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

//     // Generate a compile-time sequence for the range [start_stage, end_stage]
//     []<std::size_t... I>(
//         std::index_sequence<I...>, tree::AppData& data, tree::omp::TmpStorage& tmp_storage) {
//       // Each I is offset by (start_stage - 1)
//       ((tree::omp::run_stage<start_stage + I>(data, tmp_storage)), ...);
//     }(std::make_index_sequence<end_stage - start_stage + 1>{},
//       *task.app_data,
//       *task.omp_tmp_storage);
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
//   requires ValidStageRange<start_stage, end_stage>
// void run_gpu_stages(Task& task) {
//   // Generate a compile-time sequence for the range [start_stage, end_stage]
//   []<std::size_t... I>(
//       std::index_sequence<I...>, tree::AppData& data, tree::vulkan::TmpStorage& tmp_storage) {
//     ((tree::vulkan::Singleton::getInstance().run_stage<start_stage + I>(data, tmp_storage)),
//     ...);
//   }(std::make_index_sequence<end_stage - start_stage + 1>{},
//     *task.app_data,
//     *task.vulkan_tmp_storage);
// }
