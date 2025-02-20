#pragma once

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"

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
void run_stages(cifar_dense::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9, "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

#pragma omp parallel num_threads(num_threads)
  {
    // Bind to core if needed:
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
    []<std::size_t... I>(std::index_sequence<I...>, cifar_dense::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((cifar_dense::omp::run_stage<start_stage + I>(data)), ...);
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
void run_gpu_stages(cifar_dense::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9, "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

  // Generate a compile-time sequence for the range [start_stage, end_stage]
  []<std::size_t... I>(std::index_sequence<I...>, cifar_dense::AppData& data) {
    ((cifar_dense::vulkan::Singleton::getInstance().run_stage<start_stage + I>(data)), ...);
  }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
}
