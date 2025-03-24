#pragma once

#include "../concepts.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-sparse/omp/dispatchers.hpp"
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
    []<std::size_t... I>(std::index_sequence<I...>, cifar_sparse::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((cifar_sparse::omp::run_stage<start_stage + I>(data)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{}, *task.app_data);
  }
}

// ---------------------------------------------------------------------
// GPU stages
// ---------------------------------------------------------------------

template <int start_stage, int end_stage>
  requires ValidStageRange<start_stage, end_stage>
void run_gpu_stages(Task& task) {
  // Generate a compile-time sequence for the range [start_stage, end_stage]
  []<std::size_t... I>(std::index_sequence<I...>, cifar_sparse::AppData& data) {
    ((cifar_sparse::cuda::run_stage<start_stage + I>(data)), ...);
  }(std::make_index_sequence<end_stage - start_stage + 1>{}, *task.app_data);
}
