
#include <concurrentqueue.h>

#include <cstdint>
#include <memory_resource>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"
#include "cifar-sparse/vulkan/vk_dispatcher.hpp"
#include "spdlog/common.h"

enum class ProcessorType {
  kAny,
  kLittleCore,
  kMediumCore,
  kBigCore,
};

template <int start_stage, int end_stage, ProcessorType processor_type>
void run_stages(cifar_sparse::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9,
                "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

#pragma omp parallel
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_core(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_core(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_core(g_big_cores);
    }

    // Generate a compile-time sequence for the range [start_stage, end_stage]
    []<std::size_t... I>(std::index_sequence<I...>,
                         cifar_sparse::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((cifar_sparse::omp::run_stage<start_stage + I>(data)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
  }
}

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  uint32_t uid;
  cifar_sparse::AppData* app_data;
};

std::atomic<bool> done(false);

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks,
                                           std::pmr::memory_resource* mr) {
  std::vector<Task> tasks(num_tasks);

  spdlog::set_level(spdlog::level::off);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{i, new cifar_sparse::AppData(mr)};

    // We actually need to run all the stages first.
    run_stages<1, 9, ProcessorType::kAny>(tasks[i].app_data);
  }

  spdlog::set_level(spdlog::level::debug);

  return tasks;
}

// ---------------------------------------------------------------------
// Tmp
// ---------------------------------------------------------------------

void tmp() {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();
  auto tasks = init_tasks(10, mr);

  run_stages<1, 4, ProcessorType::kLittleCore>(tasks[0].app_data);
  run_stages<5, 6, ProcessorType::kMediumCore>(tasks[0].app_data);
  run_stages<7, 9, ProcessorType::kBigCore>(tasks[0].app_data);

  // cleanup
  for (auto& task : tasks) {
    delete task.app_data;
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  tmp();

  return 0;
}