
#include <concurrentqueue.h>

#include <cstdint>
#include <memory_resource>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"
#include "cifar-sparse/vulkan/vk_dispatcher.hpp"
#include "spdlog/common.h"

enum class ProcessorType {
  kLittleCore,
  kMediumCore,
  kBigCore,
};

template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
void run_stages(cifar_sparse::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9, "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

#pragma omp parallel num_threads(num_threads)
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
    []<std::size_t... I>(std::index_sequence<I...>, cifar_sparse::AppData& data) {
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

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  spdlog::set_level(spdlog::level::off);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{i, new cifar_sparse::AppData(mr)};
  }

  spdlog::set_level(spdlog::level::debug);

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}

// ---------------------------------------------------------------------
// Tmp
// ---------------------------------------------------------------------

void tmp() {
  auto tasks = init_tasks(10);

  if (g_device_id == "3A021JEHN02756") {
    // Little cores: 0 1 2 3
    // Mid cores: 4 5
    // Big cores: 6 7

    std::thread t1(run_stages<1, 4, ProcessorType::kLittleCore, 4>, tasks[0].app_data);
    std::thread t2(run_stages<5, 6, ProcessorType::kMediumCore, 2>, tasks[0].app_data);
    std::thread t3(run_stages<7, 9, ProcessorType::kBigCore, 2>, tasks[0].app_data);

    t1.join();
    t2.join();
    t3.join();

  } else if (g_device_id == "9b034f1b") {
    // Little cores: 0 1 2
    // Mid cores: 3 4
    // Big cores:

    std::thread t1(run_stages<1, 4, ProcessorType::kLittleCore, 3>, tasks[0].app_data);
    std::thread t2(run_stages<5, 9, ProcessorType::kMediumCore, 2>, tasks[0].app_data);

    t1.join();
    t2.join();

  } else if (g_device_id == "ce0717178d7758b00b7e") {
    // Little cores: 4 5 6 7
    // Mid cores: 0 1 2 3
    // Big cores:

    std::thread t1(run_stages<1, 4, ProcessorType::kLittleCore, 4>, tasks[0].app_data);
    std::thread t2(run_stages<5, 9, ProcessorType::kMediumCore, 4>, tasks[0].app_data);

    t1.join();
    t2.join();
  }

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  tmp();

  return 0;
}