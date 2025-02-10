
#include <concurrentqueue.h>

#include <cstdint>
#include <memory_resource>

#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"
#include "cifar-sparse/vulkan/vk_dispatcher.hpp"

template <int start_stage, int end_stage>
void run_stages(cifar_sparse::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9,
                "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

  // Function table to map stage numbers to their corresponding functions
  constexpr auto stage_functions = [](auto* app_data) {
    constexpr auto stages = std::array{cifar_sparse::omp::process_stage_1,
                                       cifar_sparse::omp::process_stage_2,
                                       cifar_sparse::omp::process_stage_3,
                                       cifar_sparse::omp::process_stage_4,
                                       cifar_sparse::omp::process_stage_5,
                                       cifar_sparse::omp::process_stage_6,
                                       cifar_sparse::omp::process_stage_7,
                                       cifar_sparse::omp::process_stage_8,
                                       cifar_sparse::omp::process_stage_9};

    // Execute only the required number of stages (up to `n_stages`)
    for (int i = start_stage; i < end_stage; ++i) {
      stages[i](*app_data);
    }
  };

#pragma omp parallel
  { stage_functions(app_data); }
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

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{i, new cifar_sparse::AppData(mr)};

    // We actually need to run all the stages first.
    run_stages<1, 9>(tasks[i].app_data);
  }

  return tasks;
}

// ---------------------------------------------------------------------
//
// ---------------------------------------------------------------------

void tmp() {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();
  auto tasks = init_tasks(10, mr);

  // cleanup
  for (auto& task : tasks) {
    delete task.app_data;
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::debug);

  tmp();
  return 0;
}