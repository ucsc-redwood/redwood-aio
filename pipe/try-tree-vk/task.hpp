#pragma once

#include "builtin-apps/tree/omp/func_sort.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------
struct Task {
  tree::AppData* app_data = nullptr;
  tree::omp::TmpStorage* omp_tmp_storage = nullptr;
  tree::vulkan::TmpStorage* vulkan_tmp_storage = nullptr;
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] inline std::queue<Task> init_tasks(const size_t num_tasks) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> tasks;

  constexpr auto n_inputs = 640 * 480;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new tree::AppData(mr, n_inputs),
        .omp_tmp_storage = new tree::omp::TmpStorage(),
        .vulkan_tmp_storage = new tree::vulkan::TmpStorage(mr, n_inputs),
        .done = false,
    };

    const auto n_threads = std::thread::hardware_concurrency();
    task.omp_tmp_storage->allocate(n_threads, n_threads);
    tasks.push(task);
  }

  // create a sentinel task
  tasks.push(Task{
      .app_data = nullptr,
      .omp_tmp_storage = nullptr,
      .vulkan_tmp_storage = nullptr,
      .done = true,
  });

  return tasks;
}

inline void cleanup(std::queue<Task>& tasks) {
  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (task.is_sentinel()) {
      tasks.pop();
      continue;
    }
    delete task.app_data;
    delete task.omp_tmp_storage;
    delete task.vulkan_tmp_storage;
  }
}
