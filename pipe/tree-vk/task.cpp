#include "task.hpp"

#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new tree::AppData(mr),
        .temp_storage = new tree::omp::TempStorage(1, 1),
        .vulkan_temp_storage = new tree::vulkan::TmpStorage(mr, 640 * 480),
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}
