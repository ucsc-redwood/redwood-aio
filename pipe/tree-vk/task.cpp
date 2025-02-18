#include "task.hpp"

#include "builtin-apps/tree/vulkan/vk_dispatcher.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new tree::AppData(mr),
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}
