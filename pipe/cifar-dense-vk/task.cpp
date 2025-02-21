#include "task.hpp"

#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks;
  tasks.reserve(num_tasks + 1);  // +1 for sentinel

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new cifar_dense::AppData(mr),
        .done = false,
    };

    tasks.push_back(task);
  }

  // create a sentinel task
  tasks.push_back(Task{
      .app_data = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    if (task.is_sentinel()) {
      continue;
    }
    delete task.app_data;
  }
}
