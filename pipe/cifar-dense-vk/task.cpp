#include "task.hpp"

#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> tasks;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new cifar_dense::AppData(mr),
        .done = false,
    };

    tasks.push(task);
  }

  // create a sentinel task
  tasks.push(Task{
      .app_data = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (task.is_sentinel()) {
      tasks.pop();
      continue;
    }
    delete task.app_data;
  }
}
