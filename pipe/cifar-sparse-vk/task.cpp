#include "task.hpp"

#include "builtin-apps/cifar-sparse/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------
/**
 * Initializes a queue of tasks and adds a sentinel task at the end.
 * The sentinel task (with null pointers) is used to signal the end of
 * the task stream to the pipeline stages.
 */
[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();
  std::queue<Task> tasks;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new cifar_sparse::AppData(mr),
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
    if (!task.is_sentinel()) {
      delete task.app_data;
    }
    tasks.pop();
  }
}