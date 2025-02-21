#include "task.hpp"

#include "builtin-apps/cifar-dense/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new cifar_dense::AppData(mr),
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}
