#pragma once

#include "builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData* app_data;  // basically just a pointer
};

[[nodiscard]] inline std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new cifar_dense::AppData(mr),
    };
  }

  return tasks;
}

inline void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}
