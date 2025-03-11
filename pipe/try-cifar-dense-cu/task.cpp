#include "task.hpp"

#include <spdlog/spdlog.h>

[[nodiscard]] moodycamel::ConcurrentQueue<Task*> init_tasks(std::vector<cifar_dense::AppData>& data,
                                                            cuda::CudaManager* mgr,
                                                            const size_t initial_capacity) {
  // Initialize queue with reasonable capacity to avoid resizing
  moodycamel::ConcurrentQueue<Task*> tasks(initial_capacity);

  // Reserve space for all tasks plus sentinel
  for (auto& app_data : data) {
    Task* task = new Task(&app_data, mgr);
    if (!tasks.enqueue(task)) {
      delete task;
      throw std::runtime_error("Failed to enqueue task");
    }
  }

  // Add sentinel task
  tasks.enqueue(nullptr);

  return tasks;
}

// #include "builtin-apps/common/cuda/cu_mem_resource.cuh"

// // ---------------------------------------------------------------------
// // Global Variables
// // ---------------------------------------------------------------------

// cuda::CudaManagedResource g_mr;

// // ---------------------------------------------------------------------
// // Task structure
// // ---------------------------------------------------------------------

// /**
//  * Initializes a queue of tasks and adds a sentinel task at the end.
//  * The sentinel task (with null pointers) is used to signal the end of
//  * the task stream to the pipeline stages.
//  */
// [[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
//   std::vector<Task> tasks;
//   tasks.reserve(num_tasks + 1);  // Reserve space including sentinel

//   for (uint32_t i = 0; i < num_tasks; ++i) {
//     tasks.push_back(Task{
//         // .app_data = std::make_unique<cifar_dense::AppData>(&g_mr),
//         .app_data = new cifar_dense::AppData(&g_mr),
//         .done = false,
//     });
//   }

//   // Add sentinel task
//   tasks.push_back(Task{
//       .app_data = nullptr,
//       .done = true,
//   });

//   return tasks;
// }

// // ---------------------------------------------------------------------
// // Queue version
// // ---------------------------------------------------------------------

// [[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks) {
//   std::queue<Task> tasks;

//   for (uint32_t i = 0; i < num_tasks; ++i) {
//     Task t{
//         // .app_data = std::make_unique<cifar_dense::AppData>(&g_mr),
//         .app_data = new cifar_dense::AppData(&g_mr),
//         .done = false,
//     };

//     tasks.push(std::move(t));
//   }

//   tasks.push(Task{
//       .app_data = nullptr,
//       .done = true,
//   });

//   return tasks;
// }

// void cleanup(std::queue<Task>& tasks) {
//   spdlog::trace("cleanup, tasks.size() = {}", tasks.size());

//   while (!tasks.empty()) {
//     auto& task = tasks.front();
//     if (task.is_sentinel()) {
//       tasks.pop();
//       continue;
//     }

//     delete task.app_data;
//     task.app_data = nullptr;

//     tasks.pop();
//   }
// }
