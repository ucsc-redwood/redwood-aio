

#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "builtin-apps/tree/vulkan/vk_dispatcher.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  tree::AppData* app_data;  // basically just a pointer
  tree::omp::TmpStorage* temp_storage;
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new tree::AppData(mr),
        .temp_storage = new tree::omp::TmpStorage(),
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks);

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  return 0;
}