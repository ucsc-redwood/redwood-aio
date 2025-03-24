#include "../templates_vk.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/vk_appdata.hpp"
#include "run_stages.hpp"
#include "task.hpp"

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  // spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
  spdlog::set_level(spdlog::level::trace);

  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  std::vector<tree::vulkan::VkAppData_Safe> preallocated_data =
      init_vk_appdata<tree::vulkan::VkAppData_Safe>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;
  moodycamel::ConcurrentQueue<Task *> q_2_3;

  std::thread t1([&]() {
    chunk<Task, tree::SafeAppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
  });
  std::thread t2([&]() {
    chunk<Task, tree::SafeAppData>(
        q_0_1, &q_1_2, omp::run_multiple_stages<2, 3, ProcessorType::kBigCore, 2>);
  });
  std::thread t3([&]() {
    chunk<Task, tree::vulkan::VkAppData_Safe>(q_1_2, &q_2_3, vulkan::run_gpu_stages<4, 4>);
  });
  std::thread t4([&]() {
    chunk<Task, tree::SafeAppData>(
        q_2_3, nullptr, omp::run_multiple_stages<5, 7, ProcessorType::kMediumCore, 2>);
  });

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  // ---------------------------------------------------------------------

  spdlog::info("Done");

  return 0;
}
