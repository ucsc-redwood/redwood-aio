#include "../templates.hpp"
#include "../templates_vk.hpp"
#include "builtin-apps/app.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// struct DataBundle {
//   tree::AppData data;
//   tree::omp::TmpStorage omp_tmp_storage;
//   tree::vulkan::TmpStorage vulkan_tmp_storage;
// };

// [[nodiscard]] inline std::queue<Task> init_tasks(const size_t num_tasks) {
//   auto mr = tree::vulkan::Singleton::getInstance().get_mr();
//   std::queue<Task> tasks;

//   constexpr auto n_inputs = 640 * 480;

//   for (uint32_t i = 0; i < num_tasks; ++i) {
//     Task task{
//         .app_data = new tree::SafeAppData(mr),
//         .omp_tmp_storage = new tree::omp::TmpStorage(),
//         .vulkan_tmp_storage = new tree::vulkan::TmpStorage(mr, n_inputs),
//         .done = false,
//     };

//     const auto n_threads = std::thread::hardware_concurrency();
//     task.omp_tmp_storage->allocate(n_threads, n_threads);
//     tasks.push(task);
//   }

//   // create a sentinel task
//   tasks.push(Task{
//       .app_data = nullptr,
//       .omp_tmp_storage = nullptr,
//       .vulkan_tmp_storage = nullptr,
//       .done = true,
//   });

//   return tasks;
// }

std::vector<tree::omp::TmpStorage> init_omp_tmp_storages(const size_t num_tasks) {
  std::vector<tree::omp::TmpStorage> omp_tmp_storages;
  omp_tmp_storages.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    omp_tmp_storages.emplace_back(tree::omp::TmpStorage());
  }
  return omp_tmp_storages;
}

std::vector<tree::vulkan::TmpStorage> init_vulkan_tmp_storages(std::pmr::memory_resource *mr,
                                                               const size_t num_tasks) {
  constexpr auto n_inputs = 640 * 480;

  std::vector<tree::vulkan::TmpStorage> vulkan_tmp_storages;
  vulkan_tmp_storages.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    vulkan_tmp_storages.emplace_back(tree::vulkan::TmpStorage(mr, n_inputs));
  }
  return vulkan_tmp_storages;
}

void chunk(moodycamel::ConcurrentQueue<Task *> &q_cur,
           moodycamel::ConcurrentQueue<Task *> *q_next,
           std::function<void(tree::AppData &)> func) {
  while (true) {
    Task *task = nullptr;
    if (q_cur.try_dequeue(task)) {
      if (task == nullptr) {
        // Sentinel => pass it on if there's a next queue and stop
        if (q_next != nullptr) {
          q_next->enqueue(nullptr);
        }
        break;
      }

      // -----------------------------------
      func(*task->data);
      // -----------------------------------

      // If there's a next queue, pass the task along
      if (q_next != nullptr) {
        q_next->enqueue(task);
      }
    } else {
      std::this_thread::yield();
    }
  }
}


// std::vector<DataBundle> init_data_bundles(std::pmr::memory_resource *mr, const size_t num_tasks)
// {
//   std::vector<DataBundle> data_bundles;
//   data_bundles.reserve(num_tasks);
//   for (size_t i = 0; i < num_tasks; ++i) {
//     data_bundles.emplace_back(DataBundle{
//         .data = tree::AppData(mr),
//         .omp_tmp_storage = tree::omp::TmpStorage(),
//         .vulkan_tmp_storage = tree::vulkan::TmpStorage(mr, n_inputs),
//     });
//   }
//   return data_bundles;
// }

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  constexpr size_t num_tasks = 20;

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<tree::AppData>(mr, num_tasks);
  // auto preallocated_omp_tmp_storages = init_omp_tmp_storages(num_tasks);
  auto preallocated_vulkan_tmp_storages = init_vulkan_tmp_storages(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(
      preallocated_data, preallocated_vulkan_tmp_storages);

  // Queue definitions:
  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;

  // Thread calls:
  std::thread t1([&]() {
    chunk<Task, tree::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kMediumCore, 2>);
  });
  std::thread t2([&]() {
    chunk<Task, tree::AppData>(
        q_0_1, &q_1_2, omp::run_multiple_stages<4, 4, ProcessorType::kLittleCore, 3>);
  });
  std::thread t3(
      [&]() { chunk<Task, tree::AppData>(q_1_2, nullptr, vulkan::run_gpu_stages<5, 7>); });

  // Thread joins:
  t1.join();
  t2.join();
  t3.join();

  // ---------------------------------------------------------------------

  return 0;
}
