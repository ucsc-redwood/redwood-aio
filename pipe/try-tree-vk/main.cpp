
#include <concurrentqueue.h>

#include <atomic>
#include <functional>
#include <thread>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "builtin-apps/tree/vulkan/vk_dispatcher.hpp"

// template <int start_stage, int end_stage>
// concept ValidStageRange = requires() {
//   { start_stage >= 1 && end_stage <= 9 } -> std::same_as<bool>;
//   { start_stage <= end_stage } -> std::same_as<bool>;
// };

// template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
//   requires ValidStageRange<start_stage, end_stage>
// void run_stages(tree::AppData* app_data, tree::omp::TmpStorage* temp_storage) {
// #pragma omp parallel num_threads(num_threads)
//   {
//     // Bind to core if needed:
//     if constexpr (processor_type == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if constexpr (processor_type == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if constexpr (processor_type == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     } else {
//       assert(false);
//     }

//     // Generate a compile-time sequence for the range [start_stage, end_stage]
//     []<std::size_t... I>(
//         std::index_sequence<I...>, tree::AppData& data, tree::omp::TmpStorage& temp_storage) {
//       // Each I is offset by (start_stage - 1)
//       ((tree::omp::run_stage<start_stage + I>(data, temp_storage)), ...);
//     }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data, *temp_storage);
//   }
// }

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

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
    delete task.temp_storage;
  }
}

// ---------------------------------------------------------------------
// Define a Schedule
// ---------------------------------------------------------------------

namespace TestSchedule {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_big_cores.size())
    {
      bind_thread_to_cores(g_big_cores);

      tree::omp::run_stage<1>(*task.app_data, *task.temp_storage);
      tree::omp::run_stage<2>(*task.app_data, *task.temp_storage);
      tree::omp::run_stage<3>(*task.app_data, *task.temp_storage);
    }
    // ---------------------------------------------------------------------
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_medium_cores.size())
      {
        bind_thread_to_cores(g_medium_cores);

        tree::omp::run_stage<4>(*task.app_data, *task.temp_storage);
        tree::omp::run_stage<5>(*task.app_data, *task.temp_storage);
        tree::omp::run_stage<6>(*task.app_data, *task.temp_storage);
      }
      // ---------------------------------------------------------------------
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
      // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_medium_cores.size())
    {
      bind_thread_to_cores(g_medium_cores);

      tree::omp::run_stage<4>(*task.app_data, *task.temp_storage);
      tree::omp::run_stage<5>(*task.app_data, *task.temp_storage);
      tree::omp::run_stage<6>(*task.app_data, *task.temp_storage);
    }
    // ---------------------------------------------------------------------
    out_q.enqueue(task);
  }
}

void chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_little_cores.size())
      {
        bind_thread_to_cores(g_little_cores);

        tree::omp::run_stage<7>(*task.app_data, *task.temp_storage);
      }
      // ---------------------------------------------------------------------
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
      // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_little_cores.size())
    {
      bind_thread_to_cores(g_little_cores);

      tree::omp::run_stage<7>(*task.app_data, *task.temp_storage);
    }
    // ---------------------------------------------------------------------
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace TestSchedule

// ---------------------------------------------------------------------
// Test Schedule
// ---------------------------------------------------------------------

void run_test_schedule() {
  auto tasks = init_tasks(20);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  TestSchedule::run_pipeline(tasks, out_tasks);
  // ---------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "[schedule Test]: Average time per iteration: " << avg_time << " ms" << std::endl;

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  run_test_schedule();

  return 0;
}