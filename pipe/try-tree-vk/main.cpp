#include <concurrentqueue.h>
#include <omp.h>

#include <queue>
#include <thread>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------
struct Task {
  tree::AppData* app_data = nullptr;
  tree::omp::TmpStorage* omp_tmp_storage = nullptr;
  tree::vulkan::TmpStorage* vulkan_tmp_storage = nullptr;
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> tasks;

  constexpr auto n_inputs = 640 * 480;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new tree::AppData(mr, n_inputs),
        .omp_tmp_storage = new tree::omp::TmpStorage(),
        .vulkan_tmp_storage = new tree::vulkan::TmpStorage(mr, n_inputs),
        .done = false,
    };

    const auto n_threads = std::thread::hardware_concurrency();
    task.omp_tmp_storage->allocate(n_threads, n_threads);
    tasks.push(task);
  }

  // create a sentinel task
  tasks.push(Task{
      .app_data = nullptr,
      .omp_tmp_storage = nullptr,
      .vulkan_tmp_storage = nullptr,
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
    delete task.omp_tmp_storage;
    delete task.vulkan_tmp_storage;
  }
}

// ---------------------------------------------------------------------
// New Design
// ---------------------------------------------------------------------

template <int Stage>
concept ValidStage = (Stage >= 1) && (Stage <= 7);

template <int Start, int End>
concept ValidStageRange = ValidStage<Start> && ValidStage<End> && (Start <= End);

// Helper function that unfolds the stage calls.
template <int Start, int... Is>
void run_cpu_stages_impl(tree::AppData* app_data,
                         tree::omp::TmpStorage* tmp_storage,
                         std::integer_sequence<int, Is...>) {
  // Expand the calls: run_stage<Start + 0>(), run_stage<Start + 1>(), ...
  (tree::omp::run_stage<Start + Is>(*app_data, *tmp_storage), ...);
}

// Main interface
template <int Start, int End, ProcessorType processor_type, int n_threads>
  requires ValidStageRange<Start, End>
void run_cpu_stages(Task& task) {
  // Bind to the selected cores
  if constexpr (processor_type == ProcessorType::kLittleCore) {
    bind_thread_to_cores(g_little_cores);
  } else if constexpr (processor_type == ProcessorType::kMediumCore) {
    bind_thread_to_cores(g_medium_cores);
  } else if constexpr (processor_type == ProcessorType::kBigCore) {
    bind_thread_to_cores(g_big_cores);
  }

#pragma omp parallel num_threads(n_threads)
  {
    // Generate the sequence [0, 1, 2, ..., (End-Start)]
    // and expand the calls.
    run_cpu_stages_impl<Start>(
        task.app_data, task.omp_tmp_storage, std::make_integer_sequence<int, End - Start + 1>{});
  }
}

// Helper function that unfolds the stage calls.
template <int Start, int... Is>
void run_gpu_stages_impl(tree::AppData* app_data,
                         tree::vulkan::TmpStorage* tmp_storage,
                         std::integer_sequence<int, Is...>) {
  // Expand the calls: run_stage<Start + 0>(), run_stage<Start + 1>(), ...
  (tree::vulkan::Singleton::getInstance().run_stage<Start + Is>(*app_data, *tmp_storage), ...);
}

// Main interface
template <int Start, int End>
  requires ValidStageRange<Start, End>
void run_gpu_stages(Task& task) {
  // Generate the sequence [0, 1, 2, ..., (End-Start)]
  // and expand the calls.
  run_gpu_stages_impl<Start>(
      task.app_data, task.vulkan_tmp_storage, std::make_integer_sequence<int, End - Start + 1>{});
}

// ---------------------------------------------------------------------
// Old working design
// ---------------------------------------------------------------------

// template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
//   requires(start_stage <= end_stage) && (start_stage >= 1) && (end_stage <= 7)
// void run_stages(tree::AppData* app_data, tree::omp::TmpStorage* omp_tmp_storage) {
// #pragma omp parallel num_threads(num_threads)
//   {
//     // Bind to core if needed:
//     if constexpr (processor_type == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if constexpr (processor_type == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if constexpr (processor_type == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

//     // Generate a compile-time sequence for the range [start_stage, end_stage]
//     []<std::size_t... I>(
//         std::index_sequence<I...>, tree::AppData& data, tree::omp::TmpStorage& omp_tmp_storage) {
//       // Each I is offset by (start_stage - 1)
//       ((tree::omp::run_stage<start_stage + I>(data, omp_tmp_storage)), ...);
//     }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data, *omp_tmp_storage);
//   }
// }

// /**
//  * @brief Runs stages of the CIFAR dense network on GPU using Vulkan
//  *
//  * @tparam start_stage First stage to execute (must be >= 1)
//  * @tparam end_stage Last stage to execute (must be <= 9)
//  * @param app_data Pointer to application data containing network state
//  *
//  * This template function executes the specified range of network stages on the GPU using Vulkan.
//  * The stages are run in sequence using compile-time unrolling.
//  */
// template <int start_stage, int end_stage>
//   requires(start_stage <= end_stage) && (start_stage >= 1) && (end_stage <= 7)
// void run_gpu_stages(tree::AppData* app_data, tree::vulkan::TmpStorage* vulkan_tmp_storage) {
//   // Generate a compile-time sequence for the range [start_stage, end_stage]
//   []<std::size_t... I>(std::index_sequence<I...>,
//                        tree::AppData& data,
//                        tree::vulkan::TmpStorage& vulkan_tmp_storage) {
//     ((tree::vulkan::Singleton::getInstance().run_stage<start_stage + I>(data,
//     vulkan_tmp_storage)),
//      ...);
//   }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data, *vulkan_tmp_storage);
// }

// ---------------------------------------------------------------------
// Define a Schedule
// ---------------------------------------------------------------------

namespace TestSchedule {

void chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    // ---------------------------------------------------------------------
    run_cpu_stages<1, 3, ProcessorType::kBigCore, 2>(task);
    // ---------------------------------------------------------------------

    out_q.enqueue(task);
    in_tasks.pop();
  }
}

void chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    if (Task task; in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      // ---------------------------------------------------------------------
      run_cpu_stages<4, 6, ProcessorType::kLittleCore, 4>(task);
      // ---------------------------------------------------------------------

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    if (Task task; in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      // ---------------------------------------------------------------------
      run_gpu_stages<7, 7>(task);
      // ---------------------------------------------------------------------

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

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
  constexpr auto num_tasks = 20;
  auto tasks = init_tasks(num_tasks);
  std::queue<Task> out_tasks;

  auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  TestSchedule::run_pipeline(tasks, out_tasks);
  // ---------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(num_tasks);

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

  if (g_device_id != "3A021JEHN02756") {
    exit(0);
  }

  run_test_schedule();

  return 0;
}