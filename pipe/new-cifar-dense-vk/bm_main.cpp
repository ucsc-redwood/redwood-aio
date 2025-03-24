#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "../templates_vk.hpp"
#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// =============================================================================
// AUTOMATICALLY GENERATED BENCHMARK CODE
// =============================================================================

namespace device_9b034f1b {

// =============================================================================
// AUTOMATICALLY GENERATED BENCHMARK CODE
// =============================================================================

// -----------------------------------------------------------------------------
// Schedule 001: 9b034f1b_CifarDense_schedule_022
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 2
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_022(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, vulkan::run_gpu_stages<3, 9>); });

    // Thread joins:
    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 002: 9b034f1b_CifarDense_schedule_043
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_043(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 6>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<7, 7, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 3>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 003: 9b034f1b_CifarDense_schedule_013
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 2
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_013(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 7>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 3>);
    });

    // Thread joins:
    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 004: 9b034f1b_CifarDense_schedule_002
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_002(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 3>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 005: 9b034f1b_CifarDense_schedule_031
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_031(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<3, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, nullptr, vulkan::run_gpu_stages<5, 9>); });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 006: 9b034f1b_CifarDense_schedule_040
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_040(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<5, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 3>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 007: 9b034f1b_CifarDense_schedule_001
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_001(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 008: 9b034f1b_CifarDense_schedule_007
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_007(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 3>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<2, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 009: 9b034f1b_CifarDense_schedule_047
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 2
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_047(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, vulkan::run_gpu_stages<1, 6>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 010: 9b034f1b_CifarDense_schedule_008
// Device: 9b034f1b
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_9b034f1b_CifarDense_schedule_008(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<2, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 3>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

}  // namespace device_9b034f1b

namespace device_3A021JEHN02756 {

// -----------------------------------------------------------------------------
// Schedule 001: 3A021JEHN02756_CifarDense_schedule_022
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_022(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 4>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 002: 3A021JEHN02756_CifarDense_schedule_043
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_043(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 003: 3A021JEHN02756_CifarDense_schedule_013
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_013(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 004: 3A021JEHN02756_CifarDense_schedule_002
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_002(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 005: 3A021JEHN02756_CifarDense_schedule_031
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_031(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 8>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 006: 3A021JEHN02756_CifarDense_schedule_040
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_040(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 4>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 007: 3A021JEHN02756_CifarDense_schedule_001
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_001(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 008: 3A021JEHN02756_CifarDense_schedule_007
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_007(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<8, 8, ProcessorType::kMediumCore, 2>);
    });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 009: 3A021JEHN02756_CifarDense_schedule_047
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_047(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

// -----------------------------------------------------------------------------
// Schedule 010: 3A021JEHN02756_CifarDense_schedule_008
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_008(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<8, 8, ProcessorType::kBigCore, 2>);
    });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

}  // namespace device_3A021JEHN02756

// Extract benchmark registration logic outside of device namespaces
namespace benchmark_registration {

// Define a type for benchmark registration functions
using BenchmarkFunc = void (*)(benchmark::State &);

// Map of device IDs to their schedule IDs
const std::unordered_map<std::string, std::vector<int>> device_schedule_ids = {
    {"9b034f1b", {22, 43, 13, 2, 31, 40, 1, 7, 47, 8}},
    {"3A021JEHN02756", {22, 43, 13, 2, 31, 40, 1, 7, 47, 8}}};

// Create a hash table mapping device IDs and schedule IDs to their benchmark functions and names
const std::unordered_map<std::string,
                         std::unordered_map<int, std::pair<BenchmarkFunc, std::string>>>
    benchmark_map = {{"9b034f1b",
                      {{22,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_022,
                         "BM_schedule_9b034f1b_CifarDense_schedule_022"}},
                       {43,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_043,
                         "BM_schedule_9b034f1b_CifarDense_schedule_043"}},
                       {13,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_013,
                         "BM_schedule_9b034f1b_CifarDense_schedule_013"}},
                       {2,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_002,
                         "BM_schedule_9b034f1b_CifarDense_schedule_002"}},
                       {31,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_031,
                         "BM_schedule_9b034f1b_CifarDense_schedule_031"}},
                       {40,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_040,
                         "BM_schedule_9b034f1b_CifarDense_schedule_040"}},
                       {1,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_001,
                         "BM_schedule_9b034f1b_CifarDense_schedule_001"}},
                       {7,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_007,
                         "BM_schedule_9b034f1b_CifarDense_schedule_007"}},
                       {47,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_047,
                         "BM_schedule_9b034f1b_CifarDense_schedule_047"}},
                       {8,
                        {&device_9b034f1b::BM_schedule_9b034f1b_CifarDense_schedule_008,
                         "BM_schedule_9b034f1b_CifarDense_schedule_008"}}}},
                     {"3A021JEHN02756",
                      {{22,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_022,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_022"}},
                       {43,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_043,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_043"}},
                       {13,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_013,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_013"}},
                       {2,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_002,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_002"}},
                       {31,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_031,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_031"}},
                       {40,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_040,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_040"}},
                       {1,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_001,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_001"}},
                       {7,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_007,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_007"}},
                       {47,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_047,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_047"}},
                       {8,
                        {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarDense_schedule_008,
                         "BM_schedule_3A021JEHN02756_CifarDense_schedule_008"}}}}};

// Helper function to register a single benchmark
void register_single_benchmark(const std::string &device_id, int schedule_id) {
  auto device_it = benchmark_map.find(device_id);
  if (device_it == benchmark_map.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  auto schedule_it = device_it->second.find(schedule_id);
  if (schedule_it == device_it->second.end()) {
    throw std::runtime_error("Invalid schedule ID " + std::to_string(schedule_id) + " for device " +
                             device_id);
  }

  benchmark::RegisterBenchmark(schedule_it->second.second.c_str(), schedule_it->second.first)
      ->Unit(benchmark::kMillisecond)
      ->Iterations(10);
}

// Function to register benchmarks dynamically
void register_benchmarks(const std::string &device_id, int index) {
  auto device_schedule_it = device_schedule_ids.find(device_id);
  if (device_schedule_it == device_schedule_ids.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  const auto &schedule_ids = device_schedule_it->second;

  if (index == -1) {
    // Register all benchmarks for this device
    for (int schedule_id : schedule_ids) {
      register_single_benchmark(device_id, schedule_id);
    }
    return;
  }

  if (index < 0 || index >= static_cast<int>(schedule_ids.size())) {
    throw std::runtime_error("Invalid index for device " + device_id + ". Must be between 0 and " +
                             std::to_string(schedule_ids.size() - 1) + ", or -1 for all");
  }

  register_single_benchmark(device_id, schedule_ids[index]);
}

}  // namespace benchmark_registration

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv);
  benchmark::Initialize(&new_argc, new_argv.data());

  // Register the benchmark based on the device ID and index
  try {
    benchmark_registration::register_benchmarks(g_device_id, schedule_index);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
