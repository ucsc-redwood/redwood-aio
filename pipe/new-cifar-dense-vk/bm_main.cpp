#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "../templates_vk.hpp"
#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
// #include "generated-code/all_schedules.hpp"
#include "run_stages.hpp"
#include "spdlog/common.h"
#include "task.hpp"

static void bench(benchmark::State &state) {
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
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Use chrono for high resolution timing
    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

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

    t1.join();
    t2.join();
    t3.join();
    t4.join();

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

BENCHMARK(bench)->Unit(benchmark::kMillisecond)->Iterations(10);

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  if (g_device_id == "3A021JEHN02756") {
    auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv);

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
  }

  return 0;
}
