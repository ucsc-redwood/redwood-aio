#include <spdlog/spdlog.h>

#include <iomanip>
#include <iostream>

#include "generated-code/test.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// Device-specific pipeline stages (3A021JEHN02756)
// ---------------------------------------------------------------------

// namespace device_3A021JEHN02756 {

// namespace instance_2 {
// // --- Valid Execution Schedule #81 ---
// // Schedule Report:
// //   Chunk 1: Hardware = little, Threads = 4
// //     Stage 1: 4.24 ms
// //     Chunk Total Time: 4.24 ms
// //   Chunk 2: Hardware = big, Threads = 2
// //     Stage 2: 0.153 ms
// //     Stage 3: 23.8 ms
// //     Stage 4: 0.118 ms
// //     Chunk Total Time: 24.070999999999998 ms
// //   Chunk 3: Hardware = gpu, Threads = 1
// //     Stage 5: 9.02 ms
// //     Stage 6: 12.4 ms
// //     Stage 7: 9.66 ms
// //     Chunk Total Time: 31.080000000000002 ms
// //   Chunk 4: Hardware = medium, Threads = 2
// //     Stage 8: 0.046 ms
// //     Stage 9: 0.025 ms
// //     Chunk Total Time: 0.07100000000000001 ms
// // Pipeline Total Time: 59.462 ms
// // Max (Slowest) Chunk Time: 31.080000000000002 ms

// std::atomic<bool> done(false);

// void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
//   for (auto& task : in_tasks) {
//     // ---------
//     run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
//     // ---------
//     q_AB.enqueue(task);
//   }

//   done = true;
// }

// void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB,
//                    moodycamel::ConcurrentQueue<Task>& q_BC) {
//   while (!done) {
//     Task task;
//     if (q_AB.try_dequeue(task)) {
//       // ---------
//       run_stages<2, 4, ProcessorType::kBigCore, 2>(task.app_data);
//       // ---------
//       q_BC.enqueue(task);
//     } else {
//       std::this_thread::yield();
//     }
//   }
// }

// void stage_group_C(moodycamel::ConcurrentQueue<Task>& q_BC,
//                    moodycamel::ConcurrentQueue<Task>& q_CD) {
//   while (!done) {
//     Task task;
//     if (q_BC.try_dequeue(task)) {
//       // ---------
//       run_gpu_stages<5, 7>(task.app_data);
//       // ---------
//       q_CD.enqueue(task);
//     } else {
//       std::this_thread::yield();
//     }
//   }
// }

// void stage_group_D(moodycamel::ConcurrentQueue<Task>& q_CD, std::vector<Task>& out_tasks) {
//   while (!done) {
//     Task task;
//     if (q_CD.try_dequeue(task)) {
//       // ---------
//       run_stages<8, 9, ProcessorType::kMediumCore, 1>(task.app_data);
//       // ---------
//       out_tasks.push_back(task);
//     }
//   }
// }

// }  // namespace instance_2

// }  // namespace device_3A021JEHN02756

// ---------------------------------------------------------------------
// Pipeline Instance (Best)
// ---------------------------------------------------------------------

void run_best() {
  auto tasks = init_tasks(20);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  if (g_device_id == "3A021JEHN02756") {
    // Little cores: 0 1 2 3
    // Mid cores: 4 5
    // Big cores: 6 7

    schedule_3A021JEHN02756_CifarDense_schedule_001::run_pipeline(tasks, out_tasks);

  } else if (g_device_id == "9b034f1b") {
    exit(0);
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    exit(0);
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "Pipeline: Average time per iteration: " << avg_time << " us" << "\t "
            << avg_time / 1000.0 << " ms" << std::endl;

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------

[[nodiscard]] std::chrono::duration<double> run_baseline(const int num_threads) {
  auto tasks = init_tasks(10);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (auto& task : tasks) {
#pragma omp parallel num_threads(num_threads)
    {
      cifar_dense::omp::run_stage<1>(*task.app_data);
      cifar_dense::omp::run_stage<2>(*task.app_data);
      cifar_dense::omp::run_stage<3>(*task.app_data);
      cifar_dense::omp::run_stage<4>(*task.app_data);
      cifar_dense::omp::run_stage<5>(*task.app_data);
      cifar_dense::omp::run_stage<6>(*task.app_data);
      cifar_dense::omp::run_stage<7>(*task.app_data);
      cifar_dense::omp::run_stage<8>(*task.app_data);
      cifar_dense::omp::run_stage<9>(*task.app_data);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;

  cleanup(tasks);
  return duration;
}

[[nodiscard]] std::chrono::duration<double> run_gpu_baseline() {
  auto tasks = init_tasks(10);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (auto& task : tasks) {
    run_gpu_stages<1, 9>(task.app_data);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = end - start;

  cleanup(tasks);
  return duration;
}

void find_best_baseline() {
  std::chrono::duration<double> min_duration = std::chrono::duration<double>::max();
  int best_threads = 1;
  const int max_threads = std::thread::hardware_concurrency();

  spdlog::info("Running baseline benchmarks with 1-{} threads...", max_threads);

  // First run CPU benchmarks
  for (int i = 1; i <= max_threads; ++i) {
    auto duration = run_baseline(i);
    double ms = std::chrono::duration<double, std::milli>(duration).count();

    // Use std::cout for progress updates
    std::cout << "CPU time with " << i << " thread" << (i == 1 ? "" : "s") << ": " << std::fixed
              << std::setprecision(2) << ms << " ms" << std::endl;

    if (duration < min_duration) {
      min_duration = duration;
      best_threads = i;
    }
  }

  // Now run GPU benchmark
  std::cout << "\nRunning GPU benchmark..." << std::endl;
  auto gpu_duration = run_gpu_baseline();
  double gpu_ms = std::chrono::duration<double, std::milli>(gpu_duration).count();
  std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_ms << " ms" << std::endl;

  // Compare GPU vs CPU
  double best_ms = std::chrono::duration<double, std::milli>(min_duration).count();
  if (gpu_duration < min_duration) {
    spdlog::info("Best configuration: GPU ({:.2f} ms)", gpu_ms);
    spdlog::info("Best CPU configuration: {} thread{} ({:.2f} ms)",
                 best_threads,
                 best_threads == 1 ? "" : "s",
                 best_ms);
  } else {
    spdlog::info("Best configuration: CPU with {} thread{} ({:.2f} ms)",
                 best_threads,
                 best_threads == 1 ? "" : "s",
                 best_ms);
    spdlog::info("GPU configuration: {:.2f} ms", gpu_ms);
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  run_best();

  return 0;
}