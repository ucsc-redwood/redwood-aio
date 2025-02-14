#include <spdlog/spdlog.h>

#include <cstdint>
#include <iomanip>
#include <iostream>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"
#include "cifar-dense/vulkan/vk_dispatcher.hpp"
#include "third-party/concurrentqueue.h"

enum class ProcessorType {
  kLittleCore,
  kMediumCore,
  kBigCore,
};

template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
void run_stages(cifar_dense::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9, "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

#pragma omp parallel num_threads(num_threads)
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    } else {
      assert(false);
    }

    // Generate a compile-time sequence for the range [start_stage, end_stage]
    []<std::size_t... I>(std::index_sequence<I...>, cifar_dense::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((cifar_dense::omp::run_stage<start_stage + I>(data)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
  }
}

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData* app_data;  // basically just a pointer
};

std::atomic<bool> done(false);

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        .app_data = new cifar_dense::AppData(mr),
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}

// ---------------------------------------------------------------------
// Device-specific pipeline stages (3A021JEHN02756)
// ---------------------------------------------------------------------

namespace device_3A021JEHN02756 {

namespace instance_2 {
// --- Valid Execution Schedule #81 ---
// Schedule Report:
//   Chunk 1: Hardware = little, Threads = 4
//     Stage 1: 4.24 ms
//     Chunk Total Time: 4.24 ms
//   Chunk 2: Hardware = big, Threads = 2
//     Stage 2: 0.153 ms
//     Stage 3: 23.8 ms
//     Stage 4: 0.118 ms
//     Chunk Total Time: 24.070999999999998 ms
//   Chunk 3: Hardware = gpu, Threads = 1
//     Stage 5: 9.02 ms
//     Stage 6: 12.4 ms
//     Stage 7: 9.66 ms
//     Chunk Total Time: 31.080000000000002 ms
//   Chunk 4: Hardware = medium, Threads = 2
//     Stage 8: 0.046 ms
//     Stage 9: 0.025 ms
//     Chunk Total Time: 0.07100000000000001 ms
// Pipeline Total Time: 59.462 ms
// Max (Slowest) Chunk Time: 31.080000000000002 ms

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB,
                   moodycamel::ConcurrentQueue<Task>& q_BC) {
  while (!done) {
    Task task;

    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<2, 4, ProcessorType::kBigCore, 2>(task.app_data);
      // ---------

      q_BC.enqueue(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

void stage_group_C(moodycamel::ConcurrentQueue<Task>& q_BC,
                   moodycamel::ConcurrentQueue<Task>& q_CD) {
  while (!done) {
    Task task;

    if (q_BC.try_dequeue(task)) {
      // ---------
      cifar_dense::vulkan::Singleton::getInstance().run_stage<5>(*task.app_data);
      cifar_dense::vulkan::Singleton::getInstance().run_stage<6>(*task.app_data);
      cifar_dense::vulkan::Singleton::getInstance().run_stage<7>(*task.app_data);
      // ---------

      q_CD.enqueue(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

void stage_group_D(moodycamel::ConcurrentQueue<Task>& q_CD, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;

    if (q_CD.try_dequeue(task)) {
      // ---------
      run_stages<8, 9, ProcessorType::kMediumCore, 1>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    }
  }
}

}  // namespace instance_2

}  // namespace device_3A021JEHN02756

// ---------------------------------------------------------------------
// Device-specific pipeline stages (9b034f1b)
// ---------------------------------------------------------------------

namespace device_9b034f1b {

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 4, ProcessorType::kLittleCore, 3>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;
    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<5, 9, ProcessorType::kMediumCore, 2>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

}  // namespace device_9b034f1b

// ---------------------------------------------------------------------
// Device-specific pipeline stages (ce0717178d7758b00b7e)
// ---------------------------------------------------------------------

namespace device_ce0717178d7758b00b7e {

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;
    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<5, 9, ProcessorType::kMediumCore, 4>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

}  // namespace device_ce0717178d7758b00b7e

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

    moodycamel::ConcurrentQueue<Task> q_AB;
    moodycamel::ConcurrentQueue<Task> q_BC;
    moodycamel::ConcurrentQueue<Task> q_CD;

    std::thread t_A(
        device_3A021JEHN02756::instance_2::stage_group_A, std::ref(tasks), std::ref(q_AB));
    std::thread t_B(
        device_3A021JEHN02756::instance_2::stage_group_B, std::ref(q_AB), std::ref(q_BC));
    std::thread t_C(
        device_3A021JEHN02756::instance_2::stage_group_C, std::ref(q_BC), std::ref(q_CD));
    std::thread t_D(
        device_3A021JEHN02756::instance_2::stage_group_D, std::ref(q_CD), std::ref(out_tasks));

    t_A.join();
    t_B.join();
    t_C.join();
    t_D.join();

    assert(out_tasks.size() == 20);

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
    cifar_dense::vulkan::Singleton::getInstance().run_stage<1>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<2>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<3>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<4>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<5>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<6>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<7>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<8>(*task.app_data);
    cifar_dense::vulkan::Singleton::getInstance().run_stage<9>(*task.app_data);
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

  spdlog::set_level(spdlog::level::info);

  // find_best_baseline();
  // std::cout << "\nRunning GPU benchmark..." << std::endl;
  // auto gpu_duration = run_gpu_baseline();
  // double gpu_ms = std::chrono::duration<double, std::milli>(gpu_duration).count();
  // std::cout << "GPU time: " << std::fixed << std::setprecision(2) << gpu_ms << " ms" <<
  // std::endl;

  run_best();

  return 0;
}