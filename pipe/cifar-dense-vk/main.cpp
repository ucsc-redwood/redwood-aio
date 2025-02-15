#include <spdlog/spdlog.h>

#include <iomanip>
#include <iostream>

#include "generated_codes.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// Pipeline Instance (Best)
// ---------------------------------------------------------------------

void run_one_schedule(int schedule_id) {
  auto tasks = init_tasks(20);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  device_3A021JEHN02756::get_run_pipeline(schedule_id)(tasks, out_tasks);
  // ---------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "[schedule " << schedule_id << "]: Average time per iteration: " << avg_time << " ms"
            << std::endl;

  cleanup(tasks);
}

void run_all_schedules() {
  const auto num_schedules = device_3A021JEHN02756::get_num_schedules();
  for (auto i = 1; i <= num_schedules; ++i) {
    // spdlog::info("Running schedule {}", i);
    std::cout << "Running schedule " << i << std::endl;
    run_one_schedule(i);
  }
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
  CLI::App app{"default"};
  app.add_option("-d,--device", g_device_id, "Device ID")->required();
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info");

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (g_device_id.empty()) {
    throw std::runtime_error("Device ID is required");
  }

  auto& registry = GlobalDeviceRegistry();

  try {
    const Device& device = registry.getDevice(g_device_id);

    auto littleCores = device.getCores(ProcessorType::kLittleCore);
    auto mediumCores = device.getCores(ProcessorType::kMediumCore);
    auto bigCores = device.getCores(ProcessorType::kBigCore);

    std::cout << "Little cores: ";
    for (const auto& core : littleCores) {
      std::cout << core.id << " ";
      g_little_cores.push_back(core.id);
    }
    std::cout << std::endl;

    std::cout << "Medium cores: ";
    for (const auto& core : mediumCores) {
      std::cout << core.id << " ";
      g_medium_cores.push_back(core.id);
    }
    std::cout << std::endl;

    std::cout << "Big cores: ";
    for (const auto& core : bigCores) {
      std::cout << core.id << " ";
      g_big_cores.push_back(core.id);
    }
    std::cout << std::endl;

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "3A021JEHN02756") {
    run_one_schedule(which_schedule);
  } else if (g_device_id == "9b034f1b") {
    return 0;
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    return 0;
  }

  return 0;
}