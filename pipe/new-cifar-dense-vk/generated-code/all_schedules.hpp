// Auto-generated header for schedule: 3A021JEHN02756_CifarDense_schedule_001
// Device: 3A021JEHN02756, Application: CifarDense

#pragma once

#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <thread>

#include "../../templates.hpp"
#include "../../templates_vk.hpp"
#include "../run_stages.hpp"
#include "../task.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "spdlog/common.h"

namespace device_3A021JEHN02756 {
namespace schedule_3A021JEHN02756_CifarDense_schedule_001 {

using AppData = cifar_dense::AppData;

inline void run_pipeline(const int num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(mr, num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;
  moodycamel::ConcurrentQueue<Task *> q_2_3;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
  });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
  });
  std::thread t3([&]() { chunk<Task, AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
  std::thread t4([&]() {
    chunk<Task, AppData>(
        q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kBigCore, 2>);
  });

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {:.3f} ms", duration.count() / 1000.0 / num_tasks);
}

inline void run_pipeline_warmup() {
  constexpr size_t num_tasks = 5;

  spdlog::set_level(spdlog::level::off);

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(mr, num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;
  moodycamel::ConcurrentQueue<Task *> q_2_3;

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
  });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
  });
  std::thread t3([&]() { chunk<Task, AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
  std::thread t4([&]() {
    chunk<Task, AppData>(
        q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kBigCore, 2>);
  });

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
}

}  // namespace schedule_3A021JEHN02756_CifarDense_schedule_001
}  // namespace device_3A021JEHN02756
