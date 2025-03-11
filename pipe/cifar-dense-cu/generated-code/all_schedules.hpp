// Aggregated schedules for device: jetson, application: CifarDense
#pragma once

#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <queue>
#include <thread>

#include "../../templates.hpp"
#include "../run_stages.hpp"
#include "../task.hpp"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/common/cuda/manager.cuh"

namespace device_jetson {

using AppData = cifar_dense::AppData;

namespace schedule_jetson_CifarDense_schedule_001 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 7>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_001

namespace schedule_jetson_CifarDense_schedule_002 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 8>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_002

namespace schedule_jetson_CifarDense_schedule_003 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_003

namespace schedule_jetson_CifarDense_schedule_004 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_004

namespace schedule_jetson_CifarDense_schedule_005 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t_only(
      [&]() { chunk<Task, AppData>(q_input, nullptr, cuda::run_multiple_stages<1, 9>, mgr); });

  t_only.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_005

namespace schedule_jetson_CifarDense_schedule_006 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_006

namespace schedule_jetson_CifarDense_schedule_007 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_007

namespace schedule_jetson_CifarDense_schedule_008 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_008

namespace schedule_jetson_CifarDense_schedule_009 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_009

namespace schedule_jetson_CifarDense_schedule_010 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<6, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_010

namespace schedule_jetson_CifarDense_schedule_011 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_011

namespace schedule_jetson_CifarDense_schedule_012 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<5, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_012

namespace schedule_jetson_CifarDense_schedule_013 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<4, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_013

namespace schedule_jetson_CifarDense_schedule_014 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<3, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_014

namespace schedule_jetson_CifarDense_schedule_015 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1(
      [&]() { chunk<Task, AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr); });
  std::thread t2([&]() {
    chunk<Task, AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<2, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_015

namespace schedule_jetson_CifarDense_schedule_016 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<8, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_016

namespace schedule_jetson_CifarDense_schedule_017 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  moodycamel::ConcurrentQueue<Task *> q_0_1;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 8, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, AppData>(q_0_1, nullptr, cuda::run_multiple_stages<9, 9>, mgr); });

  t1.join();
  t2.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_017

namespace schedule_jetson_CifarDense_schedule_018 {

inline void run_pipeline(const int num_tasks) {
  cuda::CudaManager mgr;

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<AppData>(&mgr.get_mr(), num_tasks);

  // Initialize input queue with tasks
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t_only([&]() {
    chunk<Task, AppData>(
        q_input, nullptr, omp::run_multiple_stages<1, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t_only.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace schedule_jetson_CifarDense_schedule_018

// --------------------------------------------------------------------------
// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(int);

// Array of function pointers to all run_pipeline implementations
static const RunPipelineFunc run_pipeline_table[] = {
    schedule_jetson_CifarDense_schedule_001::run_pipeline,
    schedule_jetson_CifarDense_schedule_002::run_pipeline,
    schedule_jetson_CifarDense_schedule_003::run_pipeline,
    schedule_jetson_CifarDense_schedule_004::run_pipeline,
    schedule_jetson_CifarDense_schedule_005::run_pipeline,
    schedule_jetson_CifarDense_schedule_006::run_pipeline,
    schedule_jetson_CifarDense_schedule_007::run_pipeline,
    schedule_jetson_CifarDense_schedule_008::run_pipeline,
    schedule_jetson_CifarDense_schedule_009::run_pipeline,
    schedule_jetson_CifarDense_schedule_010::run_pipeline,
    schedule_jetson_CifarDense_schedule_011::run_pipeline,
    schedule_jetson_CifarDense_schedule_012::run_pipeline,
    schedule_jetson_CifarDense_schedule_013::run_pipeline,
    schedule_jetson_CifarDense_schedule_014::run_pipeline,
    schedule_jetson_CifarDense_schedule_015::run_pipeline,
    schedule_jetson_CifarDense_schedule_016::run_pipeline,
    schedule_jetson_CifarDense_schedule_017::run_pipeline,
    schedule_jetson_CifarDense_schedule_018::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() {
  return sizeof(run_pipeline_table) / sizeof(run_pipeline_table[0]);
}

[[nodiscard]] inline RunPipelineFunc get_run_pipeline_func(const int schedule_id) {
  if (schedule_id < 1 || schedule_id > get_num_schedules()) {
    spdlog::error("Invalid schedule ID: {}", schedule_id);
    throw std::invalid_argument("Invalid schedule ID");
  }
  return run_pipeline_table[schedule_id - 1];
}

}  // namespace device_jetson
