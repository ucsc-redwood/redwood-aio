#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/conf.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// Define a Schedule (Nvidia PC)
// ---------------------------------------------------------------------

namespace device_pc {

void program(const int num_tasks) {
  cuda::CudaManager mgr;

  auto preallocated_data = init_appdata<cifar_dense::AppData>(&mgr.get_mr(), num_tasks);

  // ---------------------------------------------------------------------
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);
  moodycamel::ConcurrentQueue<Task *> q_12;
  moodycamel::ConcurrentQueue<Task *> q_23;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_12, omp::run_multiple_stages<1, 3, ProcessorType::kBigCore, 8>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_12, &q_23, cuda::run_multiple_stages<4, 6>, mgr);
  });
  std::thread t3([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_23, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 12>, mgr);
  });

  t1.join();
  t2.join();
  t3.join();

  // ---------------------------------------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {
void program(const int num_tasks) {
  cuda::CudaManager mgr;

  auto preallocated_data = init_appdata<cifar_dense::AppData>(&mgr.get_mr(), num_tasks);

  // ---------------------------------------------------------------------
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);
  moodycamel::ConcurrentQueue<Task *> q_12;
  moodycamel::ConcurrentQueue<Task *> q_23;

  auto start = std::chrono::high_resolution_clock::now();

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_12, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_12, &q_23, cuda::run_multiple_stages<4, 6>, mgr);
  });
  std::thread t3([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_23, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 3>, mgr);
  });

  t1.join();
  t2.join();
  t3.join();

  // ---------------------------------------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

}  // namespace device_jetson

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "pc") {
    device_pc::program(20);
  } else if (g_device_id == "jetson") {
    device_jetson::program(20);
  }

  return 0;
}
