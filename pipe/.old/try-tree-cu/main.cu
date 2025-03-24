#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "run_stages.hpp"
#include "task.hpp"

[[nodiscard]] std::vector<tree::AppData> init_appdata(std::pmr::memory_resource *mr,
                                                      const int num_tasks) {
  constexpr auto input_size = 640 * 480;

  std::vector<tree::AppData> all_data;

  all_data.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    all_data.emplace_back(mr, input_size, true);  // Each has big vectors
  }
  return all_data;
}

// ---------------------------------------------------------------------
// Define a Schedule (Nvidia PC)
// ---------------------------------------------------------------------

namespace device_pc {

void program(const int num_tasks) {
  cuda::CudaManager mgr;

  auto preallocated_data = init_appdata(&mgr.get_mr(), num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);
  moodycamel::ConcurrentQueue<Task *> q_12;
  moodycamel::ConcurrentQueue<Task *> q_23;

  auto start = std::chrono::high_resolution_clock::now();

  // ---------------------------------------------------------------------
  // omp::run_multiple_stages<1, 3, ProcessorType::kBigCore, 8>(preallocated_data[0]);

  std::thread t1([&]() {
    chunk<Task, tree::AppData>(
        q_input, &q_12, omp::run_multiple_stages<1, 3, ProcessorType::kBigCore, 8>, mgr);
  });

  std::thread t2(
      [&]() { chunk<Task, tree::AppData>(q_12, &q_23, cuda::run_multiple_stages<4, 6>, mgr); });
  std::thread t3([&]() {
    chunk<Task, tree::AppData>(
        q_23, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 12>, mgr);
  });

  t1.join();
  t2.join();
  t3.join();

  // ---------------------------------------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  spdlog::info("Time taken per task: {:.3f} ms", duration.count() / static_cast<double>(num_tasks));
}

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {
void program(const int num_tasks) {
  cuda::CudaManager mgr;

  auto preallocated_data = init_appdata<tree::AppData>(&mgr.get_mr(), num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);
  moodycamel::ConcurrentQueue<Task *> q_12;
  moodycamel::ConcurrentQueue<Task *> q_23;

  auto start = std::chrono::high_resolution_clock::now();

  // ---------------------------------------------------------------------

  std::thread t1([&]() {
    chunk<Task, tree::AppData>(
        q_input, &q_12, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 3>, mgr);
  });
  std::thread t2(
      [&]() { chunk<Task, tree::AppData>(q_12, &q_23, cuda::run_multiple_stages<4, 6>, mgr); });
  std::thread t3([&]() {
    chunk<Task, tree::AppData>(
        q_23, nullptr, omp::run_multiple_stages<7, 7, ProcessorType::kLittleCore, 3>, mgr);
  });

  t1.join();
  t2.join();
  t3.join();

  // ---------------------------------------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  spdlog::info("Time taken per task: {:.3f} ms", duration.count() / static_cast<double>(num_tasks));
}

}  // namespace device_jetson

void fun_program() {
  cuda::CudaManager mgr;

  auto preallocated_data = init_appdata<tree::AppData>(&mgr.get_mr(), 20);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data, &mgr);

  std::thread t1([&]() {
    chunk<Task, tree::AppData>(
        // q_input, nullptr, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 3>, mgr);
        q_input,
        nullptr,
        cuda::run_multiple_stages<1, 7>,
        mgr);
  });

  t1.join();

  for (auto &data : preallocated_data) {
    auto is_sorted = std::ranges::is_sorted(data.u_morton_keys_sorted_s2);
    spdlog::info("Is sorted: {}", (is_sorted ? "true" : "false"));
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "pc") {
    device_pc::program(30);
  } else if (g_device_id == "jetson") {
    device_jetson::program(30);
  }

  fun_program();

  return 0;
}
