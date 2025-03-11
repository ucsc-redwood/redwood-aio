#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/conf.hpp"
#include "run_stages.hpp"
#include "task.hpp"

template <typename AppDataType>
[[nodiscard]] std::vector<AppDataType> init_appdata(std::pmr::memory_resource *mr,
                                                    const int num_tasks) {
  std::vector<AppDataType> all_data;
  all_data.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    all_data.emplace_back(mr);  // Each has big vectors
  }
  return all_data;
}

template <typename TaskType>
void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
           moodycamel::ConcurrentQueue<TaskType *> *q_next,
           std::function<void(TaskType *, cuda::CudaManager &)> func,
           cuda::CudaManager &mgr) {
  while (true) {
    Task *task = nullptr;
    if (q_cur.try_dequeue(task)) {
      if (task == nullptr) {
        // Sentinel => pass it on if there's a next queue and stop
        if (q_next != nullptr) {
          q_next->enqueue(nullptr);
        }
        break;
      }

      // -----------------------------------
      func(task, mgr);
      // -----------------------------------

      // If there's a next queue, pass the task along
      if (q_next != nullptr) {
        q_next->enqueue(task);
      }
    } else {
      std::this_thread::yield();
    }
  }
}

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
    chunk<Task>(
        q_input,
        &q_12,
        [](Task *task, cuda::CudaManager &mgr) {
          omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 8>(*task->data, mgr);
        },
        mgr);
  });

  std::thread t2([&]() {
    chunk<Task>(
        q_12,
        &q_23,
        [](Task *task, cuda::CudaManager &mgr) {
          cuda::run_multiple_stages<3, 4>(*task->data, mgr);
        },
        mgr);
  });

  std::thread t3([&]() {
    chunk<Task>(
        q_23,
        nullptr,
        [](Task *task, cuda::CudaManager &mgr) {
          omp::run_multiple_stages<5, 6, ProcessorType::kLittleCore, 12>(*task->data, mgr);
        },
        mgr);
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
    chunk<Task>(
        q_input,
        &q_12,
        [](Task *task, cuda::CudaManager &mgr) {
          omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 3>(*task->data, mgr);
        },
        mgr);
  });

  std::thread t2([&]() {
    chunk<Task>(
        q_12,
        &q_23,
        [](Task *task, cuda::CudaManager &mgr) {
          cuda::run_multiple_stages<3, 4>(*task->data, mgr);
        },
        mgr);
  });

  std::thread t3([&]() {
    chunk<Task>(
        q_23,
        nullptr,
        [](Task *task, cuda::CudaManager &mgr) {
          omp::run_multiple_stages<5, 6, ProcessorType::kLittleCore, 3>(*task->data, mgr);
        },
        mgr);
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
