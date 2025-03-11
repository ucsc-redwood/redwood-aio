#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/conf.hpp"
#include "run_stages.hpp"
#include "task.hpp"

[[nodiscard]] std::vector<cifar_dense::AppData> init_appdata(std::pmr::memory_resource *mr,
                                                             const int num_tasks) {
  std::vector<cifar_dense::AppData> all_data;
  all_data.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    all_data.emplace_back(mr);  // Each has big vectors
  }
  return all_data;
}

template <typename TaskType>
void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
           moodycamel::ConcurrentQueue<TaskType *> *q_next,
           // lambda function and its parameters
           std::function<void(TaskType *, cuda::CudaManager *)> func,
           cuda::CudaManager *mgr) {
  while (true) {
    TaskType *task = nullptr;
    if (q_cur.try_dequeue(task)) {
      if (task == nullptr) {
        // Sentinel => pass it on if there's a next queue and stop
        if (q_next != nullptr) {
          q_next->enqueue(nullptr);
        }
        break;
      }

      // -----------------------------------
      func(task->data, mgr);
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

  std::vector<cifar_dense::AppData> allData = init_appdata(&mgr.get_mr(), num_tasks);

  // ---------------------------------------------------------------------
  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(allData, &mgr);
  moodycamel::ConcurrentQueue<Task *> q_12;
  moodycamel::ConcurrentQueue<Task *> q_23;
  moodycamel::ConcurrentQueue<Task *> q_34;

  auto start = std::chrono::high_resolution_clock::now();

  // std::thread t1(chunk, std::ref(q_input), &q_12, omp::run_multiple_stages<1, 2,
  // ProcessorType::kBigCore, 8>); std::thread t2(chunk, std::ref(q_23), &q_34,
  // cuda::run_multiple_stages<5, 6>, &mgr); std::thread t3(chunk, std::ref(q_34), nullptr,
  // omp::run_multiple_stages<7, 9,   ProcessorType::kLittleCore, 12>);

  Task *task = nullptr;
  omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 8>(*task->data);

  cuda::run_multiple_stages<5, 6>(*task->data, mgr);

  // ---------------------------------------------------------------------
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
}

// void run_pipeline_queue(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
//   moodycamel::ConcurrentQueue<Task> q_01;

//   std::thread t_chunk1(
//       [&]() { chunk_first(tasks, q_01, run_cpu_stages<1, 3, ProcessorType::kBigCore, 8>); });
//   std::thread t_chunk4([&]() { chunk_last(q_01, out_tasks, run_gpu_stages<4, 7>); });

//   t_chunk1.join();
//   t_chunk4.join();
// }

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {

// void run_pipeline_queue(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
//   moodycamel::ConcurrentQueue<Task> q_01;

//   std::thread t_chunk1(
//       [&]() { chunk_first(tasks, q_01, run_cpu_stages<1, 3, ProcessorType::kLittleCore, 6>); });
//   std::thread t_chunk4([&]() { chunk_last(q_01, out_tasks, run_gpu_stages<4, 7>); });

//   t_chunk1.join();
//   t_chunk4.join();
// }

}  // namespace device_jetson

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "pc") {
    // run_pipelined_schedule<Task>(init_tasks_queue, device_pc::run_pipeline_queue, cleanup);
  } else if (g_device_id == "jetson") {
    // run_pipelined_schedule<Task>(init_tasks_queue, device_jetson::run_pipeline_queue, cleanup);
  }

  return 0;
}
