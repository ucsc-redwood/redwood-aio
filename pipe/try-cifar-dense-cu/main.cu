#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <queue>

#include "../templates.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "task.hpp"

// ---------------------------------------------------------------------
// Define a Schedule (Nvidia PC)
// ---------------------------------------------------------------------

namespace device_pc {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(std::move(task));
      continue;
    }

    // ---------------------------------------------------------------------
    // run_cpu_stages<1, 3, ProcessorType::kBigCore, 8>(task);

#pragma omp parallel num_threads(8)
    {
      bind_thread_to_cores(g_big_cores);

      cifar_dense::omp::run_stage<1>(*task.app_data);
      cifar_dense::omp::run_stage<2>(*task.app_data);
      cifar_dense::omp::run_stage<3>(*task.app_data);
    }

    // ---------------------------------------------------------------------

    out_q.enqueue(std::move(task));
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(std::move(task));
        break;
      }

      // ---------------------------------------------------------------------
      cifar_dense::cuda::run_stage<4>(*task.app_data);
      cifar_dense::cuda::run_stage<5>(*task.app_data);
      cifar_dense::cuda::run_stage<6>(*task.app_data);
      // ---------------------------------------------------------------------

      out_tasks.push_back(std::move(task));
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk4.join();
}

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(std::move(task));
      continue;
    }

    // ---------------------------------------------------------------------
    // run_cpu_stages<1, 3, ProcessorType::kLittleCore, 6>(task);

#pragma omp parallel num_threads(6)
    {
      bind_thread_to_cores(g_little_cores);

      cifar_dense::omp::run_stage<1>(*task.app_data);
      cifar_dense::omp::run_stage<2>(*task.app_data);
      cifar_dense::omp::run_stage<3>(*task.app_data);
    }

    // ---------------------------------------------------------------------

    out_q.enqueue(std::move(task));
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(std::move(task));
        break;
      }

      // ---------------------------------------------------------------------
      // run_gpu_stages<4, 9>(task);

      cifar_dense::cuda::run_stage<4>(*task.app_data);
      cifar_dense::cuda::run_stage<5>(*task.app_data);
      cifar_dense::cuda::run_stage<6>(*task.app_data);

      // ---------------------------------------------------------------------

      out_tasks.push_back(std::move(task));
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk4.join();
}

}  // namespace device_jetson

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "pc") {
    run_pipelined_schedule<Task>(init_tasks, device_pc::run_pipeline);
  } else if (g_device_id == "jetson") {
    run_pipelined_schedule<Task>(init_tasks, device_jetson::run_pipeline);
  }

  return 0;
}
