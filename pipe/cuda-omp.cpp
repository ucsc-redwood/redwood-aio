#include <concurrentqueue.h>
#include <omp.h>

#include <queue>
#include <thread>

#include "../builtin-apps/cifar-dense/arg_max.hpp"
#include "../builtin-apps/cifar-dense/cuda/cu_dense_kernel.cuh"
#include "../builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "../builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "affinity.hpp"
#include "app.hpp"

struct Task {
  uint32_t uid;
  cifar_dense::AppData* app_data;
  bool is_sentinel;
};

Task new_task(cifar_dense::AppData* app_data) {
  static uint32_t uid = 0;
  Task task;
  task.uid = uid++;
  task.app_data = app_data;
  task.is_sentinel = false;
  return task;
}

Task new_sentinel_task() {
  Task task;
  task.uid = 0;
  task.app_data = nullptr;
  task.is_sentinel = true;
  return task;
}

void process_stage_A(cifar_dense::AppData& app_data) {
#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_core(g_little_cores);

    cifar_dense::omp::process_stage_1(app_data);
    cifar_dense::omp::process_stage_2(app_data);
    cifar_dense::omp::process_stage_3(app_data);
    // cifar_dense::omp::process_stage_5(app_data);
    // cifar_dense::omp::process_stage_6(app_data);
    // cifar_dense::omp::process_stage_7(app_data);
    // cifar_dense::omp::process_stage_8(app_data);
    // cifar_dense::omp::process_stage_9(app_data);
  }
}

void process_stage_B(cifar_dense::AppData& app_data) {
  cifar_dense::omp::process_stage_4(app_data);

  cifar_dense::cuda::process_stage_5(app_data);
  cifar_dense::cuda::process_stage_6(app_data);
  cifar_dense::cuda::process_stage_7(app_data);
  cifar_dense::cuda::process_stage_8(app_data);
  cifar_dense::cuda::process_stage_9(app_data);
  cifar_dense::cuda::device_sync();
}

void run_2_stages() {
  constexpr auto n_tasks = 100;

  std::queue<Task> q_A;
  moodycamel::ConcurrentQueue<Task> q_AB(n_tasks + 1);
  std::queue<Task> q_B;

  auto mr = cuda::CudaMemoryResource();

  std::vector<cifar_dense::AppData> appdatas;
  for (int i = 0; i < n_tasks; ++i) {
    appdatas.push_back(cifar_dense::AppData(&mr));
  }

  auto start = std::chrono::high_resolution_clock::now();

  // populate 100 tasks into q_A
  for (int i = 0; i < n_tasks; ++i) {
    q_A.push(new_task(&appdatas[i]));
  }
  q_A.push(new_sentinel_task());

  // --------------------------------------------------------------------------

  std::thread t_A([&]() {
    while (true) {
      Task task = q_A.front();
      q_A.pop();

      if (task.is_sentinel) {
        q_AB.enqueue(task);
        break;
      }

      process_stage_A(*task.app_data);
      q_AB.enqueue(task);
    }
  });

  std::thread t_B([&]() {
    while (true) {
      Task task;
      if (q_AB.try_dequeue(task)) {
        if (task.is_sentinel) {
          q_B.push(task);
          break;
        }

        process_stage_B(*task.app_data);

        q_B.push(task);
      }
    }
  });

  t_A.join();
  t_B.join();

  // --------------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_per_task = static_cast<double>(duration.count()) / n_tasks;
  std::cout << "Average time per task: " << avg_per_task << " ms" << std::endl;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  run_2_stages();

  return 0;
}
