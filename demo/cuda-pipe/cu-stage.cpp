#include <concurrentqueue.h>

#include <CLI/CLI.hpp>
#include <affinity.hpp>
// #include <cifar_dense_cu_kernel.cuh>

#include <cifar_dense_kernel.hpp>

#include "cuda/cifar_dense_cu_kernel.cuh"
#include "cuda/cu_mem_resource.cuh"

// #include <cifar_sparse_kernel.hpp>
#include <cstdint>
#include <iostream>
#include <queue>
#include <string>
#include <thread>

#include "../conf.hpp"

std::vector<int> g_little_cores;
std::vector<int> g_medium_cores;
std::vector<int> g_big_cores;

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

void process_stage_A(const Task& task) {
  // little cores
#pragma omp parallel
  {
    bind_thread_to_core(g_little_cores);

    cifar_dense::omp::process_stage_1(*task.app_data);
    cifar_dense::omp::process_stage_2(*task.app_data);
    cifar_dense::omp::process_stage_3(*task.app_data);
    cifar_dense::omp::process_stage_4(*task.app_data);
  }
}

void process_stage_B(const Task& task) {
  cifar_dense::cuda::device_sync();
  cifar_dense::cuda::run_stage5_sync(task.app_data);
  cifar_dense::cuda::run_stage6_sync(task.app_data);
  cifar_dense::cuda::run_stage7_sync(task.app_data);
  cifar_dense::cuda::run_stage8_sync(task.app_data);
  cifar_dense::cuda::run_stage9_sync(task.app_data);
  cifar_dense::cuda::device_sync();

  // // medium cores
  // #pragma omp parallel
  //   {
  //     bind_thread_to_core(g_medium_cores);

  //     cifar_dense::omp::process_stage_5(*task.app_data);
  //     cifar_dense::omp::process_stage_6(*task.app_data);
  //     cifar_dense::omp::process_stage_7(*task.app_data);
  //     cifar_dense::omp::process_stage_8(*task.app_data);
  //     cifar_dense::omp::process_stage_9(*task.app_data);
  //   }
}

int main(int argc, char** argv) {
  std::string device_id;

  CLI::App app{"Cifar Dense Benchmark"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  std::cout << "Device ID: " << device_id << std::endl;

  auto device = get_device(device_id);
  g_little_cores = device.get_ge_A(const Task& task) {
    // little cores
#pragma omp parallelinable_cores(kMediumCoreType);
    g_big_cores = device.get_pinable_cores(kBigCoreType);

    // ---
    constexpr auto n_tasks = 100;

    std::queue<Task> q_A;
    moodycamel::ConcurrentQueue<Task> q_AB(n_tasks + 1);
    std::queue<Task> q_B;

    // auto mr = std::pmr::new_delete_resource();
    cuda::CudaMemoryResource mr;

    cifar_dense::AppData app_data(&mr);

    // cifar_dense::cuda::run_stage1_sync(&app_data);
    // cifar_dense::cuda::run_stage2_sync(&app_data);
    // cifar_dense::cuda::run_stage3_sync(&app_data);
    // cifar_dense::cuda::run_stage4_sync(&app_data);

    cifar_dense::cuda::device_sync();

    bind_thread_to_core(g_little_cores);
    cifar_dense::omp::process_stage_1(app_data);
    cifar_dense::omp::process_stage_2(app_data);
    cifar_dense::omp::process_stage_3(app_data);
    cifar_dense::omp::process_stage_4(app_data);

    cifar_dense::cuda::run_stage5_sync(&app_data);
    cifar_dense::cuda::run_stage6_sync(&app_data);
    cifar_dense::cuda::run_stage7_sync(&app_data);
    cifar_dense::cuda::run_stage8_sync(&app_data);
    cifar_dense::cuda::run_stage9_sync(&app_data);
    cifar_dense::cuda::device_sync();

    std::cout << "OK" << std::endl;

    std::vector<cifar_dense::AppData> app_data{
        cifar_dense::AppData(&mr),
        cifar_dense::AppData(&mr),
    };

    // populate the queue with tasks
    for (size_t i = 0; i < n_tasks; ++i) {
      auto& data = app_data[i % app_data.size()];
      q_A.push(new_task(&data));
    }
    q_A.push(new_sentinel_task());

    // for (size_t i = 0; i < n_tasks; ++i) {
    //   auto& data = app_data[i % app_data.size()];
    //   q_A.enqueue(new_task(&data));
    // }
    // q_A.enqueue(new_sentinel_task());

    auto start = std::chrono::high_resolution_clock::now();

    std::thread stage_A_thread([&]() {
      while (!q_A.empty()) {
        Task task = q_A.front();
        q_A.pop();

        if (task.is_sentinel) {
          q_AB.enqueue(task);
          break;
        }

        process_stage_A(task);

        q_AB.enqueue(task);
        // std::cout << "[thread A] processed task " << task.uid << std::endl;
      }
    });

    std::thread stage_B_thread([&]() {
      while (true) {
        Task task;
        if (q_AB.try_dequeue(task)) {
          if (task.is_sentinel) {
            q_B.push(task);
            break;
          }

          process_stage_B(task);

          q_B.push(task);

          // std::cout << "[thread B] processed task " << task.uid <<
          std::endl;
        }
      }
    });

    stage_A_thread.join();
    stage_B_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << "Time taken: " << duration << "ms" << std::endl;
    std::cout << "average per iteration: " << duration / n_tasks << "ms"
              << std::endl;

    std::cout << "Done processing tasks" << std::endl;

    return 0;
  }
