#include <omp.h>
#include <spdlog/spdlog.h>

#include <algorithm>  // for std::min
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "cifar-dense/cuda/cu_dense_kernel.cuh"
#include "cifar-dense/dense_appdata.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"
#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "concurrentqueue.h"  // Moodycamel's ConcurrentQueue

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  int uid;
  cifar_dense::AppData* appdata_ptr;
};

// Global atomic flag to control threads
std::atomic<bool> done(false);
std::mutex mtx;

constexpr int N = 640 * 480;

// ---------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------

void producer(moodycamel::ConcurrentQueue<Task>& queue,
              int num_tasks,
              std::vector<Task>& tasks,
              cudaStream_t stream) {
  for (int i = 0; i < num_tasks; ++i) {
    // CUDA_CHECK(cudaStreamAttachMemAsync(
    //     stream, tasks[i].u_data, 0, cudaMemAttachHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // kernelA_CPU(&hostParams);

#pragma omp parallel
    {
      cifar_dense::omp::process_stage_1(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_2(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_3(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_4(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_5(*tasks[i].appdata_ptr);
    }

    // Enqueue task
    queue.enqueue(tasks[i]);
  }

  // Signal consumer to stop
  done = true;
}

// ---------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------

void consumer(moodycamel::ConcurrentQueue<Task>& queue, cudaStream_t stream) {
  while (!done) {
    Task task;
    if (queue.try_dequeue(task)) {
      // CUDA_CHECK(cudaStreamAttachMemAsync(
      //     stream, task.u_data, 0, cudaMemAttachGlobal));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      // // Process the task on GPU
      // kernelB_GPU<<<1, 256>>>(task.u_data, N);
      // kernelC_GPU<<<1, 256>>>(task.u_data, N);

      cifar_dense::cuda::process_stage_6(*task.appdata_ptr);
      cifar_dense::cuda::process_stage_7(*task.appdata_ptr);
      cifar_dense::cuda::process_stage_8(*task.appdata_ptr);
      cifar_dense::cuda::process_stage_9(*task.appdata_ptr);

    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

// ---------------------------------------------------------------------
// 2 stage pipeline
// ---------------------------------------------------------------------

void run_2_stage() {
  moodycamel::ConcurrentQueue<Task> q_AB;
  const int num_tasks = 20;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // 1. prepare tasks
  auto mr = cuda::CudaMemoryResource();
  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    tasks[i].uid = i;
    tasks[i].appdata_ptr = new cifar_dense::AppData(&mr);
  }

  // 2. Start producer and consumer threads

  auto start = std::chrono::high_resolution_clock::now();

  std::thread producer_thread(
      producer, std::ref(q_AB), num_tasks, std::ref(tasks), stream);
  std::thread consumer_thread(consumer, std::ref(q_AB), stream);

  // 3. Join threads
  producer_thread.join();
  consumer_thread.join();

  auto end = std::chrono::high_resolution_clock::now();
  auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << " --- Total time taken: " << total_ms << " ms" << std::endl;
  std::cout << " --- Average time per task: " << total_ms / num_tasks << " ms"
            << std::endl;

  // 4. Free all pinned memory at the end
  for (int i = 0; i < num_tasks; ++i) {
    delete tasks[i].appdata_ptr;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char* argv[]) {
  // spdlog::set_level(spdlog::level::trace);

  run_2_stage();

  std::cout << "All tasks processed and memory freed." << std::endl;
  return 0;
}
