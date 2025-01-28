#include <omp.h>

#include <algorithm>  // for std::min
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "cifar-dense/dense_appdata.hpp"
#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "concurrentqueue.h"  // Moodycamel's ConcurrentQueue

// // Simple check macro
// #define CUDA_CHECK(call)                                                   \
//   do {                                                                     \
//     cudaError_t _status = call;                                            \
//     if (_status != cudaSuccess) {                                          \
//       std::cerr << "Error: " << cudaGetErrorString(_status) << " at line " \
//                 << __LINE__ << std::endl;                                  \
//       exit(EXIT_FAILURE);                                                  \
//     }                                                                      \
//   } while (0)

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  int uid;
  // int* u_data;
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
  // Produce tasks using the preallocated memory
  for (int i = 0; i < num_tasks; ++i) {
    // "Initialize" each preallocated task in host memory
    // HostKernelParams hostParams;
    // hostParams.data = tasks[i].u_data;
    // hostParams.N = N;

    // CUDA_CHECK(cudaStreamAttachMemAsync(
    //     stream, tasks[i].u_data, 0, cudaMemAttachHost));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // kernelA_CPU(&hostParams);

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
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char* argv[]) {
  auto mr = cuda::CudaMemoryResource();

  moodycamel::ConcurrentQueue<Task> q_AB;
  const int num_tasks = 20;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // 1. Preallocate all tasks (along with pinned memory) at the beginning
  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    tasks[i].uid = i;
    tasks[i].appdata_ptr = new cifar_dense::AppData(&mr);
  }

  // 2. Start producer and consumer threads
  std::thread producer_thread(
      producer, std::ref(q_AB), num_tasks, std::ref(tasks), stream);
  std::thread consumer_thread(consumer, std::ref(q_AB), stream);

  // 3. Join threads
  producer_thread.join();
  consumer_thread.join();

  // 4. Free all pinned memory at the end
  for (int i = 0; i < num_tasks; ++i) {
    delete tasks[i].appdata_ptr;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));

  std::cout << "All tasks processed and memory freed." << std::endl;
  return 0;
}
