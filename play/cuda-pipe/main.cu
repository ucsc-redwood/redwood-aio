#include <algorithm>  // for std::min
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "concurrentqueue.h"  // Moodycamel's ConcurrentQueue

// Simple check macro
#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t _status = call;                                            \
    if (_status != cudaSuccess) {                                          \
      std::cerr << "Error: " << cudaGetErrorString(_status) << " at line " \
                << __LINE__ << std::endl;                                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------

__global__ void kernelA_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = idx;
  }
}

__global__ void kernelB_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] += 10;
  }
}

__global__ void kernelC_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] *= 2;
  }
}

// ---------------------------------------------------------------------
// CPU (host) versions of the same "kernels."
// ---------------------------------------------------------------------
struct HostKernelParams {
  int* data;  // Points to pinned host memory
  int N;
};

static void kernelA_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] = i;
  }
}

static void kernelB_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] += 10;
  }
}

static void kernelC_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] *= 2;
  }
}

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  int uid;
  // int* h_data;  // pinned host memory
  // int* d_data;  // device pointer (mapped to pinned host memory)
  int* u_data;
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
              std::vector<Task>& tasks) {
  // Produce tasks using the preallocated memory
  for (int i = 0; i < num_tasks; ++i) {
    // "Initialize" each preallocated task in host memory
    HostKernelParams hostParams;
    hostParams.data = tasks[i].u_data;
    hostParams.N = N;

    CHECK_CUDA(cudaStreamAttachMemAsync(
        nullptr, tasks[i].u_data, 0, cudaMemAttachHost));
    cudaDeviceSynchronize();

    kernelA_CPU(&hostParams);

    // Enqueue task
    queue.enqueue(tasks[i]);
  }

  // Signal consumer to stop
  done = true;
}

// ---------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------

void consumer(moodycamel::ConcurrentQueue<Task>& queue) {
  // The consumer continues while not done.
  // If no tasks are available, yield.
  while (!done) {
    Task task;
    if (queue.try_dequeue(task)) {
      CHECK_CUDA(cudaStreamAttachMemAsync(
          nullptr, task.u_data, 0, cudaMemAttachGlobal));
      // CHECK_CUDA(cudaStreamSynchronize(nullptr));
      cudaDeviceSynchronize();

      // Process the task on GPU
      kernelB_GPU<<<1, 256>>>(task.u_data, N);
      kernelC_GPU<<<1, 256>>>(task.u_data, N);
      cudaDeviceSynchronize();

      // IMPORTANT: We do NOT free pinned memory here anymore.
      // We'll free it after joining the threads, in main().
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
  moodycamel::ConcurrentQueue<Task> queue;
  const int num_tasks = 20;


  // 1. Preallocate all tasks (along with pinned memory) at the beginning
  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    // int* h_data = nullptr;
    // CHECK_CUDA(cudaHostAlloc(&h_data, N * sizeof(int), cudaHostAllocMapped));

    // int* d_data = nullptr;
    // CHECK_CUDA(cudaHostGetDevicePointer(&d_data, h_data, 0));

    int* u_data = nullptr;
    CHECK_CUDA(cudaMallocManaged(&u_data, N * sizeof(int)));
    CHECK_CUDA(
        cudaStreamAttachMemAsync(nullptr, u_data, 0, cudaMemAttachGlobal));

    tasks[i].uid = i;
    tasks[i].u_data = u_data;
  }

  // 2. Start producer and consumer threads
  std::thread producer_thread(
      producer, std::ref(queue), num_tasks, std::ref(tasks));
  std::thread consumer_thread(consumer, std::ref(queue));

  // 3. Join threads
  producer_thread.join();
  consumer_thread.join();

  // 4. Free all pinned memory at the end
  for (int i = 0; i < num_tasks; ++i) {
    CHECK_CUDA(cudaFree(tasks[i].u_data));
  }

  std::cout << "All tasks processed and memory freed." << std::endl;
  return 0;
}
