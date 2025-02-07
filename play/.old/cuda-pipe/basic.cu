#include <algorithm>  // for std::min
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include "concurrentqueue.h"  // Include Moodycamel's ConcurrentQueue

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

  // if (idx == 0) {
  //   printf("\tgpu A\n");
  // }

  if (idx < N) {
    data[idx] = idx;
  }
}

__global__ void kernelB_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // if (idx == 0) {
  //   printf("\tgpu B\n");
  // }

  if (idx < N) {
    data[idx] += 10;
  }
}

__global__ void kernelC_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // if (idx == 0) {
  //   printf("\tgpu C\n");
  // }

  if (idx < N) {
    data[idx] *= 2;
  }
}

// ---------------------------------------------------------------------
// CPU (host) versions of the same "kernels."
// We'll call them from a Host Node in the CUDA Graph.
// ---------------------------------------------------------------------
struct HostKernelParams {
  int* data;  // Points to pinned host memory
  int N;
};

static void kernelA_CPU(void* userData) {
  // std::cout << "\tcpu A" << std::endl;

  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] = i;
  }
}

static void kernelB_CPU(void* userData) {
  // std::cout << "\tcpu B" << std::endl;

  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] += 10;
  }
}

static void kernelC_CPU(void* userData) {
  // std::cout << "\tcpu C" << std::endl;

  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] *= 2;
  }
}

// ---------------------------------------------------------------------
// Task
// ---------------------------------------------------------------------

struct Task {
  int uid;
  int* h_data;
  int* d_data;
};

// Global atomic flag to control threads
std::atomic<bool> done(false);
std::mutex mtx;

constexpr int N = 640 * 480;

void producer(moodycamel::ConcurrentQueue<Task>& queue, int num_tasks) {
  for (int i = 0; i < num_tasks; ++i) {
    int* h_data = nullptr;
    CHECK_CUDA(cudaHostAlloc(&h_data, N * sizeof(int), cudaHostAllocMapped));

    int* d_data = nullptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&d_data, h_data, 0));

    Task task{i, h_data, d_data};

    HostKernelParams hostParams;
    hostParams.data = h_data;
    hostParams.N = N;

    kernelA_CPU(&hostParams);

    queue.enqueue(task);

    // {
    //   std::lock_guard<std::mutex> lock(mtx);
    //   std::cout << "Produced Task ID: " << i << "\n";
    // }
  }

  done = true;  // Signal consumer to stop
}

void consumer(moodycamel::ConcurrentQueue<Task>& queue) {
  // while (!done || !queue.isEmpty()) {

  while (!done) {
    Task task;
    if (queue.try_dequeue(task)) {
      // {
      //   std::lock_guard<std::mutex> lock(mtx);
      //   std::cout << "Consumed Task ID: " << task.uid << "\n";
      // }

      // kernelA_GPU<<<1, 256>>>(task.d_data, N);
      kernelB_GPU<<<1, 256>>>(task.d_data, N);
      kernelC_GPU<<<1, 256>>>(task.d_data, N);
      cudaDeviceSynchronize();

      // Clean up dynamically allocated data
      CHECK_CUDA(cudaFreeHost(task.h_data));
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

int main(int argc, char* argv[]) {
  moodycamel::ConcurrentQueue<Task> queue;
  const int num_tasks = 20;

  // Start producer and consumer threads
  std::thread producer_thread(producer, std::ref(queue), num_tasks);
  std::thread consumer_thread(consumer, std::ref(queue));

  // Join threads
  producer_thread.join();
  consumer_thread.join();

  std::cout << "All tasks processed." << std::endl;

  return 0;
}
