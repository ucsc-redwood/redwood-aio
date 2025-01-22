
#include <concurrentqueue.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

inline void checkCuda(cudaError_t result, const char* file, int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result) << "("
              << cudaGetErrorString(result) << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(x) checkCuda(x, __FILE__, __LINE__)

struct Task {
  size_t uid;
  float* input;
  float* output;
  int n;
  bool is_sentinel;
};

Task new_task(float* input, float* output, int n) {
  static size_t uid = 0;
  return Task{uid++, input, output, n, false};
}

Task new_sentinel_task() { return Task{0, nullptr, nullptr, 0, true}; }

namespace omp {

void process_stage_1(Task task) {
#pragma omp parallel for
  for (int i = 0; i < task.n; i++) {
    task.output[i] = task.input[i] + 1.0f;
  }
}

}  // namespace omp

namespace cuda {

__global__ void kernel_process_stage_1(float* input, float* output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] += input[i] + 1.0f;
  }
}

void process_stage_1(Task task) {
  constexpr auto threads_per_block = 256;
  const auto blocks = (task.n + threads_per_block - 1) / threads_per_block;

  kernel_process_stage_1<<<blocks, threads_per_block>>>(
      task.input, task.output, task.n);
  CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace cuda

int main() {
  constexpr auto n = 1'000'000;

  std::queue<Task> q_A;
  moodycamel::ConcurrentQueue<Task> q_AB;
  std::queue<Task> q_B;

  std::mutex m;

  constexpr auto n_tasks = 100;
  for (int i = 0; i < n_tasks; i++) {
    float* u_input;
    CHECK_CUDA(cudaMallocManaged(&u_input, n * sizeof(float)));
    std::fill_n(u_input, n, 1.0f);

    float* u_output;
    CHECK_CUDA(cudaMallocManaged(&u_output, n * sizeof(float)));

    q_A.push(new_task(u_input, u_output, n));
  }
  q_A.push(new_sentinel_task());

  std::thread t_omp([&]() {
    while (true) {
      Task task = q_A.front();
      q_A.pop();
      if (task.is_sentinel) {
        q_AB.enqueue(task);
        break;
      }

      {
        std::lock_guard<std::mutex> lock(m);
        std::cout << "[A] starting task " << task.uid << std::endl;
      }

      omp::process_stage_1(task);

      q_AB.enqueue(task);
    }
  });

  std::thread t_cuda([&]() {
    while (true) {
      Task task;
      if (q_AB.try_dequeue(task)) {
        if (task.is_sentinel) {
          break;
        }

        {
          std::lock_guard<std::mutex> lock(m);
          std::cout << "[B] starting task " << task.uid << std::endl;
        }

        cuda::process_stage_1(task);

        q_B.push(task);
      }
    }
  });

  t_omp.join();
  t_cuda.join();

  // Cleanup, for each task in queue_B
  while (q_B.size() > 0) {
    Task task = q_B.front();
    q_B.pop();
    CHECK_CUDA(cudaFree(task.input));
    CHECK_CUDA(cudaFree(task.output));
  }

  std::cout << "Done" << std::endl;

  return 0;
}