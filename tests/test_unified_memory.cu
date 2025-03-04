#include <concurrentqueue.h>
#include <omp.h>

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#define CUDA_CHECK(call)                                                                  \
  do {                                                                                    \
    cudaError_t error = call;                                                             \
    if (error != cudaSuccess) {                                                           \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                                                 \
    }                                                                                     \
  } while (0)

__global__ void k_Doubler(float* buffer, size_t n) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < n) {
    buffer[tid] *= 2.0f;
  }
}

struct Task {
  uint32_t uid;
  float* buffer;
  size_t n;
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return done; }
};

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  std::queue<Task> tasks;

  static uint32_t uid_counter = 0;

  constexpr auto n = 1024;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task;
    CUDA_CHECK(cudaMallocManaged(&task.buffer, n * sizeof(float)));
    task.n = n;
    task.done = false;
    task.uid = uid_counter++;
    std::fill_n(task.buffer, n, 1.0f);

    tasks.push(std::move(task));
  }

  // create a sentinel task
  Task sentinel{
      .uid = uid_counter++,
      .buffer = nullptr,
      .n = 0,
      .done = true,
  };
  tasks.push(std::move(sentinel));

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  while (!tasks.empty()) {
    auto& task = tasks.front();

    if (!task.is_sentinel()) {
      // print first 10 elements
      std::cout << "Task " << task.uid << ":\n";
      for (auto i = 0; i < 10; ++i) {
        std::cout << task.buffer[i] << " ";
      }
      std::cout << "\n";
    }

    if (task.buffer) {
      CUDA_CHECK(cudaFree(task.buffer));
    }
    tasks.pop();
  }
}

void producer(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();

    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      break;
    }

    // ---------------------------------------------------------------------
#pragma omp parallel for
    for (auto i = 0; i < task.n; ++i) {
      task.buffer[i] += 1000.0f;
    }
    // ---------------------------------------------------------------------

    out_q.enqueue(task);
    in_tasks.pop();
  }
}

void consumer(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      // ---------------------------------------------------------------------
      k_Doubler<<<4, 256>>>(task.buffer, task.n);
      CUDA_CHECK(cudaDeviceSynchronize());
      // ---------------------------------------------------------------------

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

int main() {
  constexpr auto n = 1024;

  auto in_tasks = init_tasks(5);
  moodycamel::ConcurrentQueue<Task> q;
  std::queue<Task> out_tasks;

  // Start producer and consumer threads
  std::thread p1(producer, std::ref(in_tasks), std::ref(q));
  std::thread c1(consumer, std::ref(q), std::ref(out_tasks));

  p1.join();
  c1.join();

  cleanup(out_tasks);

  return 0;
}