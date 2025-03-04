#include <concurrentqueue.h>
#include <gtest/gtest.h>

#include <queue>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "spdlog/common.h"

#define PREPARE_DATA                   \
  cifar_dense::AppData appdata(&g_mr); \
  CUDA_CHECK(cudaDeviceSynchronize());

struct Task {
  cifar_dense::AppData* app_data;  // basically just a pointer
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks);

void cleanup(std::queue<Task>& tasks);

//   auto mr = cuda::CudaMemoryResource();

// cuda::CudaMemoryResource g_mr;
cuda::CudaMemoryResource_PinnedHost g_mr;

/**
 * Initializes a queue of tasks and adds a sentinel task at the end.
 * The sentinel task (with null pointers) is used to signal the end of
 * the task stream to the pipeline stages.
 */
[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  std::queue<Task> tasks;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new cifar_dense::AppData(&g_mr),
        .done = false,
    };
    tasks.push(std::move(task));
  }

  // create a sentinel task
  Task sentinel{
      .app_data = nullptr,
      .done = true,
  };
  tasks.push(std::move(sentinel));

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (task.app_data) {
      delete task.app_data;
      task.app_data = nullptr;
    }
    tasks.pop();
  }
}

// ----------------------------------------------------------------------------
// Stages (OMP then CUDA)
// ----------------------------------------------------------------------------

TEST(CUDA_CIFAR_DENSE, Stage1_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
  }

  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);

  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage12_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
  }

  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage123_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
    cifar_dense::omp::run_stage<3>(appdata);
  }

  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage1234_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
    cifar_dense::omp::run_stage<3>(appdata);
    cifar_dense::omp::run_stage<4>(appdata);
  }

  cifar_dense::cuda::run_stage<5>(appdata);
  cifar_dense::cuda::run_stage<6>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

// ----------------------------------------------------------------------------
// Stages (CUDA then OMP)
// ----------------------------------------------------------------------------

TEST(CUDA_CIFAR_DENSE, Stage12_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<3>(appdata);
    cifar_dense::omp::run_stage<4>(appdata);
  }

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage123_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<4>(appdata);
    cifar_dense::omp::run_stage<5>(appdata);
  }

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage1234_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<5>(appdata);
    cifar_dense::omp::run_stage<6>(appdata);
  }

  SUCCEED();
}

// ----------------------------------------------------------------------------
// Queue-based Pipeline
// ----------------------------------------------------------------------------

TEST(CUDA_CIFAR_DENSE, QueuePipeline) {
  auto tasks = init_tasks(10);
  std::queue<Task> out_tasks;

  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (task.is_sentinel()) {
      out_tasks.push(task);
      tasks.pop();
      break;  // Add explicit break for sentinel
    }

    // Run OMP stages
#pragma omp parallel num_threads(g_little_cores.size())
    {
      bind_thread_to_cores(g_little_cores);
      cifar_dense::omp::run_stage<1>(*task.app_data);
      cifar_dense::omp::run_stage<2>(*task.app_data);
      cifar_dense::omp::run_stage<3>(*task.app_data);
    }

    // Run CUDA stages
    cifar_dense::cuda::run_stage<4>(*task.app_data);
    cifar_dense::cuda::run_stage<5>(*task.app_data);

    out_tasks.push(task);
    tasks.pop();
  }

  SUCCEED();

  spdlog::info("queue.size = {}", out_tasks.size());

  cleanup(out_tasks);
}

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    // ---------------------------------------------------------------------
#pragma omp parallel num_threads(g_little_cores.size())
    {
      bind_thread_to_cores(g_little_cores);
      cifar_dense::omp::run_stage<1>(*task.app_data);
      cifar_dense::omp::run_stage<2>(*task.app_data);
      cifar_dense::omp::run_stage<3>(*task.app_data);
    }
    // ---------------------------------------------------------------------

    out_q.enqueue(task);
    in_tasks.pop();
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      // ---------------------------------------------------------------------
      cifar_dense::cuda::run_stage<4>(*task.app_data);
      cifar_dense::cuda::run_stage<5>(*task.app_data);
      // ---------------------------------------------------------------------

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

TEST(CUDA_CIFAR_DENSE, QueuePipeline_TwoThreads) {
  auto tasks = init_tasks(10);
  std::queue<Task> out_tasks;

  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk4.join();

  SUCCEED();

  cleanup(out_tasks);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
  //   spdlog::set_level(spdlog::level::debug);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
