
#include <omp.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <queue>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"
#include "cifar-dense/vulkan/vk_dispatcher.hpp"
#include "concurrentqueue.h"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  int uid;
  cifar_dense::AppData* appdata_ptr;
};

std::atomic<bool> done(false);

// ---------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------

void producer(moodycamel::ConcurrentQueue<Task>& queue,
              int num_tasks,
              std::vector<Task>& tasks) {
  for (int i = 0; i < num_tasks; ++i) {
    // auto g_little_core_size = g_little_cores.size();

#pragma omp parallel
    {
      cifar_dense::omp::process_stage_1(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_2(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_3(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_4(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_5(*tasks[i].appdata_ptr);
    }

    queue.enqueue(tasks[i]);
  }

  // Signal consumer to stop
  done = true;
}

// ---------------------------------------------------------------------
// Consumer
// ---------------------------------------------------------------------

void consumer(moodycamel::ConcurrentQueue<Task>& queue) {
  while (!done) {
    Task task;
    if (queue.try_dequeue(task)) {
      cifar_dense::vulkan::Singleton::getInstance().process_stage_6(
          *task.appdata_ptr);
      cifar_dense::vulkan::Singleton::getInstance().process_stage_7(
          *task.appdata_ptr);
      cifar_dense::vulkan::Singleton::getInstance().process_stage_8(
          *task.appdata_ptr);
      cifar_dense::vulkan::Singleton::getInstance().process_stage_9(
          *task.appdata_ptr);

    } else {
      std::this_thread::yield();
    }
  }
}

// ---------------------------------------------------------------------
// 2 stage pipeline
// ---------------------------------------------------------------------

void run_2_stage() {
  moodycamel::ConcurrentQueue<Task> q_AB;
  const int num_tasks = 3;

  // 1. prepare tasks
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    tasks[i].uid = i;
    tasks[i].appdata_ptr = new cifar_dense::AppData(mr);
  }

  // 2. Start producer and consumer threads

  auto start = std::chrono::high_resolution_clock::now();

  std::thread producer_thread(
      producer, std::ref(q_AB), num_tasks, std::ref(tasks));
  std::thread consumer_thread(consumer, std::ref(q_AB));

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
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  run_2_stage();

  return 0;
}
