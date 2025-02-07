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

void init_tasks(std::queue<Task>& q_A,
                int num_tasks,
                std::pmr::memory_resource* mr) {
  for (int i = 0; i < num_tasks; ++i) {
    Task t;
    t.uid = i;
    t.appdata_ptr = new cifar_dense::AppData(mr);
    q_A.push(t);
  }
}

// ---------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------

void producer(moodycamel::ConcurrentQueue<Task>& queue,
              int num_tasks,
              std::vector<Task>& tasks) {
  for (int i = 0; i < num_tasks; ++i) {
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
// Baseline
// ---------------------------------------------------------------------

void run_baseline(int num_tasks) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  std::queue<Task> q_A;
  std::queue<Task> q_B;

  init_tasks(q_A, num_tasks, mr);

  auto start = std::chrono::high_resolution_clock::now();

  while (!q_A.empty()) {
    Task t = std::move(q_A.front());
    q_A.pop();

#pragma omp parallel
    {
      cifar_dense::omp::process_stage_1(*t.appdata_ptr);
      cifar_dense::omp::process_stage_2(*t.appdata_ptr);
      cifar_dense::omp::process_stage_3(*t.appdata_ptr);
      cifar_dense::omp::process_stage_4(*t.appdata_ptr);
      cifar_dense::omp::process_stage_5(*t.appdata_ptr);
      cifar_dense::omp::process_stage_6(*t.appdata_ptr);
      cifar_dense::omp::process_stage_7(*t.appdata_ptr);
      cifar_dense::omp::process_stage_8(*t.appdata_ptr);
      cifar_dense::omp::process_stage_9(*t.appdata_ptr);
    }

    q_B.push(t);
  }

  auto end = std::chrono::high_resolution_clock::now();

  while (!q_B.empty()) {
    Task t = std::move(q_B.front());
    q_B.pop();
    delete t.appdata_ptr;
  }

  auto total_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << " --- Total time taken: " << total_ms << " ms" << std::endl;
  std::cout << " --- Average time per task: " << total_ms / num_tasks << " ms"
            << std::endl;

  std::cout << "[run_basline] done" << std::endl;
}

// ---------------------------------------------------------------------
// 2 stage CPU pipeline
// ---------------------------------------------------------------------

void run_2_stage(int num_tasks) {
  moodycamel::ConcurrentQueue<Task> q_AB;

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

  std::cout << "[run_2_stage] done" << std::endl;
}

// ---------------------------------------------------------------------
// 2 stage CPU + 1 GPU pipeline
// ---------------------------------------------------------------------

void stage_group_1(moodycamel::ConcurrentQueue<Task>& queue,
                   int num_tasks,
                   std::vector<Task>& tasks) {
  const auto g_big_core_size = g_big_cores.size();

  for (int i = 0; i < num_tasks; ++i) {
#pragma omp parallel num_threads(g_big_core_size)
    {
      bind_thread_to_core(g_big_cores);
      cifar_dense::omp::process_stage_1(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_2(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_3(*tasks[i].appdata_ptr);
      cifar_dense::omp::process_stage_4(*tasks[i].appdata_ptr);
    }

    queue.enqueue(tasks[i]);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_2(moodycamel::ConcurrentQueue<Task>& queue_ab) {
  // GPU 5-6
  while (!done) {
    Task task;
    if (queue_ab.try_dequeue(task)) {
      cifar_dense::vulkan::Singleton::getInstance().process_stage_5(
          *task.appdata_ptr);
      cifar_dense::vulkan::Singleton::getInstance().process_stage_6(
          *task.appdata_ptr);
      cifar_dense::vulkan::Singleton::getInstance().process_stage_7(
          *task.appdata_ptr);
    } else {
      std::this_thread::yield();
    }
  }
}

void stage_group_3(moodycamel::ConcurrentQueue<Task>& queue_bc) {
  // CPU little core 8-9
  const auto g_medium_core_size = g_medium_cores.size();

  while (!done) {
    Task task;
    if (queue_bc.try_dequeue(task)) {
#pragma omp parallel num_threads(g_medium_core_size)
      {
        bind_thread_to_core(g_medium_cores);
        cifar_dense::omp::process_stage_8(*task.appdata_ptr);
        cifar_dense::omp::process_stage_9(*task.appdata_ptr);
      }

    } else {
      std::this_thread::yield();
    }
  }
}

void run_3_stage(int num_tasks) {
  moodycamel::ConcurrentQueue<Task> q_AB;
  moodycamel::ConcurrentQueue<Task> q_BC;

  // 1. prepare tasks
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  std::vector<Task> tasks(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    tasks[i].uid = i;
    tasks[i].appdata_ptr = new cifar_dense::AppData(mr);
  }

  // 2. Start producer and consumer threads

  auto start = std::chrono::high_resolution_clock::now();

  std::thread stage_group_1_thread(
      stage_group_1, std::ref(q_AB), num_tasks, std::ref(tasks));
  std::thread stage_group_2_thread(stage_group_2, std::ref(q_AB));
  std::thread stage_group_3_thread(stage_group_3, std::ref(q_BC));

  // 3. Join threads
  stage_group_1_thread.join();
  stage_group_2_thread.join();
  stage_group_3_thread.join();

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

  std::cout << "[run_3_stage] done" << std::endl;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  run_baseline(20);
  run_2_stage(20);
  // run_3_stage(20);
  return 0;
}
