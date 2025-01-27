
#include <omp.h>
#include <spdlog/spdlog.h>

#include <memory_resource>
#include <queue>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"
#include "cifar-dense/vulkan/vk_dispatcher.hpp"
#include "concurrentqueue.h"

struct Task {
  int uid;
  std::unique_ptr<cifar_dense::AppData> app_data;
  bool is_done;
};

void init_tasks(std::queue<Task> &q_A, std::pmr::memory_resource *mr) {
  for (int i = 0; i < 10; ++i) {
    Task t;
    t.uid = i;
    t.app_data = std::make_unique<cifar_dense::AppData>(mr);
    t.is_done = false;

    // Note: we use std::move since we have a unique_ptr in Task
    q_A.push(std::move(t));
  }

  // add a sentinel task
  q_A.push(Task{0, nullptr, true});
}

void run_2_cpu_gpu_stage() {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> q_A;
  moodycamel::ConcurrentQueue<Task> q_AB;
  std::queue<Task> q_B;

  init_tasks(q_A, mr);

  std::mutex mtx;

  std::thread t_A([&]() {
    while (true) {
      Task t = std::move(q_A.front());
      q_A.pop();
      if (t.is_done) {
        q_AB.enqueue(std::move(t));
        break;
      }

#pragma omp parallel
      {
        cifar_dense::omp::process_stage_1(*t.app_data);
        cifar_dense::omp::process_stage_2(*t.app_data);
        cifar_dense::omp::process_stage_3(*t.app_data);
        cifar_dense::omp::process_stage_4(*t.app_data);
        cifar_dense::omp::process_stage_5(*t.app_data);
      }

      q_AB.enqueue(std::move(t));

      {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[A] processed task " << t.uid << std::endl;
      }
    }
  });

  std::thread t_B([&]() {
    while (true) {
      Task t;
      if (q_AB.try_dequeue(t)) {
        if (t.is_done) {
          break;
        }

        cifar_dense::vulkan::Singleton::getInstance().process_stage_6(
            *t.app_data);
        cifar_dense::vulkan::Singleton::getInstance().process_stage_7(
            *t.app_data);
        cifar_dense::vulkan::Singleton::getInstance().process_stage_8(
            *t.app_data);
        cifar_dense::vulkan::Singleton::getInstance().process_stage_9(
            *t.app_data);

        q_B.push(std::move(t));

        {
          std::lock_guard<std::mutex> lock(mtx);
          std::cout << "[B] processed task " << t.uid << std::endl;
        }
      }
    }
  });

  t_A.join();
  t_B.join();

  std::cout << "[main] done" << std::endl;
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  run_2_cpu_gpu_stage();

  return 0;
}
