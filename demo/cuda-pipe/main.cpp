#include <concurrentqueue.h>

#include <CLI/CLI.hpp>
#include <affinity.hpp>
#include <cifar_dense_kernel.hpp>
#include <cifar_sparse_kernel.hpp>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

#include "../conf.hpp"

Device g_device;

struct Task {
  uint32_t uid;
  cifar_dense::AppData* app_data;
  bool is_sentinel;
};

Task new_task(cifar_dense::AppData* app_data) {
  static uint32_t uid = 0;
  Task task;
  task.uid = uid++;
  task.app_data = app_data;
  task.is_sentinel = false;
  return task;
}

Task new_sentinel_task() {
  Task task;
  task.uid = 0;
  task.app_data = nullptr;
  task.is_sentinel = true;
  return task;
}




int main(int argc, char** argv) {
  std::string device_id;

  CLI::App app{"Cifar Dense Benchmark"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  std::cout << "Device ID: " << device_id << std::endl;

  g_device = get_device(device_id);

  // ---
  constexpr auto n_tasks = 100;

  moodycamel::ConcurrentQueue<Task> task_queue(n_tasks + 1);

  auto mr = std::pmr::new_delete_resource();
  std::vector<cifar_dense::AppData> app_data{
      cifar_dense::AppData(mr),
      cifar_dense::AppData(mr),
  };

  // populate the queue with tasks
  for (size_t i = 0; i < n_tasks; ++i) {
    auto& data = app_data[i % app_data.size()];
    task_queue.enqueue(new_task(&data));
  }
  task_queue.enqueue(new_sentinel_task());

  std::thread stage_A_thread([&]() {
    while (true) {
      Task task;
      if (task_queue.try_dequeue(task)) {
        if (task.is_sentinel) {
          break;
        }

        //   std::cout << "Processing task " << task.uid
        //             << " AppData's address: " << task.app_data << std::endl;
      }
    }
  });

  stage_A_thread.join();

  std::cout << "Done processing tasks" << std::endl;

  return 0;
}
