#pragma once

#include <concurrentqueue.h>

#include <functional>
#include <memory_resource>

#include "builtin-apps/common/cuda/manager.cuh"

template <typename AppDataType>
[[nodiscard]] std::vector<AppDataType> init_appdata(std::pmr::memory_resource *mr,
                                                    const int num_tasks) {
  std::vector<AppDataType> all_data;
  all_data.reserve(num_tasks);
  for (size_t i = 0; i < num_tasks; ++i) {
    all_data.emplace_back(mr);  // Each has big vectors
  }
  return all_data;
}

template <typename TaskType, typename AppDataType>
void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
           moodycamel::ConcurrentQueue<TaskType *> *q_next,
           std::function<void(AppDataType &, cuda::CudaManager &)> func,
           cuda::CudaManager &mgr) {
  while (true) {
    TaskType *task = nullptr;
    if (q_cur.try_dequeue(task)) {
      if (task == nullptr) {
        // Sentinel => pass it on if there's a next queue and stop
        if (q_next != nullptr) {
          q_next->enqueue(nullptr);
        }
        break;
      }

      // -----------------------------------
      func(*task->data, mgr);
      // -----------------------------------

      // If there's a next queue, pass the task along
      if (q_next != nullptr) {
        q_next->enqueue(task);
      }
    } else {
      std::this_thread::yield();
    }
  }
}
