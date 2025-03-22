#pragma once

#include <concurrentqueue.h>

#include <functional>

template <typename TaskType, typename AppDataType>
void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
           moodycamel::ConcurrentQueue<TaskType *> *q_next,
           std::function<void(AppDataType &)> func) {
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
      func(*task->data);
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
