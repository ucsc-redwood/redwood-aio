#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"

struct Task {
  uint32_t uid;
  std::vector<float> data;
  bool is_sentinel = false;

  // debugging
};

Task new_task(const size_t size) {
  Task task;
  static uint32_t uid_counter = 0;
  task.uid = uid_counter++;
  task.data.resize(size);
  std::iota(task.data.begin(), task.data.end(), 0.0f);
  return task;
}

Task new_sentinel() {
  Task task;
  task.is_sentinel = true;
  return task;
}

void process_task_stage_A(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    task.data[i] += 9;
  }
}

void process_task_stage_B(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    task.data[i] += 1000000;
  }
}

void process_task_stage_C(Task& task) {
#pragma omp for
  for (size_t i = 0; i < task.data.size(); ++i) {
    task.data[i] += 40000000;
  }
}

template <typename T, size_t Size>
class SPSCQueue {
  static_assert((Size & (Size - 1)) == 0, "Size must be a power of 2");

 public:
  SPSCQueue() = default;
  ~SPSCQueue() = default;

  // Add a move version of enqueue
  bool enqueue(T&& item) {
    const size_t head = head_.load(std::memory_order_relaxed);
    const size_t next_head = (head + 1) & mask_;

    if (next_head == tail_.load(std::memory_order_acquire)) {
      return false;  // full
    }

    buffer_[head] = std::move(item);
    head_.store(next_head, std::memory_order_release);
    return true;
  }

  //   // Keep the copy version for backward compatibility
  //   bool enqueue(const T& item) {
  //     const size_t head = head_.load(std::memory_order_relaxed);
  //     const size_t next_head = (head + 1) & mask_;

  //     if (next_head == tail_.load(std::memory_order_acquire)) {
  //       return false;  // full
  //     }

  //     buffer_[head] = item;
  //     head_.store(next_head, std::memory_order_release);
  //     return true;
  //   }

  bool dequeue(T& item) {
    const size_t tail = tail_.load(std::memory_order_relaxed);

    if (tail == head_.load(std::memory_order_acquire)) {
      return false;  // empty
    }

    item = std::move(buffer_[tail]);
    tail_.store((tail + 1) & mask_, std::memory_order_release);
    return true;
  }

  bool empty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  bool full() const {
    const size_t next_head = (head_.load(std::memory_order_relaxed) + 1) & mask_;
    return next_head == tail_.load(std::memory_order_acquire);
  }

 private:
  static constexpr size_t mask_ = Size - 1;
  T buffer_[Size];

  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
};

std::atomic<bool> running(true);

static void persistent_thread_worker(std::vector<int>& cores,
                                     SPSCQueue<Task, 1024>& in_q,
                                     SPSCQueue<Task, 1024>* out_q,
                                     std::function<void(Task&)> process_task) {
#pragma omp parallel num_threads(cores.size())
  {
    bind_thread_to_cores(cores);

    // process the task
    while (true) {
      Task task;
      if (in_q.dequeue(task)) {
        if (task.is_sentinel) {
          // Pass the sentinel to the next stage before breaking
          if (out_q) {
            out_q->enqueue(std::move(task));
          }
          break;
        }

        process_task(task);

        // After processing, we should push the task to the next queue
        if (out_q) {
          out_q->enqueue(std::move(task));
        }

      } else {
        std::this_thread::yield();
      }
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  SPSCQueue<Task, 1024> little_queue;
  SPSCQueue<Task, 1024> medium_queue;
  SPSCQueue<Task, 1024> big_queue;

  // Master thread pushing tasks:
  for (size_t i = 0; i < 100; ++i) {
    little_queue.enqueue(new_task(1024));
  }

  little_queue.enqueue(new_sentinel());

#pragma omp parallel sections
  {
#pragma omp section
    persistent_thread_worker(g_little_cores, little_queue, &medium_queue, process_task_stage_A);

#pragma omp section
    persistent_thread_worker(g_medium_cores, medium_queue, &big_queue, process_task_stage_B);

#pragma omp section
    persistent_thread_worker(g_big_cores, big_queue, nullptr, process_task_stage_C);
  }

  // Wait for all threads to finish
  spdlog::info("Done");
  return 0;
}