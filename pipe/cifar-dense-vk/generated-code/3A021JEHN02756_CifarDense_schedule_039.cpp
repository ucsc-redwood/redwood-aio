// Auto-generated code for schedule: 3A021JEHN02756_CifarDense_schedule_039
// Device ID: 3A021JEHN02756

#include "3A021JEHN02756_CifarDense_schedule_039.hpp"

#include <atomic>
#include <thread>
#include "../run_stages.hpp"

namespace device_3A021JEHN02756 {
namespace CifarDense_schedule_039 {

static std::atomic<bool> done(false);

void stage_group_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
  done.store(true, std::memory_order_release);
}

void stage_group_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<3, 6>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  // Drain any remaining tasks if needed:
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<3, 6>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 8, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  // Drain any remaining tasks if needed:
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<7, 8, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<9, 9, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
  // Drain any remaining tasks if needed:
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<9, 9, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  // Create threads
  std::thread t_chunk1(stage_group_chunk1, std::ref(tasks), std::ref(q_01));
  std::thread t_chunk2(stage_group_chunk2, std::ref(q_01), std::ref(q_12));
  std::thread t_chunk3(stage_group_chunk3, std::ref(q_12), std::ref(q_23));
  std::thread t_chunk4(stage_group_chunk4, std::ref(q_23), std::ref(out_tasks));

  // Join threads
  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // end namespace CifarDense_schedule_039
}  // end namespace device_3A021JEHN02756
