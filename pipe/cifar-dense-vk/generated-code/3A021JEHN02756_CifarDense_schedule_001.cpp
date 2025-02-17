// Auto-generated code for schedule: 3A021JEHN02756_CifarDense_schedule_001
// Device ID: 3A021JEHN02756

#include "3A021JEHN02756_CifarDense_schedule_001.hpp"

#include <atomic>
#include <thread>

#include "../run_stages.hpp"

namespace device_3A021JEHN02756 {
namespace CifarDense_schedule_001 {

// --------------------------
// Reference-Counting Globals
// --------------------------
static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

// --------------------------
// Stage 1
// --------------------------
// Increments tasks_in_flight for each new task entering the pipeline.
// Does NOT set done; we rely on the last stage to set it once all tasks are drained.
void stage_group_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    // Stage 1 processing:
    run_stages<1, 3, ProcessorType::kBigCore, 2>(task.app_data);

    // Each new task entering pipeline => increment
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);

    // Enqueue to next stage
    out_q.enqueue(task);
  }
}

// --------------------------
// Stage 2
// --------------------------
// Middle stage: no increment/decrement.
void stage_group_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                        moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      // GPU processing
      run_gpu_stages<4, 6>(task.app_data);

      // Pass along
      out_q.enqueue(task);
    } else {
      // Nothing in queue, yield
      std::this_thread::yield();
    }
  }

  // Drain leftover tasks if any remain in the queue
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<4, 6>(task.app_data);
    out_q.enqueue(task);
  }
}

// --------------------------
// Stage 3
// --------------------------
// Another middle stage: no increment/decrement.
void stage_group_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                        moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 7, ProcessorType::kMediumCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }

  // Drain leftover tasks
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<7, 7, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

// --------------------------
// Stage 4 (Last Stage)
// --------------------------
// The last stage decrements tasks_in_flight after each task is fully processed.
// If tasks_in_flight goes to zero, set done = true.
void stage_group_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      // Final processing
      run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);

      // Collect in output
      out_tasks.push_back(task);

      // Decrement tasks_in_flight; if 0 => pipeline done
      int remaining = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (remaining == 0) {
        done.store(true, std::memory_order_release);
      }
    } else {
      std::this_thread::yield();
    }
  }

  // Drain leftover tasks if any
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;

    run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);

    int remaining = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (remaining == 0) {
      done.store(true, std::memory_order_release);
    }
  }
}

// --------------------------
// Main Pipeline
// --------------------------
void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  // Reset the globals in case run_pipeline is called multiple times
  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

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

}  // end namespace CifarDense_schedule_001
}  // end namespace device_3A021JEHN02756
