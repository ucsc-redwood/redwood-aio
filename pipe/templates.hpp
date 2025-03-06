#pragma once

#include <concurrentqueue.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <queue>

#include "builtin-apps/app.hpp"

// ---------------------------------------------------------------------
// queue version
// ---------------------------------------------------------------------

template <typename Task>
void run_warmup(std::function<std::queue<Task>(size_t)> init_func,
                std::function<void(std::queue<Task>&, std::queue<Task>&)> pipeline_func,
                std::function<void(std::queue<Task>&)> cleanup_func) {
  // temporarily disable logging for warmup
  spdlog::set_level(spdlog::level::off);

  constexpr auto num_tasks = 5;
  auto in_q = init_func(num_tasks);
  std::queue<Task> out_q;

  // -------------------  run the pipeline  ------------------------------
  pipeline_func(in_q, out_q);
  // ---------------------------------------------------------------------

  cleanup_func(out_q);
  // restore original log level
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
}

template <typename Task>
void run_pipelined_schedule(std::function<std::queue<Task>(size_t)> init_func,
                            std::function<void(std::queue<Task>&, std::queue<Task>&)> pipeline_func,
                            std::function<void(std::queue<Task>&)> cleanup_func) {
  constexpr auto num_tasks = 20;
  auto in_q = init_func(num_tasks);
  std::queue<Task> out_q;

  const auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  pipeline_func(in_q, out_q);
  // ---------------------------------------------------------------------

  const auto end = std::chrono::high_resolution_clock::now();

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const double avg_time = duration.count() / static_cast<double>(num_tasks);

  std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

  cleanup_func(out_q);
}

// ---------------------------------------------------------------------
// Building Block Functions
// ---------------------------------------------------------------------

template <typename TaskType, typename Callable>
void chunk_first(std::queue<TaskType>& in_tasks,
                 moodycamel::ConcurrentQueue<TaskType>& out_q,
                 Callable stage_func) {
  while (!in_tasks.empty()) {
    // Move the front element out of the queue, then pop
    TaskType task = std::move(in_tasks.front());
    in_tasks.pop();

    if (task.is_sentinel()) {
      // Pass the sentinel along
      out_q.enqueue(std::move(task));
      continue;
    }

    // ---------------------------------------------------------------------
    // Call the user-provided function that does the stage work
    stage_func(task);
    // ---------------------------------------------------------------------

    out_q.enqueue(std::move(task));
  }
}

template <typename TaskType, typename Callable>
void chunk_middle(moodycamel::ConcurrentQueue<TaskType>& in_q,
                  moodycamel::ConcurrentQueue<TaskType>& out_q,
                  Callable stage_func) {
  while (true) {
    TaskType task;
    // Non-blocking dequeue
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        // Pass the sentinel along
        out_q.enqueue(std::move(task));
        break;
      }
      // ---------------------------------------------------------------------
      stage_func(task);
      // ---------------------------------------------------------------------

      out_q.enqueue(std::move(task));
    } else {
      std::this_thread::yield();
    }
  }
}

template <typename TaskType, typename Callable>
void chunk_last(moodycamel::ConcurrentQueue<TaskType>& in_q,
                std::queue<TaskType>& out_tasks,
                Callable stage_func) {
  while (true) {
    TaskType task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        // Store sentinel and break the loop
        out_tasks.push(std::move(task));
        break;
      }
      // ---------------------------------------------------------------------
      stage_func(task);
      // ---------------------------------------------------------------------

      out_tasks.push(std::move(task));
    } else {
      std::this_thread::yield();
    }
  }
}

template <typename TaskType, typename Callable>
void chunk_single(std::queue<TaskType>& in_tasks,
                  std::queue<TaskType>& out_tasks,
                  Callable stage_func) {
  while (!in_tasks.empty()) {
    // Move the front element out of the queue, then pop
    TaskType task = std::move(in_tasks.front());
    in_tasks.pop();

    if (task.is_sentinel()) {
      // Pass the sentinel along
      out_tasks.push(std::move(task));
      continue;
    }

    // ---------------------------------------------------------------------
    // Call the user-provided function that does the stage work
    stage_func(task);
    // ---------------------------------------------------------------------

    out_tasks.push(std::move(task));
  }
}
