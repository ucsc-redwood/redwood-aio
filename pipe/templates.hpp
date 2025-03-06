#pragma once

#include <concurrentqueue.h>

#include <chrono>
#include <functional>
#include <iostream>
#include <queue>
#include <vector>

#include "builtin-apps/conf.hpp"

template <int Stage>
concept ValidStage = (Stage >= 1) && (Stage <= 9);

template <int Start, int End>
concept ValidStageRange = ValidStage<Start> && ValidStage<End> && (Start <= End);

template <ProcessorType processor_type>
concept ValidProcessorType =
    (processor_type == ProcessorType::kLittleCore) ||
    (processor_type == ProcessorType::kMediumCore) || (processor_type == ProcessorType::kBigCore);

/**
 * @brief Run a pipelined schedule with initialization, pipeline execution, and cleanup
 *
 * Example usage:
 * @code
 * // Run pipeline for PC device
 * run_pipelined_schedule<Task>(init_tasks, device_pc::run_pipeline, cleanup);
 *
 * // Run pipeline for Jetson device
 * run_pipelined_schedule<Task>(init_tasks, device_jetson::run_pipeline, cleanup);
 * @endcode
 *
 * The pipeline takes:
 * - An initialization function that creates the task queue
 * - A pipeline function that processes tasks through stages
 * - A cleanup function to free resources
 *
 * It measures and reports average execution time per task.
 */

template <typename Task>
void run_pipelined_schedule(
    std::function<std::vector<Task>(size_t)> init_func,
    std::function<void(std::vector<Task>&, std::vector<Task>&)> pipeline_func) {
  constexpr auto num_tasks = 20;
  auto tasks = init_func(num_tasks);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  const auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  pipeline_func(tasks, out_tasks);
  // ---------------------------------------------------------------------

  const auto end = std::chrono::high_resolution_clock::now();

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const double avg_time = duration.count() / static_cast<double>(num_tasks);

  std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

  //   cleanup_func(out_tasks);
  out_tasks.clear();
}

// ---------------------------------------------------------------------
// queue version
// ---------------------------------------------------------------------

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
