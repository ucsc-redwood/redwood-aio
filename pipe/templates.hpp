#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <queue>

template <typename Task>
void run_pipelined_schedule(std::function<std::queue<Task>(size_t)> init_func,
                            std::function<void(std::queue<Task>&, std::queue<Task>&)> pipeline_func,
                            std::function<void(std::queue<Task>&)> cleanup_func) {
  constexpr auto num_tasks = 20;
  auto tasks = init_func(num_tasks);
  std::queue<Task> out_tasks;

  const auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  pipeline_func(tasks, out_tasks);
  // ---------------------------------------------------------------------

  const auto end = std::chrono::high_resolution_clock::now();

  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  const double avg_time = duration.count() / static_cast<double>(num_tasks);

  std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

  cleanup_func(tasks);
}