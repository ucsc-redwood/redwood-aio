// #pragma once

// #include <chrono>
// #include <functional>
// #include <iostream>
// #include <vector>
// #include <queue>

// /**
//  * @brief Run a pipelined schedule with initialization, pipeline execution, and cleanup
//  * 
//  * Example usage:
//  * @code
//  * // Run pipeline for PC device
//  * run_pipelined_schedule<Task>(init_tasks, device_pc::run_pipeline, cleanup);
//  * 
//  * // Run pipeline for Jetson device 
//  * run_pipelined_schedule<Task>(init_tasks, device_jetson::run_pipeline, cleanup);
//  * @endcode
//  * 
//  * The pipeline takes:
//  * - An initialization function that creates the task queue
//  * - A pipeline function that processes tasks through stages
//  * - A cleanup function to free resources
//  * 
//  * It measures and reports average execution time per task.
//  */

// template <typename Task>
// void run_pipelined_schedule(std::function<std::vector<Task>(size_t)> init_func,
//                             std::function<void(std::vector<Task>&, std::vector<Task>&)> pipeline_func,
//                             std::function<void(std::vector<Task>&)> cleanup_func) {
//   constexpr auto num_tasks = 20;
//   auto tasks = init_func(num_tasks);
//   std::vector<Task> out_tasks;
//   out_tasks.reserve(tasks.size());

//   const auto start = std::chrono::high_resolution_clock::now();

//   // -------------------  run the pipeline  ------------------------------
//   pipeline_func(tasks, out_tasks);
//   // ---------------------------------------------------------------------

//   const auto end = std::chrono::high_resolution_clock::now();

//   const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//   const double avg_time = duration.count() / static_cast<double>(num_tasks);

//   std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

//   cleanup_func(out_tasks);
// }