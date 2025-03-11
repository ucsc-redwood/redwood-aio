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

// template <typename TaskType, typename AppDataType>
// void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
//            moodycamel::ConcurrentQueue<TaskType *> *q_next,
//            std::function<void(AppDataType *, cuda::CudaManager &)> func,
//            cuda::CudaManager &mgr) {
//   while (true) {
//     TaskType *task = nullptr;
//     if (q_cur.try_dequeue(task)) {
//       if (task == nullptr) {
//         // Sentinel => pass it on if there's a next queue and stop
//         if (q_next != nullptr) {
//           q_next->enqueue(nullptr);
//         }
//         break;
//       }

//       // -----------------------------------
//       func(task->data, mgr);
//       // -----------------------------------

//       // If there's a next queue, pass the task along
//       if (q_next != nullptr) {
//         q_next->enqueue(task);
//       }
//     } else {
//       std::this_thread::yield();
//     }
//   }
// }

// // ---------------------------------------------------------------------
// // queue version
// // ---------------------------------------------------------------------

// // template <typename Task>
// // void run_warmup(std::function<std::queue<Task>(size_t)> init_func,
// //                 std::function<void(std::queue<Task>&, std::queue<Task>&)> pipeline_func,
// //                 std::function<void(std::queue<Task>&)> cleanup_func) {
// //   // temporarily disable logging for warmup
// //   spdlog::set_level(spdlog::level::off);

// //   constexpr auto num_tasks = 5;
// //   auto in_q = init_func(num_tasks);
// //   std::queue<Task> out_q;

// //   // -------------------  run the pipeline  ------------------------------
// //   pipeline_func(in_q, out_q);
// //   // ---------------------------------------------------------------------

// //   cleanup_func(out_q);
// //   // restore original log level
// //   spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
// // }

// // template <typename Task>
// // void run_pipelined_schedule(std::function<std::queue<Task>(size_t)> init_func,
// //                             std::function<void(std::queue<Task>&, std::queue<Task>&)>
// //                             pipeline_func, std::function<void(std::queue<Task>&)>
// cleanup_func) {
// //   constexpr auto num_tasks = 20;
// //   auto in_q = init_func(num_tasks);
// //   std::queue<Task> out_q;

// //   const auto start = std::chrono::high_resolution_clock::now();

// //   // -------------------  run the pipeline  ------------------------------
// //   pipeline_func(in_q, out_q);
// //   // ---------------------------------------------------------------------

// //   const auto end = std::chrono::high_resolution_clock::now();

// //   const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// //   const double avg_time = duration.count() / static_cast<double>(num_tasks);

// //   std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

// //   cleanup_func(out_q);
// // }

// // ---------------------------------------------------------------------
// // Building Block Functions
// // ---------------------------------------------------------------------

// // template <typename TaskType, typename Callable>
// // void chunk_first(std::queue<TaskType>& in_tasks,
// //                  moodycamel::ConcurrentQueue<TaskType>& out_q,
// //                  Callable stage_func) {
// //   while (!in_tasks.empty()) {
// //     // Move the front element out of the queue, then pop
// //     TaskType task = std::move(in_tasks.front());
// //     in_tasks.pop();

// //     if (task.is_sentinel()) {
// //       // Pass the sentinel along
// //       out_q.enqueue(std::move(task));
// //       continue;
// //     }

// //     // ---------------------------------------------------------------------
// //     // Call the user-provided function that does the stage work
// //     stage_func(task);
// //     // ---------------------------------------------------------------------

// //     out_q.enqueue(std::move(task));
// //   }
// // }

// // template <typename TaskType, typename Callable>
// // void chunk_middle(moodycamel::ConcurrentQueue<TaskType>& in_q,
// //                   moodycamel::ConcurrentQueue<TaskType>& out_q,
// //                   Callable stage_func) {
// //   while (true) {
// //     TaskType task;
// //     // Non-blocking dequeue
// //     if (in_q.try_dequeue(task)) {
// //       if (task.is_sentinel()) {
// //         // Pass the sentinel along
// //         out_q.enqueue(std::move(task));
// //         break;
// //       }
// //       // ---------------------------------------------------------------------
// //       stage_func(task);
// //       // ---------------------------------------------------------------------

// //       out_q.enqueue(std::move(task));
// //     } else {
// //       std::this_thread::yield();
// //     }
// //   }
// // }

// // template <typename TaskType, typename Callable>
// // void chunk_last(moodycamel::ConcurrentQueue<TaskType>& in_q,
// //                 std::queue<TaskType>& out_tasks,
// //                 Callable stage_func) {
// //   while (true) {
// //     TaskType task;
// //     if (in_q.try_dequeue(task)) {
// //       if (task.is_sentinel()) {
// //         // Store sentinel and break the loop
// //         out_tasks.push(std::move(task));
// //         break;
// //       }
// //       // ---------------------------------------------------------------------
// //       stage_func(task);
// //       // ---------------------------------------------------------------------

// //       out_tasks.push(std::move(task));
// //     } else {
// //       std::this_thread::yield();
// //     }
// //   }
// // }

// // template <typename TaskType, typename Callable>
// // void chunk_single(std::queue<TaskType>& in_tasks,
// //                   std::queue<TaskType>& out_tasks,
// //                   Callable stage_func) {
// //   while (!in_tasks.empty()) {
// //     // Move the front element out of the queue, then pop
// //     TaskType task = std::move(in_tasks.front());
// //     in_tasks.pop();

// //     if (task.is_sentinel()) {
// //       // Pass the sentinel along
// //       out_tasks.push(std::move(task));
// //       continue;
// //     }

// //     // ---------------------------------------------------------------------
// //     // Call the user-provided function that does the stage work
// //     stage_func(task);
// //     // ---------------------------------------------------------------------

// //     out_tasks.push(std::move(task));
// //   }
// // }

// // -----------------------------------
// // Thread function for pipeline stage
// // -----------------------------------

// template <typename TaskType>
// void chunk(moodycamel::ConcurrentQueue<TaskType *> &q_cur,
//            moodycamel::ConcurrentQueue<TaskType *> *q_next,
//            const std::function<void(TaskType *)> &func) {
//   while (true) {
//     TaskType *task = nullptr;
//     if (q_cur.try_dequeue(task)) {
//       if (task == nullptr) {
//         // Sentinel => pass it on if there's a next queue and stop
//         if (q_next != nullptr) {
//           q_next->enqueue(nullptr);
//         }
//         break;
//       }

//       // -----------------------------------
//       func(task);
//       // -----------------------------------

//       // If there's a next queue, pass the task along
//       if (q_next != nullptr) {
//         q_next->enqueue(task);
//       }
//     } else {
//       std::this_thread::yield();
//     }
//   }
// }

// // ---------------------------------------------------------------------
// // Concurrent Queue Pipeline version
// // ---------------------------------------------------------------------

// // template <typename Task>
// // void run_pipelined_schedule(std::function<std::queue<Task>(size_t)> init_func,
// //                             std::function<void(std::queue<Task>&, std::queue<Task>&)>
// //                             pipeline_func, std::function<void(std::queue<Task>&)>
// cleanup_func) {
// //   constexpr auto num_tasks = 20;
// //   auto in_q = init_func(num_tasks);
// //   std::queue<Task> out_q;

// //   const auto start = std::chrono::high_resolution_clock::now();

// //   // -------------------  run the pipeline  ------------------------------
// //   pipeline_func(in_q, out_q);
// //   // ---------------------------------------------------------------------

// //   const auto end = std::chrono::high_resolution_clock::now();

// //   const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
// //   const double avg_time = duration.count() / static_cast<double>(num_tasks);

// //   std::cout << "[schedule]: Average time per iteration: " << avg_time << " ms" << std::endl;

// //   cleanup_func(out_q);
// // }

// // void program(const int num_tasks) {
// //   CudaManager mgr;

// //   std::vector<Appdata> allData = init_appdata(mgr.get_mr(), num_tasks);

// //   // ---------------------------------------------------------------------
// //   moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(allData, &mgr);
// //   moodycamel::ConcurrentQueue<Task *> q_12;
// //   moodycamel::ConcurrentQueue<Task *> q_23;
// //   moodycamel::ConcurrentQueue<Task *> q_34;

// //   auto start = std::chrono::high_resolution_clock::now();

// //   std::thread t1(chunk, std::ref(q_input), &q_12, omp::run_multiple_stages<1, 2>);
// //   std::thread t2(chunk, std::ref(q_12), &q_23, omp::run_multiple_stages<3, 4>);
// //   std::thread t3(chunk, std::ref(q_23), &q_34, cuda::run_multiple_stages<5, 6>);
// //   std::thread t4(chunk, std::ref(q_34), nullptr, omp::run_multiple_stages<7, 7>);

// //   t1.join();
// //   t2.join();
// //   t3.join();
// //   t4.join();
// //   // ---------------------------------------------------------------------
// //   auto end = std::chrono::high_resolution_clock::now();
// //   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
// //   spdlog::info("Time taken per task: {} microseconds", duration.count() / num_tasks);
// // }