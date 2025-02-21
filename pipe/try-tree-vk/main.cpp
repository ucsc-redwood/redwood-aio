#include <concurrentqueue.h>
#include <omp.h>

// #include <queue>
#include <thread>
#include <vector>

#include "run_stages.hpp"

// ---------------------------------------------------------------------
// Define a Schedule
// ---------------------------------------------------------------------

namespace TestSchedule {

void chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    // ---------------------------------------------------------------------
    run_cpu_stages<1, 3, ProcessorType::kBigCore, 2>(task);
    // ---------------------------------------------------------------------

    out_q.enqueue(task);
  }
}

void chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    if (Task task; in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      // ---------------------------------------------------------------------
      run_cpu_stages<4, 6, ProcessorType::kLittleCore, 4>(task);
      // ---------------------------------------------------------------------

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    if (Task task; in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      // ---------------------------------------------------------------------
      run_gpu_stages<7, 7>(task);
      // ---------------------------------------------------------------------

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace TestSchedule

// ---------------------------------------------------------------------
// Test Schedule
// ---------------------------------------------------------------------

void run_test_schedule() {
  constexpr auto num_tasks = 20;
  auto tasks = init_tasks(num_tasks);
  std::vector<Task> out_tasks;
  out_tasks.reserve(num_tasks + 1);  // +1 for sentinel

  auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  TestSchedule::run_pipeline(tasks, out_tasks);
  // ---------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(num_tasks);

  std::cout << "[schedule Test]: Average time per iteration: " << avg_time << " ms" << std::endl;

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id != "3A021JEHN02756") {
    exit(0);
  }

  run_test_schedule();

  return 0;
}