#include <omp.h>

#include <cassert>
#include <cstddef>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"

// ------------------------------------------------------------------------------------------------
#include "spsc_queue.hpp"
#include "task.hpp"

// constexpr size_t kNumTasks = 100;

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

template <ProcessorType PT>
static void worker_thread(const size_t num_threads,
                          SPSCQueue<Task, 1024>& in_queue,
                          SPSCQueue<Task, 1024>* out_queue,
                          std::function<void(Task&)> process_function) {
  while (true) {
    Task task;
    if (in_queue.dequeue(task)) {
      if (task.is_sentinel) {
        if (out_queue) {
          out_queue->enqueue(std::move(task));
        }
        break;
      }

#pragma omp parallel num_threads(num_threads)
      {
        if constexpr (PT == ProcessorType::kLittleCore) {
          bind_thread_to_cores(g_little_cores);
        } else if constexpr (PT == ProcessorType::kMediumCore) {
          bind_thread_to_cores(g_medium_cores);
        } else if constexpr (PT == ProcessorType::kBigCore) {
          bind_thread_to_cores(g_big_cores);
        }

        // Process the task
#pragma omp critical
        {
          auto num_threads = omp_get_num_threads();

          if constexpr (PT == ProcessorType::kLittleCore) {
            std::cout << "Little core processed task " << task.uid << " [" << num_threads
                      << "] with core " << sched_getcpu() << std::endl;
          } else if constexpr (PT == ProcessorType::kMediumCore) {
            std::cout << "Medium core processed task " << task.uid << " [" << num_threads
                      << "] with core " << sched_getcpu() << std::endl;
          } else if constexpr (PT == ProcessorType::kBigCore) {
            std::cout << "Big core processed task " << task.uid << " [" << num_threads
                      << "] with core " << sched_getcpu() << std::endl;
          }
        }

        process_function(task);
      }

      // Forward processed task
      if (out_queue) {
        out_queue->enqueue(std::move(task));
      }

    } else {
      std::this_thread::yield();
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  SPSCQueue<Task> q_0_1;
  SPSCQueue<Task> q_1_2;
  SPSCQueue<Task> q_2_3;

  // Master thread pushing tasks
  for (size_t i = 0; i < 100; ++i) {
    q_0_1.enqueue(new_task(1024));
  }
  q_0_1.enqueue(new_sentinel());

  // ------------------------------------------------------------------------------------------------

  {
    std::thread t1(worker_thread<ProcessorType::kLittleCore>,
                   g_little_cores.size(),
                   std::ref(q_0_1),
                   &q_1_2,
                   process_task_stage_A);
    std::thread t2(worker_thread<ProcessorType::kMediumCore>,
                   g_medium_cores.size(),
                   std::ref(q_1_2),
                   &q_2_3,
                   process_task_stage_B);
    std::thread t3(worker_thread<ProcessorType::kBigCore>,
                   g_big_cores.size(),
                   std::ref(q_2_3),
                   nullptr,
                   process_task_stage_C);

    t1.join();
    t2.join();
    t3.join();
  }

  // ------------------------------------------------------------------------------------------------

  spdlog::info("Done");
  return 0;
}