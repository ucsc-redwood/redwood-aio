#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <queue>

#include "../templates.hpp"
#include "builtin-apps/app.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// Define a Schedule (Nvidia PC)
// ---------------------------------------------------------------------

namespace device_pc {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    // ---------------------------------------------------------------------
    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 6>(task);
    // ---------------------------------------------------------------------

    out_q.enqueue(task);
    in_tasks.pop();
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      // ---------------------------------------------------------------------
      run_gpu_stages<4, 7>(task);
      // ---------------------------------------------------------------------

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk4.join();
}

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  if (g_device_id == "pc") {
    run_pipelined_schedule<Task>(init_tasks, device_pc::run_pipeline, cleanup);
  }

  return 0;
}
