#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// ---------------------------------------------------------------------
// Define a Schedule (Nvidia PC)
// ---------------------------------------------------------------------

namespace device_pc {

void run_pipeline_queue(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1(
      [&]() { chunk_first(tasks, q_01, run_cpu_stages<1, 3, ProcessorType::kBigCore, 8>); });
  std::thread t_chunk4([&]() { chunk_last(q_01, out_tasks, run_gpu_stages<4, 7>); });

  t_chunk1.join();
  t_chunk4.join();
}

}  // namespace device_pc

// ---------------------------------------------------------------------
// Define a Schedule (Jetson Orin Nano)
// ---------------------------------------------------------------------

namespace device_jetson {

void run_pipeline_queue(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1(
      [&]() { chunk_first(tasks, q_01, run_cpu_stages<1, 3, ProcessorType::kLittleCore, 6>); });
  std::thread t_chunk4([&]() { chunk_last(q_01, out_tasks, run_gpu_stages<4, 7>); });

  t_chunk1.join();
  t_chunk4.join();
}

}  // namespace device_jetson

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "pc") {
    run_pipelined_schedule<Task>(init_tasks_queue, device_pc::run_pipeline_queue, cleanup);
  } else if (g_device_id == "jetson") {
    run_pipelined_schedule<Task>(init_tasks_queue, device_jetson::run_pipeline_queue, cleanup);
  }

  return 0;
}
