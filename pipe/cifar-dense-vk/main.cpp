#include <spdlog/spdlog.h>

#include <iostream>

#include "builtin-apps/app.hpp"
#include "generated_codes.hpp"
#include "task.hpp"

template <int device_index>
void run_warmup(const int schedule_id) {
  // disable logging for warmup
  spdlog::set_level(spdlog::level::off);

  auto tasks = init_tasks(20);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  // -------------------  run the pipeline  ------------------------------
  get_run_pipeline<device_index>(schedule_id)(tasks, out_tasks);
  // ---------------------------------------------------------------------

  cleanup(tasks);

  // restore original log level
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));
}

// ---------------------------------------------------------------------
// Pipeline Instance (Best)
// ---------------------------------------------------------------------

template <int device_index>
void run_one_schedule(const int schedule_id) {
  auto tasks = init_tasks(20);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  // -------------------  run the pipeline  ------------------------------
  get_run_pipeline<device_index>(schedule_id)(tasks, out_tasks);
  // ---------------------------------------------------------------------

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "[schedule " << schedule_id << "]: Average time per iteration: " << avg_time << " ms"
            << std::endl;

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

  if (g_device_id == "3A021JEHN02756") {
    run_warmup<0>(which_schedule);  // mostly just to compile the GPU shader
    run_one_schedule<0>(which_schedule);
  } else if (g_device_id == "9b034f1b") {
    run_one_schedule<1>(which_schedule);
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    run_one_schedule<2>(which_schedule);
  }

  return 0;
}