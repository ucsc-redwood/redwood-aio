#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "generated-code/all_schedules.hpp"

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

__global__ void kernel_test() {}

void warmup() {
  kernel_test<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "jetson") {
    warmup();

    device_jetson::get_run_pipeline_func(which_schedule)(20);

    // run_warmup<Task>(
    //     init_tasks_queue, device_jetson::get_run_pipeline_func(which_schedule), cleanup);

    // run_pipelined_schedule<Task>(
    //     init_tasks_queue, device_jetson::get_run_pipeline_func(which_schedule), cleanup);
  }

  return 0;
}
