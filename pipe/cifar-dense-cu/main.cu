#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "generated-code/device_jetson_CifarDense_all.hpp"

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "jetson") {
    run_warmup<Task>(
        init_tasks_queue, device_jetson::run_pipeline_table[which_schedule - 1], cleanup);

    run_pipelined_schedule<Task>(
        init_tasks_queue, device_jetson::run_pipeline_table[which_schedule - 1], cleanup);
  }

  return 0;
}
