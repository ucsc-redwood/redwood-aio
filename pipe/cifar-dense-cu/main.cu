#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "generated-code/device_jetson_CifarDense_all.hpp"

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "jetson") {
    run_pipelined_schedule<Task>(init_tasks_queue, device_jetson::run_pipeline_table[0], cleanup);
  }

  // if (g_device_id == "pc") {
  //   run_pipelined_schedule<Task>(init_tasks_queue, device_pc::run_pipeline_queue, cleanup);
  // } else if (g_device_id == "jetson") {
  //   run_pipelined_schedule<Task>(init_tasks_queue, device_jetson::run_pipeline_queue, cleanup);
  // }

  return 0;
}
