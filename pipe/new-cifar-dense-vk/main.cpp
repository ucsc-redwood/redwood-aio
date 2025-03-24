#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int which_schedule = 1;
  app.add_option("-s,--schedule", which_schedule, "Schedule ID")->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  if (g_device_id == "3A021JEHN02756") {
    device_3A021JEHN02756::schedule_3A021JEHN02756_CifarDense_schedule_001::run_pipeline_warmup();

    device_3A021JEHN02756::schedule_3A021JEHN02756_CifarDense_schedule_001::run_pipeline(20);
  }

  return 0;
}
