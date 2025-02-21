#pragma once

#include <vector>

#include "generated-code/device_3A021JEHN02756.hpp"
#include "generated-code/device_ce0717178d7758b00b7e.hpp"
#include "task.hpp"

// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(std::vector<Task>&, std::vector<Task>&);

namespace device_3A021JEHN02756 {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {CifarSparse_schedule_001::run_pipeline,
                                                     CifarSparse_schedule_002::run_pipeline,
                                                     CifarSparse_schedule_003::run_pipeline,
                                                     CifarSparse_schedule_004::run_pipeline,
                                                     CifarSparse_schedule_005::run_pipeline,
                                                     CifarSparse_schedule_006::run_pipeline,
                                                     CifarSparse_schedule_007::run_pipeline,
                                                     CifarSparse_schedule_008::run_pipeline,
                                                     CifarSparse_schedule_009::run_pipeline,
                                                     CifarSparse_schedule_010::run_pipeline,
                                                     CifarSparse_schedule_011::run_pipeline,
                                                     CifarSparse_schedule_012::run_pipeline,
                                                     CifarSparse_schedule_013::run_pipeline,
                                                     CifarSparse_schedule_014::run_pipeline,
                                                     CifarSparse_schedule_015::run_pipeline};

[[nodiscard]] constexpr int get_num_schedules() { return 15; }

}  // namespace device_3A021JEHN02756

namespace device_ce0717178d7758b00b7e {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {
    CifarSparse_schedule_001::run_pipeline, CifarSparse_schedule_002::run_pipeline,
    CifarSparse_schedule_003::run_pipeline, CifarSparse_schedule_004::run_pipeline,
    CifarSparse_schedule_005::run_pipeline, CifarSparse_schedule_006::run_pipeline,
    CifarSparse_schedule_007::run_pipeline, CifarSparse_schedule_008::run_pipeline,
    CifarSparse_schedule_009::run_pipeline, CifarSparse_schedule_010::run_pipeline,
    CifarSparse_schedule_011::run_pipeline, CifarSparse_schedule_012::run_pipeline,
    CifarSparse_schedule_013::run_pipeline, CifarSparse_schedule_014::run_pipeline,
    CifarSparse_schedule_015::run_pipeline, CifarSparse_schedule_016::run_pipeline,
    CifarSparse_schedule_017::run_pipeline, CifarSparse_schedule_018::run_pipeline,
    CifarSparse_schedule_019::run_pipeline, CifarSparse_schedule_020::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() { return 20; }

}  // namespace device_ce0717178d7758b00b7e

template <int device_index>
[[nodiscard]] constexpr RunPipelineFunc get_run_pipeline(int schedule_id) {
  if constexpr (device_index == 0) {
    return device_3A021JEHN02756::run_pipeline_table[schedule_id - 1];
  } else if constexpr (device_index == 1) {
    return nullptr;
  } else if constexpr (device_index == 2) {
    return device_ce0717178d7758b00b7e::run_pipeline_table[schedule_id - 1];
  }
}
