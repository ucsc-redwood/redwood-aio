#pragma once

#include <queue>

#include "generated-code/device_3A021JEHN02756.hpp"
#include "generated-code/device_9b034f1b.hpp"
#include "generated-code/device_ce0717178d7758b00b7e.hpp"
#include "task.hpp"

// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(std::queue<Task>&, std::queue<Task>&);

namespace device_3A021JEHN02756 {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {
    CifarDense_schedule_001::run_pipeline, CifarDense_schedule_002::run_pipeline,
    CifarDense_schedule_003::run_pipeline, CifarDense_schedule_004::run_pipeline,
    CifarDense_schedule_005::run_pipeline, CifarDense_schedule_006::run_pipeline,
    CifarDense_schedule_007::run_pipeline, CifarDense_schedule_008::run_pipeline,
    CifarDense_schedule_009::run_pipeline, CifarDense_schedule_010::run_pipeline,
    CifarDense_schedule_011::run_pipeline, CifarDense_schedule_012::run_pipeline,
    CifarDense_schedule_013::run_pipeline, CifarDense_schedule_014::run_pipeline,
    CifarDense_schedule_015::run_pipeline, CifarDense_schedule_016::run_pipeline,
    CifarDense_schedule_017::run_pipeline, CifarDense_schedule_018::run_pipeline,
    CifarDense_schedule_019::run_pipeline, CifarDense_schedule_020::run_pipeline,
    CifarDense_schedule_021::run_pipeline, CifarDense_schedule_022::run_pipeline,
    CifarDense_schedule_023::run_pipeline, CifarDense_schedule_024::run_pipeline,
    CifarDense_schedule_025::run_pipeline, CifarDense_schedule_026::run_pipeline,
    CifarDense_schedule_027::run_pipeline, CifarDense_schedule_028::run_pipeline,
    CifarDense_schedule_029::run_pipeline, CifarDense_schedule_030::run_pipeline,
    CifarDense_schedule_031::run_pipeline, CifarDense_schedule_032::run_pipeline,
    CifarDense_schedule_033::run_pipeline, CifarDense_schedule_034::run_pipeline,
    CifarDense_schedule_035::run_pipeline, CifarDense_schedule_036::run_pipeline,
    CifarDense_schedule_037::run_pipeline, CifarDense_schedule_038::run_pipeline,
    CifarDense_schedule_039::run_pipeline, CifarDense_schedule_040::run_pipeline,
    CifarDense_schedule_041::run_pipeline, CifarDense_schedule_042::run_pipeline,
    CifarDense_schedule_043::run_pipeline, CifarDense_schedule_044::run_pipeline,
    CifarDense_schedule_045::run_pipeline, CifarDense_schedule_046::run_pipeline,
    CifarDense_schedule_047::run_pipeline, CifarDense_schedule_048::run_pipeline,
    CifarDense_schedule_049::run_pipeline, CifarDense_schedule_050::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() { return 50; }

}  // namespace device_3A021JEHN02756

namespace device_9b034f1b {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {
    CifarDense_schedule_001::run_pipeline, CifarDense_schedule_002::run_pipeline,
    CifarDense_schedule_003::run_pipeline, CifarDense_schedule_004::run_pipeline,
    CifarDense_schedule_005::run_pipeline, CifarDense_schedule_006::run_pipeline,
    CifarDense_schedule_007::run_pipeline, CifarDense_schedule_008::run_pipeline,
    CifarDense_schedule_009::run_pipeline, CifarDense_schedule_010::run_pipeline,
    CifarDense_schedule_011::run_pipeline, CifarDense_schedule_012::run_pipeline,
    CifarDense_schedule_013::run_pipeline, CifarDense_schedule_014::run_pipeline,
    CifarDense_schedule_015::run_pipeline, CifarDense_schedule_016::run_pipeline,
    CifarDense_schedule_017::run_pipeline, CifarDense_schedule_018::run_pipeline,
    CifarDense_schedule_019::run_pipeline, CifarDense_schedule_020::run_pipeline,
    CifarDense_schedule_021::run_pipeline, CifarDense_schedule_022::run_pipeline,
    CifarDense_schedule_023::run_pipeline, CifarDense_schedule_024::run_pipeline,
    CifarDense_schedule_025::run_pipeline, CifarDense_schedule_026::run_pipeline,
    CifarDense_schedule_027::run_pipeline, CifarDense_schedule_028::run_pipeline,
    CifarDense_schedule_029::run_pipeline, CifarDense_schedule_030::run_pipeline,
    CifarDense_schedule_031::run_pipeline, CifarDense_schedule_032::run_pipeline,
    CifarDense_schedule_033::run_pipeline, CifarDense_schedule_034::run_pipeline,
    CifarDense_schedule_035::run_pipeline, CifarDense_schedule_036::run_pipeline,
    CifarDense_schedule_037::run_pipeline, CifarDense_schedule_038::run_pipeline,
    CifarDense_schedule_039::run_pipeline, CifarDense_schedule_040::run_pipeline,
    CifarDense_schedule_041::run_pipeline, CifarDense_schedule_042::run_pipeline,
    CifarDense_schedule_043::run_pipeline, CifarDense_schedule_044::run_pipeline,
    CifarDense_schedule_045::run_pipeline, CifarDense_schedule_046::run_pipeline,
    CifarDense_schedule_047::run_pipeline, CifarDense_schedule_048::run_pipeline,
    CifarDense_schedule_049::run_pipeline, CifarDense_schedule_050::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() { return 50; }

}  // namespace device_9b034f1b

namespace device_ce0717178d7758b00b7e {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {
    CifarDense_schedule_001::run_pipeline, CifarDense_schedule_002::run_pipeline,
    CifarDense_schedule_003::run_pipeline, CifarDense_schedule_004::run_pipeline,
    CifarDense_schedule_005::run_pipeline, CifarDense_schedule_006::run_pipeline,
    CifarDense_schedule_007::run_pipeline, CifarDense_schedule_008::run_pipeline,
    CifarDense_schedule_009::run_pipeline, CifarDense_schedule_010::run_pipeline,
    CifarDense_schedule_011::run_pipeline, CifarDense_schedule_012::run_pipeline,
    CifarDense_schedule_013::run_pipeline, CifarDense_schedule_014::run_pipeline,
    CifarDense_schedule_015::run_pipeline, CifarDense_schedule_016::run_pipeline,
    CifarDense_schedule_017::run_pipeline, CifarDense_schedule_018::run_pipeline,
    CifarDense_schedule_019::run_pipeline, CifarDense_schedule_020::run_pipeline,
    CifarDense_schedule_021::run_pipeline, CifarDense_schedule_022::run_pipeline,
    CifarDense_schedule_023::run_pipeline, CifarDense_schedule_024::run_pipeline,
    CifarDense_schedule_025::run_pipeline, CifarDense_schedule_026::run_pipeline,
    CifarDense_schedule_027::run_pipeline, CifarDense_schedule_028::run_pipeline,
    CifarDense_schedule_029::run_pipeline, CifarDense_schedule_030::run_pipeline,
    CifarDense_schedule_031::run_pipeline, CifarDense_schedule_032::run_pipeline,
    CifarDense_schedule_033::run_pipeline, CifarDense_schedule_034::run_pipeline,
    CifarDense_schedule_035::run_pipeline, CifarDense_schedule_036::run_pipeline,
    CifarDense_schedule_037::run_pipeline, CifarDense_schedule_038::run_pipeline,
    CifarDense_schedule_039::run_pipeline, CifarDense_schedule_040::run_pipeline,
    CifarDense_schedule_041::run_pipeline, CifarDense_schedule_042::run_pipeline,
    CifarDense_schedule_043::run_pipeline, CifarDense_schedule_044::run_pipeline,
    CifarDense_schedule_045::run_pipeline, CifarDense_schedule_046::run_pipeline,
    CifarDense_schedule_047::run_pipeline, CifarDense_schedule_048::run_pipeline,
    CifarDense_schedule_049::run_pipeline, CifarDense_schedule_050::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() { return 50; }

}  // namespace device_ce0717178d7758b00b7e

template <int device_index>
[[nodiscard]] constexpr RunPipelineFunc get_run_pipeline(int schedule_id) {
  if constexpr (device_index == 0) {
    return device_3A021JEHN02756::run_pipeline_table[schedule_id - 1];
  } else if constexpr (device_index == 1) {
    return device_9b034f1b::run_pipeline_table[schedule_id - 1];
  } else if constexpr (device_index == 2) {
    return device_ce0717178d7758b00b7e::run_pipeline_table[schedule_id - 1];
  }
}
