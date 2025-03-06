#pragma once

#include <queue>

#include "generated-code/device_jetson_CifarDense_all.hpp"
#include "task.hpp"

// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(std::queue<Task>&, std::queue<Task>&);

namespace device_jetson {

// Array of function pointers to all run_pipeline implementations
// Index 0 corresponds to schedule_001, etc.
static const RunPipelineFunc run_pipeline_table[] = {
    schedule_jetson_CifarDense_schedule_001::run_pipeline,
    schedule_jetson_CifarDense_schedule_002::run_pipeline,
    schedule_jetson_CifarDense_schedule_003::run_pipeline,
    schedule_jetson_CifarDense_schedule_004::run_pipeline,
    schedule_jetson_CifarDense_schedule_005::run_pipeline,
    schedule_jetson_CifarDense_schedule_006::run_pipeline,
    schedule_jetson_CifarDense_schedule_007::run_pipeline,
    schedule_jetson_CifarDense_schedule_008::run_pipeline,
    schedule_jetson_CifarDense_schedule_009::run_pipeline,
    schedule_jetson_CifarDense_schedule_010::run_pipeline,
    schedule_jetson_CifarDense_schedule_011::run_pipeline,
    schedule_jetson_CifarDense_schedule_012::run_pipeline,
    schedule_jetson_CifarDense_schedule_013::run_pipeline,
    schedule_jetson_CifarDense_schedule_014::run_pipeline,
    schedule_jetson_CifarDense_schedule_015::run_pipeline,
    schedule_jetson_CifarDense_schedule_016::run_pipeline,
    schedule_jetson_CifarDense_schedule_017::run_pipeline,
    schedule_jetson_CifarDense_schedule_018::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() { return 50; }

}  // namespace device_jetson
