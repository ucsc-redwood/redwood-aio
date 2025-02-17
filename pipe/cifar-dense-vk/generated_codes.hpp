#pragma once

#include <vector>

#include "generated-code/3A021JEHN02756_CifarDense_schedule_001.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_002.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_003.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_004.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_005.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_006.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_007.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_008.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_009.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_010.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_011.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_012.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_013.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_014.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_015.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_016.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_017.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_018.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_019.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_020.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_021.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_022.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_023.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_024.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_025.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_026.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_027.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_028.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_029.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_030.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_031.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_032.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_033.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_034.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_035.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_036.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_037.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_038.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_039.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_040.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_041.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_042.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_043.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_044.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_045.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_046.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_047.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_048.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_049.hpp"
#include "generated-code/3A021JEHN02756_CifarDense_schedule_050.hpp"
#include "task.hpp"

namespace device_3A021JEHN02756 {

// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(std::vector<Task>&, std::vector<Task>&);

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

// Helper function to get the run_pipeline function for a given schedule ID (1-based indexing)
[[nodiscard]] inline RunPipelineFunc get_run_pipeline(int schedule_id) {
  if (schedule_id < 1 || schedule_id > 76) {
    return nullptr;
  }
  return run_pipeline_table[schedule_id - 1];
}

[[nodiscard]] inline int get_num_schedules() { return 76; }

}  // namespace device_3A021JEHN02756
