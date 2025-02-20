#pragma once

#include "../tree_appdata.hpp"
#include "func_sort.hpp"

namespace tree::omp {

// input -> morton
void process_stage_1(AppData &appdata, TmpStorage &temp_storage);

// morton -> sorted morton
void process_stage_2(AppData &appdata, TmpStorage &temp_storage);

// sorted morton -> unique morton
void process_stage_3(AppData &appdata, TmpStorage &temp_storage);

// unique morton -> brt
void process_stage_4(AppData &appdata, TmpStorage &temp_storage);

// brt -> edge count
void process_stage_5(AppData &appdata, TmpStorage &temp_storage);

// edge count -> edge offset
void process_stage_6(AppData &appdata, TmpStorage &temp_storage);

// *everything above* -> octree
void process_stage_7(AppData &appdata, TmpStorage &temp_storage);

template <int stage>
void run_stage(AppData &appdata, TmpStorage &temp_storage) {
  if constexpr (stage == 1) {
    process_stage_1(appdata, temp_storage);
  } else if constexpr (stage == 2) {
    process_stage_2(appdata, temp_storage);
  } else if constexpr (stage == 3) {
    process_stage_3(appdata, temp_storage);
  } else if constexpr (stage == 4) {
    process_stage_4(appdata, temp_storage);
  } else if constexpr (stage == 5) {
    process_stage_5(appdata, temp_storage);
  } else if constexpr (stage == 6) {
    process_stage_6(appdata, temp_storage);
  } else if constexpr (stage == 7) {
    process_stage_7(appdata, temp_storage);
  } else {
    throw std::runtime_error("Invalid stage");
  }
}

}  // namespace tree::omp
