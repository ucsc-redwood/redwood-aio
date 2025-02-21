#pragma once

#include "../tree_appdata.hpp"
#include "func_sort.hpp"

namespace tree::omp {

void process_stage_1(AppData &appdata, TmpStorage &temp_storage);
void process_stage_2(AppData &appdata, TmpStorage &temp_storage);
void process_stage_3(AppData &appdata, TmpStorage &temp_storage);
void process_stage_4(AppData &appdata, TmpStorage &temp_storage);
void process_stage_5(AppData &appdata, TmpStorage &temp_storage);
void process_stage_6(AppData &appdata, TmpStorage &temp_storage);
void process_stage_7(AppData &appdata, TmpStorage &temp_storage);

template <int Stage>
  requires(Stage >= 1 && Stage <= 7)
void run_stage(AppData &appdata, TmpStorage &temp_storage) {
  if constexpr (Stage == 1) {
    process_stage_1(appdata, temp_storage);
  } else if constexpr (Stage == 2) {
    process_stage_2(appdata, temp_storage);
  } else if constexpr (Stage == 3) {
    process_stage_3(appdata, temp_storage);
  } else if constexpr (Stage == 4) {
    process_stage_4(appdata, temp_storage);
  } else if constexpr (Stage == 5) {
    process_stage_5(appdata, temp_storage);
  } else if constexpr (Stage == 6) {
    process_stage_6(appdata, temp_storage);
  } else if constexpr (Stage == 7) {
    process_stage_7(appdata, temp_storage);
  } else {
    throw std::runtime_error("Invalid Stage");
  }
}

}  // namespace tree::omp
