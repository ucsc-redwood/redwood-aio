#pragma once

#include "../tree_appdata.hpp"
#include "temp_storage.cuh"

namespace tree::cuda {

void process_stage_1(AppData &app_data, TempStorage &tmp);
void process_stage_2(AppData &app_data, TempStorage &tmp);
void process_stage_3(AppData &app_data, TempStorage &tmp);
void process_stage_4(AppData &app_data, TempStorage &tmp);
void process_stage_5(AppData &app_data, TempStorage &tmp);
void process_stage_6(AppData &app_data, TempStorage &tmp);
void process_stage_7(AppData &app_data, TempStorage &tmp);

template <int Stage>
requires(Stage >= 1 && Stage <= 7) void run_stage(AppData &appdata, TempStorage &temp_storage) {
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
  }
}

}  // namespace tree::cuda
