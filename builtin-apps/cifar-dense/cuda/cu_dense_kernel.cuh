#pragma once

#include "../dense_appdata.hpp"

namespace cifar_dense::cuda {

void process_stage_1(AppData &app_data);
void process_stage_2(AppData &app_data);
void process_stage_3(AppData &app_data);
void process_stage_4(AppData &app_data);
void process_stage_5(AppData &app_data);
void process_stage_6(AppData &app_data);
void process_stage_7(AppData &app_data);
void process_stage_8(AppData &app_data);
void process_stage_9(AppData &app_data);

template <int stage>
void run_stage(AppData &appdata) {
  if constexpr (stage == 1) {
    process_stage_1(appdata);
  } else if constexpr (stage == 2) {
    process_stage_2(appdata);
  } else if constexpr (stage == 3) {
    process_stage_3(appdata);
  } else if constexpr (stage == 4) {
    process_stage_4(appdata);
  } else if constexpr (stage == 5) {
    process_stage_5(appdata);
  } else if constexpr (stage == 6) {
    process_stage_6(appdata);
  } else if constexpr (stage == 7) {
    process_stage_7(appdata);
  } else if constexpr (stage == 8) {
    process_stage_8(appdata);
  } else if constexpr (stage == 9) {
    process_stage_9(appdata);
  } else {
    throw std::runtime_error("Invalid stage");
  }
}

}  // namespace cifar_dense::cuda
