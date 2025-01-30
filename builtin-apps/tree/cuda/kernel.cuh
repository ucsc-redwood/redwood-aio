#pragma once

#include "../tree_appdata.hpp"

namespace tree::cuda {

void process_stage_1(AppData &app_data);
void process_stage_2(AppData &app_data);
void process_stage_3(AppData &app_data);
void process_stage_4(AppData &app_data);
void process_stage_5(AppData &app_data);
void process_stage_6(AppData &app_data);
void process_stage_7(AppData &app_data);

void cleanup();

}  // namespace tree::cuda
