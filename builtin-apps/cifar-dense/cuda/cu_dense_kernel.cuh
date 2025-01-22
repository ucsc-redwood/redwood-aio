#pragma once

#include "../dense_appdata.hpp"

namespace cifar_dense::cuda {

void process_stage_1(AppData *app_data);
void process_stage_2(AppData *app_data);
void process_stage_3(AppData *app_data);
void process_stage_4(AppData *app_data);
void process_stage_5(AppData *app_data);
void process_stage_6(AppData *app_data);
void process_stage_7(AppData *app_data);
void process_stage_8(AppData *app_data);
void process_stage_9(AppData *app_data);

void device_sync();

}  // namespace cifar_dense::cuda
