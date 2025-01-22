#pragma once

#include "dense_appdata.hpp"

namespace cifar_dense::cuda {

void run_stage1_sync(cifar_dense::AppData *app_data);
void run_stage2_sync(cifar_dense::AppData *app_data);
void run_stage3_sync(cifar_dense::AppData *app_data);
void run_stage4_sync(cifar_dense::AppData *app_data);
void run_stage5_sync(cifar_dense::AppData *app_data);
void run_stage6_sync(cifar_dense::AppData *app_data);
void run_stage7_sync(cifar_dense::AppData *app_data);
void run_stage8_sync(cifar_dense::AppData *app_data);
void run_stage9_sync(cifar_dense::AppData *app_data);

void device_sync();

}  // namespace cifar_dense::cuda
