#pragma once

#include "../dense_appdata.hpp"

namespace cifar_dense::omp {

// Stage processing functions
void process_stage_1(cifar_dense::AppData &app_data);
void process_stage_2(cifar_dense::AppData &app_data);
void process_stage_3(cifar_dense::AppData &app_data);
void process_stage_4(cifar_dense::AppData &app_data);
void process_stage_5(cifar_dense::AppData &app_data);
void process_stage_6(cifar_dense::AppData &app_data);
void process_stage_7(cifar_dense::AppData &app_data);
void process_stage_8(cifar_dense::AppData &app_data);
void process_stage_9(cifar_dense::AppData &app_data);

}  // namespace cifar_dense::omp
