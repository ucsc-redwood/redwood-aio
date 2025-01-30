#pragma once

#include <cuda_runtime_api.h>

#include "../sparse_appdata.hpp"

namespace cifar_sparse::cuda {

void process_stage_1(AppData &appdata);
void process_stage_2(AppData &appdata);
void process_stage_3(AppData &appdata);
void process_stage_4(AppData &appdata);
void process_stage_5(AppData &appdata);
void process_stage_6(AppData &appdata);
void process_stage_7(AppData &appdata);
void process_stage_8(AppData &appdata);
void process_stage_9(AppData &appdata);

void device_sync();

}  // namespace cifar_sparse::cuda
