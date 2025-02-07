#pragma once

#include "../tree_appdata.hpp"

namespace tree::cuda {

// void warmup(AppData &app_data);
// void cleanup();

struct TempStorage {
  explicit TempStorage();
  ~TempStorage();

  struct {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
  } sort;

  struct {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
  } unique;
  uint32_t *g_num_selected_out = nullptr;

  struct {
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
  } prefix_sum;
};

void process_stage_1(AppData &app_data);
void process_stage_2(AppData &app_data, TempStorage &tmp);
void process_stage_3(AppData &app_data, TempStorage &tmp);
void process_stage_4(AppData &app_data);
void process_stage_5(AppData &app_data);
void process_stage_6(AppData &app_data, TempStorage &tmp);
void process_stage_7(AppData &app_data);

// void cleanup();

}  // namespace tree::cuda
