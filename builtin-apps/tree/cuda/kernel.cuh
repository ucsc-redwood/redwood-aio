#pragma once

#include "../tree_appdata.hpp"

namespace tree::cuda {

/**
 * @brief Before calling any process_stage functions, you must create an
 * instance of TempStorage. The TempStorage object manages temporary device
 * memory needed by various stages. It automatically allocates memory in its
 * constructor and frees it in its destructor.
 *
 * Example usage:
 * @code
 * tree::cuda::TempStorage tmp;
 * process_stage_1(app_data);
 * process_stage_2(app_data, tmp);
 * process_stage_3(app_data, tmp);
 * // etc...
 * @endcode
 */
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

  // Unified
  uint32_t *u_num_selected_out = nullptr;

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
