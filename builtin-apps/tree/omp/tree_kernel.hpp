#pragma once

#include "../tree_appdata.hpp"
#include "func_sort.hpp"

namespace tree {

namespace omp {

// input -> morton
void process_stage_1(tree::AppData &app_data);

// morton -> sorted morton
namespace v2 {
void process_stage_2(tree::AppData &app_data, v2::TempStorage &temp_storage);
}

// sorted morton -> unique morton
void process_stage_3(tree::AppData &app_data);

// unique morton -> brt
void process_stage_4(tree::AppData &app_data);

// brt -> edge count
void process_stage_5(tree::AppData &app_data);

// edge count -> edge offset
void process_stage_6(tree::AppData &app_data);

// *everything above* -> octree
void process_stage_7(tree::AppData &app_data);

// template <int stage>
// void run_stage(tree::AppData &app_data) {
//   if constexpr (stage == 1) {
//     process_stage_1(app_data);
//   } else if constexpr (stage == 2) {
//     process_stage_2(app_data);
//   } else if constexpr (stage == 3) {
//     process_stage_3(app_data);
//   } else if constexpr (stage == 4) {
//     process_stage_4(app_data);
//   } else if constexpr (stage == 5) {
//     process_stage_5(app_data);
//   } else if constexpr (stage == 6) {
//     process_stage_6(app_data);
//   } else if constexpr (stage == 7) {
//     process_stage_7(app_data);
//   } else {
//     throw std::runtime_error("Invalid stage");
//   }
// }

}  // namespace omp

}  // namespace tree
