#pragma once

#include "../tree_appdata.hpp"

namespace tree {

namespace omp {

// input -> morton
void process_stage_1(tree::AppData &app_data);

// morton -> sorted morton
void process_stage_2(tree::AppData &app_data);

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

}  // namespace omp

}  // namespace tree
