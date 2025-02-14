#pragma once

#include <stdexcept>

#include "../dense_appdata.hpp"

namespace cifar_dense {

namespace omp {

// Function declarations
void conv2d_omp(const float *input_data,
                const int image_input_channels,
                const int input_height,
                const int input_width,
                const float *weight_data,
                const int weight_output_channels,
                const int weight_input_channels,
                const int weight_height,
                const int weight_width,
                const float *bias_data,
                const int bias_number_of_elements,
                const int kernel_size,
                const int stride,
                const int padding,
                const bool relu,
                float *output_data,
                const int start,
                const int end);

void maxpool2d_omp(const float *input_data,
                   const int input_channels,
                   const int input_height,
                   const int input_width,
                   const int pool_size,
                   const int stride,
                   float *output_data,
                   const int start,
                   const int end);

void linear_omp(const float *input,
                const float *weights,
                const float *bias,
                float *output,
                const uint32_t input_size,
                const uint32_t output_size,
                const int start,
                const int end);

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

template <int stage>
void run_stage(cifar_dense::AppData &app_data) {
  if constexpr (stage == 1) {
    process_stage_1(app_data);
  } else if constexpr (stage == 2) {
    process_stage_2(app_data);
  } else if constexpr (stage == 3) {
    process_stage_3(app_data);
  } else if constexpr (stage == 4) {
    process_stage_4(app_data);
  } else if constexpr (stage == 5) {
    process_stage_5(app_data);
  } else if constexpr (stage == 6) {
    process_stage_6(app_data);
  } else if constexpr (stage == 7) {
    process_stage_7(app_data);
  } else if constexpr (stage == 8) {
    process_stage_8(app_data);
  } else if constexpr (stage == 9) {
    process_stage_9(app_data);
  } else {
    throw std::runtime_error("Invalid stage");
  }
}

}  // namespace omp

}  // namespace cifar_dense
