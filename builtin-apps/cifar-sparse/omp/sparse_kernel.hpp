#pragma once

#include "../sparse_appdata.hpp"

// --------------------------------------------------------------------------------
// Kernel
// --------------------------------------------------------------------------------

namespace cifar_sparse {
namespace omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Sparse)
// ----------------------------------------------------------------------------

// Input Image dimensions
constexpr int kInputChannels = 3;
constexpr int kInputHeight = 32;
constexpr int kInputWidth = 32;

// Convolution parameters
constexpr int kKernelSize = 3;
constexpr int kStride = 1;
constexpr int kPadding = 1;

// Pooling parameters
constexpr int kPoolSize = 2;
constexpr int kPoolStride = 2;

constexpr bool kRelu = true;

// Function declarations
void conv2d_omp(const float *input_data,
                int image_input_channels,
                int input_height,
                int input_width,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                int bias_size,
                int kernel_size,
                int stride,
                int padding,
                bool relu,
                float *output_data,
                int start,
                int end);

void maxpool2d_omp(const float *input_data,
                   int input_channels,
                   int input_height,
                   int input_width,
                   int pool_size,
                   int stride,
                   float *output_data,
                   int start,
                   int end);

void linear_omp(const float *input_data,
                const CSRMatrix &weight_matrix,
                const float *bias_data,
                float *output_data,
                int start,
                int end);

// Pipeline processing functions
void process_stage_1(cifar_sparse::AppData &app_data);
void process_stage_2(cifar_sparse::AppData &app_data);
void process_stage_3(cifar_sparse::AppData &app_data);
void process_stage_4(cifar_sparse::AppData &app_data);
void process_stage_5(cifar_sparse::AppData &app_data);
void process_stage_6(cifar_sparse::AppData &app_data);
void process_stage_7(cifar_sparse::AppData &app_data);
void process_stage_8(cifar_sparse::AppData &app_data);
void process_stage_9(cifar_sparse::AppData &app_data);

template <int stage>
void run_stage(cifar_sparse::AppData &app_data) {
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
}  // namespace cifar_sparse
