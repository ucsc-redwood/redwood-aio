#include <spdlog/spdlog.h>

#include "cu_dispatcher.cuh"
#include "cu_kernels.cuh"
// #include "redwood/cuda/helpers.cuh"
#include "../../common/cuda/helpers.cuh"

namespace cifar_sparse::cuda {

// -----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// -----------------------------------------------------------------------------

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

void process_stage_1(AppData *appdata) {
  auto app_data = *appdata;
  // void run_stage1(AppData &app_data, const cudaStream_t stream, bool sync) {
  // static const auto total_iterations =
  //     model::kConv1OutChannels * model::kConv1OutHeight *
  //     model::kConv1OutWidth;

  // SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);
  // SPDLOG_DEBUG_LAUNCH_PARAMS("conv2d");

  // kernels::dense::conv2d<<<grid_dim, block_dim, shared_mem>>>(

  // );

  // total_iterations = app_data.conv1_weights.rows;

  // constexpr auto input_height = kInputHeight;
  // constexpr auto input_width = kInputWidth;
  // constexpr auto padding = kPadding;
  // constexpr auto kernel_size = kKernelSize;
  // constexpr auto stride = kStride;

  // int output_height = (kInputHeight + 2 * kPadding - kKernelSize) / kStride +
  // 1; int output_width = (kInputWidth + 2 * kPadding - kKernelSize) / kStride
  // + 1;

  // // Each block processes an 8x8x8 region
  // constexpr dim3 block_dim(8, 8, 8);
  // const dim3 grid_dim(app_data.conv1_weights.rows,
  //                     (output_height + 7) / 8,
  //                     (output_width + 7) / 8);
  // const auto shared_mem = 0;

  const auto total_iterations = app_data.conv1_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);
  // SPDLOG_DEBUG_LAUNCH_PARAMS("conv2d");

  // __global__ void conv2d(const float* input_data,
  //                        const int image_input_channels,
  //                        const int input_height,
  //                        const int input_width,
  //                        const CSRMatrix& weight_matrix,
  //                        const float* bias_data,
  //                        const int bias_size,
  //                        const int kernel_size,
  //                        const int stride,
  //                        const int padding,
  //                        const bool relu,
  //                        float* output_data

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_image_data.data(),
                                              kInputChannels,
                                              kInputHeight,
                                              kInputWidth,
                                              // app_data.conv1_weights,
                                              app_data.conv1_weights.values,
                                              app_data.conv1_weights.row_ptr,
                                              app_data.conv1_weights.col_idx,
                                              app_data.conv1_weights.rows,
                                              app_data.conv1_weights.cols,
                                              app_data.conv1_weights.nnz,
                                              app_data.u_conv1_bias.data(),
                                              64,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              app_data.u_conv1_output.data());
}

void process_stage_2(AppData *appdata) {
  auto app_data = *appdata;

  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = kInputChannels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(
      app_data.u_conv1_output.data(),
      kInputChannels,
      kInputHeight,
      kInputWidth,
      kPoolSize,
      kPoolStride,
      app_data.u_pool1_output.data());
}

void process_stage_3(AppData *appdata) {
  auto app_data = *appdata;

  const auto total_iterations = app_data.conv2_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool1_output.data(),
                                              64,
                                              16,
                                              16,
                                              app_data.conv2_weights.values,
                                              app_data.conv2_weights.row_ptr,
                                              app_data.conv2_weights.col_idx,
                                              app_data.conv2_weights.rows,
                                              app_data.conv2_weights.cols,
                                              app_data.conv2_weights.nnz,
                                              app_data.u_conv2_bias.data(),
                                              192,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              app_data.u_conv2_output.data());
}

void process_stage_4(AppData *appdata) {
  auto app_data = *appdata;

  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(
      app_data.u_conv2_output.data(),
      input_channels,
      input_height,
      input_width,
      kPoolSize,
      kPoolStride,
      app_data.u_pool2_output.data());
}

void process_stage_5(AppData *appdata) {
  auto app_data = *appdata;

  const auto total_iterations = app_data.conv3_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool2_output.data(),
                                              192,
                                              8,
                                              8,
                                              app_data.conv3_weights.values,
                                              app_data.conv3_weights.row_ptr,
                                              app_data.conv3_weights.col_idx,
                                              app_data.conv3_weights.rows,
                                              app_data.conv3_weights.cols,
                                              app_data.conv3_weights.nnz,
                                              app_data.u_conv3_bias.data(),
                                              384,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              app_data.u_conv3_output.data());
}

void process_stage_6(AppData *appdata) {
  auto app_data = *appdata;

  const auto total_iterations = app_data.conv4_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv3_output.data(),
                                              384,
                                              8,
                                              8,
                                              app_data.conv4_weights.values,
                                              app_data.conv4_weights.row_ptr,
                                              app_data.conv4_weights.col_idx,
                                              app_data.conv4_weights.rows,
                                              app_data.conv4_weights.cols,
                                              app_data.conv4_weights.nnz,
                                              app_data.u_conv4_bias.data(),
                                              256,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              app_data.u_conv4_output.data());
}

void process_stage_7(AppData *appdata) {
  auto app_data = *appdata;

  const auto total_iterations = app_data.conv5_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv4_output.data(),
                                              256,
                                              8,
                                              8,
                                              app_data.conv5_weights.values,
                                              app_data.conv5_weights.row_ptr,
                                              app_data.conv5_weights.col_idx,
                                              app_data.conv5_weights.rows,
                                              app_data.conv5_weights.cols,
                                              app_data.conv5_weights.nnz,
                                              app_data.u_conv5_bias.data(),
                                              256,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              app_data.u_conv5_output.data());
}

void process_stage_8(AppData *appdata) {
  auto app_data = *appdata;

  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(
      app_data.u_conv5_output.data(),
      input_channels,
      input_height,
      input_width,
      kPoolSize,
      kPoolStride,
      app_data.u_pool3_output.data());
}

void process_stage_9(AppData *appdata) {
  auto app_data = *appdata;

  const auto total_iterations = app_data.linear_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  linear<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool3_output.data(),
                                              app_data.linear_weights.values,
                                              app_data.linear_weights.row_ptr,
                                              app_data.linear_weights.col_idx,
                                              app_data.linear_weights.rows,
                                              app_data.linear_weights.cols,
                                              app_data.linear_weights.nnz,
                                              app_data.u_linear_bias.data(),
                                              app_data.u_linear_output.data());
}

}  // namespace cifar_sparse::cuda
