#include "../../common/cuda/helpers.cuh"
#include "cu_dispatcher.cuh"
#include "cu_kernels.cuh"

namespace cifar_sparse {
namespace cuda {

void device_sync() { CUDA_CHECK(cudaDeviceSynchronize()); }

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

void process_stage_1(AppData &appdata) {
  // static const auto total_iterations =
  //     model::kConv1OutChannels * model::kConv1OutHeight *
  //     model::kConv1OutWidth;

  // SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);
  // SPDLOG_DEBUG_LAUNCH_PARAMS("conv2d");

  // kernels::dense::conv2d<<<grid_dim, block_dim, shared_mem>>>(

  // );

  // total_iterations = appdata.conv1_weights.rows;

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
  // const dim3 grid_dim(appdata.conv1_weights.rows,
  //                     (output_height + 7) / 8,
  //                     (output_width + 7) / 8);
  // const auto shared_mem = 0;

  auto total_iterations = appdata.conv1_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

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

  conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_image_data.data(),
                                              kInputChannels,
                                              kInputHeight,
                                              kInputWidth,
                                              // appdata.conv1_weights,
                                              appdata.conv1_weights.values,
                                              appdata.conv1_weights.row_ptr,
                                              appdata.conv1_weights.col_idx,
                                              appdata.conv1_weights.rows,
                                              appdata.conv1_weights.cols,
                                              appdata.conv1_weights.nnz,
                                              appdata.u_conv1_bias.data(),
                                              64,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              appdata.u_conv1_output.data());
}

void process_stage_2(AppData &appdata) {
  constexpr auto output_height = (kInputHeight - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (kInputWidth - kPoolSize) / kPoolStride + 1;
  auto total_iterations = kInputChannels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv1_output.data(),
                                                 kInputChannels,
                                                 kInputHeight,
                                                 kInputWidth,
                                                 kPoolSize,
                                                 kPoolStride,
                                                 appdata.u_pool1_output.data());
}

void process_stage_3(AppData &appdata) {
  const auto total_iterations = appdata.conv2_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool1_output.data(),
                                              64,
                                              16,
                                              16,
                                              appdata.conv2_weights.values,
                                              appdata.conv2_weights.row_ptr,
                                              appdata.conv2_weights.col_idx,
                                              appdata.conv2_weights.rows,
                                              appdata.conv2_weights.cols,
                                              appdata.conv2_weights.nnz,
                                              appdata.u_conv2_bias.data(),
                                              192,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              appdata.u_conv2_output.data());
}

void process_stage_4(AppData &appdata) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv2_output.data(),
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 kPoolSize,
                                                 kPoolStride,
                                                 appdata.u_pool2_output.data());
}

void process_stage_5(AppData &appdata) {
  const auto total_iterations = appdata.conv3_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool2_output.data(),
                                              192,
                                              8,
                                              8,
                                              appdata.conv3_weights.values,
                                              appdata.conv3_weights.row_ptr,
                                              appdata.conv3_weights.col_idx,
                                              appdata.conv3_weights.rows,
                                              appdata.conv3_weights.cols,
                                              appdata.conv3_weights.nnz,
                                              appdata.u_conv3_bias.data(),
                                              384,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              appdata.u_conv3_output.data());
}

void process_stage_6(AppData &appdata) {
  const auto total_iterations = appdata.conv4_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv3_output.data(),
                                              384,
                                              8,
                                              8,
                                              appdata.conv4_weights.values,
                                              appdata.conv4_weights.row_ptr,
                                              appdata.conv4_weights.col_idx,
                                              appdata.conv4_weights.rows,
                                              appdata.conv4_weights.cols,
                                              appdata.conv4_weights.nnz,
                                              appdata.u_conv4_bias.data(),
                                              256,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              appdata.u_conv4_output.data());
}

void process_stage_7(AppData &appdata) {
  const auto total_iterations = appdata.conv5_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv4_output.data(),
                                              256,
                                              8,
                                              8,
                                              appdata.conv5_weights.values,
                                              appdata.conv5_weights.row_ptr,
                                              appdata.conv5_weights.col_idx,
                                              appdata.conv5_weights.rows,
                                              appdata.conv5_weights.cols,
                                              appdata.conv5_weights.nnz,
                                              appdata.u_conv5_bias.data(),
                                              256,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kRelu,
                                              appdata.u_conv5_output.data());
}

void process_stage_8(AppData &appdata) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(appdata.u_conv5_output.data(),
                                                 input_channels,
                                                 input_height,
                                                 input_width,
                                                 kPoolSize,
                                                 kPoolStride,
                                                 appdata.u_pool3_output.data());
}

void process_stage_9(AppData &appdata) {
  const auto total_iterations = appdata.linear_weights.rows;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  linear<<<grid_dim, block_dim, shared_mem>>>(appdata.u_pool3_output.data(),
                                              appdata.linear_weights.values,
                                              appdata.linear_weights.row_ptr,
                                              appdata.linear_weights.col_idx,
                                              appdata.linear_weights.rows,
                                              appdata.linear_weights.cols,
                                              appdata.linear_weights.nnz,
                                              appdata.u_linear_bias.data(),
                                              appdata.u_linear_output.data());
}

}  // namespace cuda
}  // namespace cifar_sparse