#include <cuda_runtime_api.h>

// #include "cu_dispatcher.cuh"

#include "cifar_dense_cu_kernel.cuh"
#include "cu_kernels.cuh"

namespace cifar_dense::cuda {

constexpr size_t div_up(const size_t a, const size_t b) {
  return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// -----------------------------------------------------------------------------

void run_stage1_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv1OutChannels * kConv1OutHeight * kConv1OutWidth;

  // SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);
  // SPDLOG_DEBUG_LAUNCH_PARAMS("conv2d");

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_image.data(),
                                              app_data->u_conv1_weights.data(),
                                              app_data->u_conv1_bias.data(),
                                              app_data->u_conv1_out.data(),
                                              kInputHeight,
                                              kInputWidth,
                                              kConv1OutChannels,
                                              kInputChannels,
                                              kKernelSize,
                                              kKernelSize,
                                              kConv1BiasSize,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kConv1OutHeight,
                                              kConv1OutWidth,
                                              kRelu);
}

// -----------------------------------------------------------------------------
// Stage 2 (maxpool)
// -----------------------------------------------------------------------------

void run_stage2_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv1OutChannels * kPool1OutHeight * kPool1OutWidth;

  // SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);
  // SPDLOG_DEBUG_LAUNCH_PARAMS("maxpool2d");

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_conv1_out.data(),
                                                 app_data->u_pool1_out.data(),
                                                 kConv1OutChannels,
                                                 kConv1OutHeight,
                                                 kConv1OutWidth,
                                                 kPoolSize,
                                                 kStride,
                                                 kPool1OutHeight,
                                                 kPool1OutWidth);
}

// -----------------------------------------------------------------------------
// Stage 3 (second conv2d)
// -----------------------------------------------------------------------------

void run_stage3_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv2OutChannels * kConv2OutHeight * kConv2OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_pool1_out.data(),
                                              app_data->u_conv2_weights.data(),
                                              app_data->u_conv2_bias.data(),
                                              app_data->u_conv2_out.data(),
                                              kPool1OutHeight,
                                              kPool1OutWidth,
                                              kConv2OutChannels,
                                              kConv1OutChannels,
                                              kKernelSize,
                                              kKernelSize,
                                              kConv2BiasSize,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kConv2OutHeight,
                                              kConv2OutWidth,
                                              kRelu);
}

// -----------------------------------------------------------------------------
// Stage 4 (second maxpool2d)
// -----------------------------------------------------------------------------

void run_stage4_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv2OutChannels * kPool2OutHeight * kPool2OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_conv2_out.data(),
                                                 app_data->u_pool2_out.data(),
                                                 kConv2OutChannels,
                                                 kConv2OutHeight,
                                                 kConv2OutWidth,
                                                 kPoolSize,
                                                 kStride,
                                                 kPool2OutHeight,
                                                 kPool2OutWidth);
}

// -----------------------------------------------------------------------------
// Stage 5 (third conv2d)
// -----------------------------------------------------------------------------

void run_stage5_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv3OutChannels * kConv3OutHeight * kConv3OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_pool2_out.data(),
                                              app_data->u_conv3_weights.data(),
                                              app_data->u_conv3_bias.data(),
                                              app_data->u_conv3_out.data(),
                                              kPool2OutHeight,
                                              kPool2OutWidth,
                                              kConv3OutChannels,
                                              kConv2OutChannels,
                                              kKernelSize,
                                              kKernelSize,
                                              kConv3BiasSize,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kConv3OutHeight,
                                              kConv3OutWidth,
                                              kRelu);
}

// -----------------------------------------------------------------------------
// Stage 6 (fourth conv2d)
// -----------------------------------------------------------------------------

void run_stage6_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv4OutChannels * kConv4OutHeight * kConv4OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_conv3_out.data(),
                                              app_data->u_conv4_weights.data(),
                                              app_data->u_conv4_bias.data(),
                                              app_data->u_conv4_out.data(),
                                              kConv3OutHeight,
                                              kConv3OutWidth,
                                              kConv4OutChannels,
                                              kConv3OutChannels,
                                              kKernelSize,
                                              kKernelSize,
                                              kConv4BiasSize,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kConv4OutHeight,
                                              kConv4OutWidth,
                                              kRelu);
}

// -----------------------------------------------------------------------------
// Stage 7 (fifth conv2d)
// -----------------------------------------------------------------------------

void run_stage7_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv5OutChannels * kConv5OutHeight * kConv5OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_conv4_out.data(),
                                              app_data->u_conv5_weights.data(),
                                              app_data->u_conv5_bias.data(),
                                              app_data->u_conv5_out.data(),
                                              kConv4OutHeight,
                                              kConv4OutWidth,
                                              kConv5OutChannels,
                                              kConv4OutChannels,
                                              kKernelSize,
                                              kKernelSize,
                                              kConv5BiasSize,
                                              kKernelSize,
                                              kStride,
                                              kPadding,
                                              kConv5OutHeight,
                                              kConv5OutWidth,
                                              kRelu);
}

// -----------------------------------------------------------------------------
// Stage 8 (third maxpool2d)
// -----------------------------------------------------------------------------

void run_stage8_sync(AppData *app_data) {
  constexpr auto total_iterations =
      kConv5OutChannels * kPool3OutHeight * kPool3OutWidth;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data->u_conv5_out.data(),
                                                 app_data->u_pool3_out.data(),
                                                 kConv5OutChannels,
                                                 kConv5OutHeight,
                                                 kConv5OutWidth,
                                                 kPoolSize,
                                                 kStride,
                                                 kPool3OutHeight,
                                                 kPool3OutWidth);
}

// -----------------------------------------------------------------------------
// Stage 9 (linear)
// -----------------------------------------------------------------------------

void run_stage9_sync(AppData *app_data) {
  constexpr auto total_iterations = kLinearOutFeatures;

  constexpr auto grid_dim = div_up(total_iterations, 256);
  constexpr auto block_dim = dim3{256, 1, 1};
  constexpr auto shared_mem = 0;

  linear<<<grid_dim, block_dim, shared_mem>>>(app_data->u_pool3_out.data(),
                                              app_data->u_linear_weights.data(),
                                              app_data->u_linear_bias.data(),
                                              app_data->u_linear_out.data(),
                                              kLinearInFeatures,
                                              kLinearOutFeatures);
}

}  // namespace cifar_dense::cuda
