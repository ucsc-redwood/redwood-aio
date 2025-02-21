#include <cuda_runtime.h>

#include "../../common/cuda/helpers.cuh"
#include "dispatchers.cuh"
#include "kernels/cu_kernels.cuh"

namespace cifar_dense::cuda {

// -----------------------------------------------------------------------------
// Stage 1 (first conv2d)
// -----------------------------------------------------------------------------

void process_stage_1(AppData &app_data) {
  constexpr auto total_iterations = kConv1OutChannels * kConv1OutHeight * kConv1OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_image.data(),
                                              app_data.u_conv1_weights.data(),
                                              app_data.u_conv1_bias.data(),
                                              app_data.u_conv1_out.data(),
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

void process_stage_2(AppData &app_data) {
  constexpr auto total_iterations = kConv1OutChannels * kPool1OutHeight * kPool1OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv1_out.data(),
                                                 app_data.u_pool1_out.data(),
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

void process_stage_3(AppData &app_data) {
  constexpr auto total_iterations = kConv2OutChannels * kConv2OutHeight * kConv2OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool1_out.data(),
                                              app_data.u_conv2_weights.data(),
                                              app_data.u_conv2_bias.data(),
                                              app_data.u_conv2_out.data(),
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

void process_stage_4(AppData &app_data) {
  constexpr auto total_iterations = kConv2OutChannels * kPool2OutHeight * kPool2OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv2_out.data(),
                                                 app_data.u_pool2_out.data(),
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

void process_stage_5(AppData &app_data) {
  constexpr auto total_iterations = kConv3OutChannels * kConv3OutHeight * kConv3OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool2_out.data(),
                                              app_data.u_conv3_weights.data(),
                                              app_data.u_conv3_bias.data(),
                                              app_data.u_conv3_out.data(),
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

void process_stage_6(AppData &app_data) {
  constexpr auto total_iterations = kConv4OutChannels * kConv4OutHeight * kConv4OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv3_out.data(),
                                              app_data.u_conv4_weights.data(),
                                              app_data.u_conv4_bias.data(),
                                              app_data.u_conv4_out.data(),
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

void process_stage_7(AppData &app_data) {
  constexpr auto total_iterations = kConv5OutChannels * kConv5OutHeight * kConv5OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  conv2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv4_out.data(),
                                              app_data.u_conv5_weights.data(),
                                              app_data.u_conv5_bias.data(),
                                              app_data.u_conv5_out.data(),
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

void process_stage_8(AppData &app_data) {
  constexpr auto total_iterations = kConv5OutChannels * kPool3OutHeight * kPool3OutWidth;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  maxpool2d<<<grid_dim, block_dim, shared_mem>>>(app_data.u_conv5_out.data(),
                                                 app_data.u_pool3_out.data(),
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

void process_stage_9(AppData &app_data) {
  constexpr auto total_iterations = kLinearOutFeatures;

  SETUP_DEFAULT_LAUNCH_PARAMS(total_iterations, 256);

  linear<<<grid_dim, block_dim, shared_mem>>>(app_data.u_pool3_out.data(),
                                              app_data.u_linear_weights.data(),
                                              app_data.u_linear_bias.data(),
                                              app_data.u_linear_out.data(),
                                              kLinearInFeatures,
                                              kLinearOutFeatures);
}

}  // namespace cifar_dense::cuda