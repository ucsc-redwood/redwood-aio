#pragma once

#include <omp.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "../dense_appdata.hpp"

namespace cifar_dense {

namespace omp {

// ----------------------------------------------------------------------------
// Convolution 2D (Dense)
// Multi-threaded version
// ----------------------------------------------------------------------------

inline void conv2d_omp(const float *input_data,
                       [[maybe_unused]] const int image_input_channels,
                       const int input_height,
                       const int input_width,
                       const float *weight_data,
                       [[maybe_unused]] const int weight_output_channels,
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
                       // Threading parameters
                       const int start,
                       const int end) {
  // Compute output dimensions
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

#pragma omp for
  for (int index = start; index < end; ++index) {
    int out_channel = index / (output_height * output_width);
    int y = (index / output_width) % output_height;
    int x = index % output_width;

    float sum = 0.0f;
    for (int in_channel = 0; in_channel < weight_input_channels; ++in_channel) {
      for (int ky = 0; ky < weight_height; ++ky) {
        int image_y_base = y * stride + ky - padding;
        for (int kx = 0; kx < weight_width; ++kx) {
          int image_x = x * stride + kx - padding;
          if (image_y_base >= 0 && image_y_base < input_height && image_x >= 0 &&
              image_x < input_width) {
            int file_index = ((in_channel * input_height + image_y_base) * input_width + image_x);
            int weight_index =
                ((((out_channel * weight_input_channels) + in_channel) * weight_height + ky) *
                     weight_width +
                 kx);
            sum += input_data[file_index] * weight_data[weight_index];
          }
        }
      }
    }
    // Add bias
    if (bias_data && out_channel < bias_number_of_elements) {
      sum += bias_data[out_channel];
    }
    // Apply ReLU if needed
    if (relu && sum < 0) {
      sum = 0.0f;
    }
    // Store result
    output_data[(out_channel * output_height + y) * output_width + x] = sum;
  }
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// Multi-threaded version
// ----------------------------------------------------------------------------

inline void maxpool2d_omp(const float *input_data,
                          [[maybe_unused]] const int input_channels,
                          const int input_height,
                          const int input_width,
                          const int pool_size,
                          const int stride,
                          float *output_data,
                          const int start,
                          const int end) {
  int output_height = (input_height - pool_size) / stride + 1;
  int output_width = (input_width - pool_size) / stride + 1;

#pragma omp for
  for (int index = start; index < end; index++) {
    int c = index / (output_height * output_width);
    int h = (index / output_width) % output_height;
    int w = index % output_width;

    float max_val = -FLT_MAX;
    for (int p = 0; p < pool_size * pool_size; p++) {
      int ph = p / pool_size;
      int pw = p % pool_size;

      int input_h = h * stride + ph;
      int input_w = w * stride + pw;
      if (input_h < input_height && input_w < input_width) {
        int input_index = c * (input_height * input_width) + input_h * input_width + input_w;
        max_val = std::max(max_val, input_data[input_index]);
      }
    }
    int output_index = c * (output_height * output_width) + h * output_width + w;
    output_data[output_index] = max_val;
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// Multi-threaded version
// ----------------------------------------------------------------------------

inline void linear_omp(const float *input,
                       const float *weights,
                       const float *bias,
                       float *output,
                       const uint32_t input_size,
                       [[maybe_unused]] const uint32_t output_size,
                       const int start,
                       const int end) {
#pragma omp for
  for (int i = start; i < end; ++i) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < input_size; ++j) {
      sum += input[j] * weights[i * input_size + j];
    }
    output[i] = sum + bias[i];
  }
}

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

inline void process_stage_1(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv1OutChannels * kConv1OutHeight * kConv1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  conv2d_omp(app_data.u_image.data(),
             kInputChannels,  // image_input_channels
             kInputHeight,
             kInputWidth,
             app_data.u_conv1_weights.data(),
             kConv1OutChannels,
             kInputChannels,
             kKernelSize,  // weight_height
             kKernelSize,  // weight_width
             app_data.u_conv1_bias.data(),
             kConv1BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv1_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

inline void process_stage_2(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv1OutChannels * kPool1OutHeight * kPool1OutWidth;

  const int start = 0;
  const int end = total_iterations;

  maxpool2d_omp(app_data.u_conv1_out.data(),
                kConv1OutChannels,
                kConv1OutHeight,
                kConv1OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool1_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------------------

inline void process_stage_3(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv2OutChannels * kConv2OutHeight * kConv2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  conv2d_omp(app_data.u_pool1_out.data(),
             kConv1OutChannels,
             kPool1OutHeight,
             kPool1OutWidth,
             app_data.u_conv2_weights.data(),
             kConv2OutChannels,
             kConv1OutChannels,
             kKernelSize,  // weight_height
             kKernelSize,  // weight_width
             app_data.u_conv2_bias.data(),
             kConv2BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv2_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------------------

inline void process_stage_4(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv2OutChannels * kPool2OutHeight * kPool2OutWidth;

  const int start = 0;
  const int end = total_iterations;

  maxpool2d_omp(app_data.u_conv2_out.data(),
                kConv2OutChannels,
                kConv2OutHeight,
                kConv2OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool2_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------------------

inline void process_stage_5(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv3OutChannels * kConv3OutHeight * kConv3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  conv2d_omp(app_data.u_pool2_out.data(),
             kConv2OutChannels,
             kPool2OutHeight,
             kPool2OutWidth,
             app_data.u_conv3_weights.data(),
             kConv3OutChannels,
             kConv2OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv3_bias.data(),
             kConv3BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv3_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------------------

inline void process_stage_6(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv4OutChannels * kConv4OutHeight * kConv4OutWidth;

  const int start = 0;
  const int end = total_iterations;

  conv2d_omp(app_data.u_conv3_out.data(),
             kConv3OutChannels,
             kConv3OutHeight,
             kConv3OutWidth,
             app_data.u_conv4_weights.data(),
             kConv4OutChannels,
             kConv3OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv4_bias.data(),
             kConv4BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv4_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------------------

inline void process_stage_7(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv5OutChannels * kConv5OutHeight * kConv5OutWidth;

  const int start = 0;
  const int end = total_iterations;

  conv2d_omp(app_data.u_conv4_out.data(),
             kConv4OutChannels,
             kConv4OutHeight,
             kConv4OutWidth,
             app_data.u_conv5_weights.data(),
             kConv5OutChannels,
             kConv4OutChannels,
             kKernelSize,
             kKernelSize,
             app_data.u_conv5_bias.data(),
             kConv5BiasSize,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv5_out.data(),
             start,
             end);
}

// ----------------------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------------------

inline void process_stage_8(cifar_dense::AppData &app_data) {
  const int total_iterations = kConv5OutChannels * kPool3OutHeight * kPool3OutWidth;

  const int start = 0;
  const int end = total_iterations;

  maxpool2d_omp(app_data.u_conv5_out.data(),
                kConv5OutChannels,
                kConv5OutHeight,
                kConv5OutWidth,
                kPoolSize,
                kPoolStride,
                app_data.u_pool3_out.data(),
                start,
                end);
}

// ----------------------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------------------

inline void process_stage_9(cifar_dense::AppData &app_data) {
  const int total_iterations = kLinearOutFeatures;

  const int start = 0;
  const int end = total_iterations;

  linear_omp(app_data.u_pool3_out.data(),
             app_data.u_linear_weights.data(),
             app_data.u_linear_bias.data(),
             app_data.u_linear_out.data(),
             kLinearInFeatures,
             kLinearOutFeatures,
             start,
             end);
}

}  // namespace omp

}  // namespace cifar_dense
