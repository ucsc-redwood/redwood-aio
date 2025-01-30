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

// start, end = 0, weight_matrix.rows;

inline void conv2d_omp(const float *input_data,
                       [[maybe_unused]] int image_input_channels,
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
                       int end) {
  int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
  int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

#pragma omp for
  for (int out_c = start; out_c < end; ++out_c) {
    int row_start = weight_matrix.row_ptr[out_c];
    int row_end = weight_matrix.row_ptr[out_c + 1];

    for (int oh = 0; oh < output_height; ++oh) {
      for (int ow = 0; ow < output_width; ++ow) {
        float sum = 0.0f;

        for (int nz_idx = row_start; nz_idx < row_end; ++nz_idx) {
          int flat_kernel_idx = weight_matrix.col_idx[nz_idx];
          float weight_value = weight_matrix.values[nz_idx];

          int in_c = flat_kernel_idx / (kernel_size * kernel_size);
          int rem = flat_kernel_idx % (kernel_size * kernel_size);
          int ky = rem / kernel_size;
          int kx = rem % kernel_size;

          int ih = oh * stride + ky - padding;
          int iw = ow * stride + kx - padding;

          if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
            int input_idx = (in_c * input_height + ih) * input_width + iw;
            sum += input_data[input_idx] * weight_value;
          }
        }

        if (bias_data && out_c < bias_size) {
          sum += bias_data[out_c];
        }

        if (relu && sum < 0) {
          sum = 0.0f;
        }

        output_data[(out_c * output_height + oh) * output_width + ow] = sum;
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Max Pooling 2D (Dense)
// ----------------------------------------------------------------------------

// start, end = 0, input_channels * output_height * output_width
inline void maxpool2d_omp(const float *input_data,
                          [[maybe_unused]] int input_channels,
                          int input_height,
                          int input_width,
                          int pool_size,
                          int stride,
                          float *output_data,
                          int start,
                          int end) {
  int output_height = (input_height - pool_size) / stride + 1;
  int output_width = (input_width - pool_size) / stride + 1;
  // int total_iterations = input_channels * output_height * output_width;

#pragma omp for
  for (int index = start; index < end; index++) {
    int c = index / (output_height * output_width);
    int h = (index / output_width) % output_height;
    int w = index % output_width;

    float max_val = -std::numeric_limits<float>::max();
    for (int p = 0; p < pool_size * pool_size; p++) {
      int ph = p / pool_size;
      int pw = p % pool_size;

      int input_h = h * stride + ph;
      int input_w = w * stride + pw;
      if (input_h < input_height && input_w < input_width) {
        int input_index =
            c * (input_height * input_width) + input_h * input_width + input_w;
        max_val = std::max(max_val, input_data[input_index]);
      }
    }
    int output_index =
        c * (output_height * output_width) + h * output_width + w;
    output_data[output_index] = max_val;
  }
}

// ----------------------------------------------------------------------------
// Linear Layer (Dense)
// ----------------------------------------------------------------------------
// start, end = 0, weight_matrix.rows
inline void linear_omp(const float *input_data,
                       const CSRMatrix &weight_matrix,
                       const float *bias_data,
                       float *output_data,
                       int start,
                       int end) {
#pragma omp for
  for (int i = start; i < end; ++i) {
    float sum = 0.0f;

    for (int nz_idx = weight_matrix.row_ptr[i];
         nz_idx < weight_matrix.row_ptr[i + 1];
         ++nz_idx) {
      int col = weight_matrix.col_idx[nz_idx];
      sum += input_data[col] * weight_matrix.values[nz_idx];
    }

    output_data[i] = sum + bias_data[i];
  }
}

inline void process_stage_1(cifar_sparse::AppData &app_data) {
  constexpr auto start = 0;
  const auto end = app_data.conv1_weights.rows;
  conv2d_omp(app_data.u_image_data.data(),
             kInputChannels,
             kInputHeight,
             kInputWidth,
             app_data.conv1_weights,
             app_data.u_conv1_bias.data(),
             64,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv1_output.data(),
             start,
             end);
}

inline void process_stage_2(cifar_sparse::AppData &app_data) {
  constexpr auto start = 0;

  constexpr auto input_channels = 64;
  constexpr auto input_height = 32;
  constexpr auto input_width = 32;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  constexpr auto end = total_iterations;

  maxpool2d_omp(app_data.u_conv1_output.data(),
                input_channels,
                input_height,
                input_width,
                kPoolSize,
                kPoolStride,
                app_data.u_pool1_output.data(),
                start,
                end);
}

inline void process_stage_3(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv2_weights.rows;

  conv2d_omp(app_data.u_pool1_output.data(),
             64,
             16,
             16,
             app_data.conv2_weights,
             app_data.u_conv2_bias.data(),
             192,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv2_output.data(),
             start,
             end);
}

inline void process_stage_4(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 192;
  constexpr auto input_height = 16;
  constexpr auto input_width = 16;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  const auto start = 0;
  const auto end = total_iterations;

  maxpool2d_omp(app_data.u_conv2_output.data(),
                input_channels,
                input_height,
                input_width,
                kPoolSize,
                kPoolStride,
                app_data.u_pool2_output.data(),
                start,
                end);
}

inline void process_stage_5(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv3_weights.rows;

  conv2d_omp(app_data.u_pool2_output.data(),
             192,
             8,
             8,
             app_data.conv3_weights,
             app_data.u_conv3_bias.data(),
             384,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv3_output.data(),
             start,
             end);
}

inline void process_stage_6(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv4_weights.rows;

  conv2d_omp(app_data.u_conv3_output.data(),
             384,
             8,
             8,
             app_data.conv4_weights,
             app_data.u_conv4_bias.data(),
             256,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv4_output.data(),
             start,
             end);
}

inline void process_stage_7(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.conv5_weights.rows;

  conv2d_omp(app_data.u_conv4_output.data(),
             256,
             8,
             8,
             app_data.conv5_weights,
             app_data.u_conv5_bias.data(),
             256,
             kKernelSize,
             kStride,
             kPadding,
             kRelu,
             app_data.u_conv5_output.data(),
             start,
             end);
}

inline void process_stage_8(cifar_sparse::AppData &app_data) {
  constexpr auto input_channels = 256;
  constexpr auto input_height = 8;
  constexpr auto input_width = 8;

  constexpr auto output_height = (input_height - kPoolSize) / kPoolStride + 1;
  constexpr auto output_width = (input_width - kPoolSize) / kPoolStride + 1;
  constexpr auto total_iterations =
      input_channels * output_height * output_width;

  const auto start = 0;
  const auto end = total_iterations;

  maxpool2d_omp(app_data.u_conv5_output.data(),
                input_channels,
                input_height,
                input_width,
                kPoolSize,
                kPoolStride,
                app_data.u_pool3_output.data(),
                start,
                end);
}

inline void process_stage_9(cifar_sparse::AppData &app_data) {
  const auto start = 0;
  const auto end = app_data.linear_weights.rows;

  linear_omp(app_data.u_pool3_output.data(),
             app_data.linear_weights,
             app_data.u_linear_bias.data(),
             app_data.u_linear_output.data(),
             start,
             end);
}

}  // namespace omp
}  // namespace cifar_sparse