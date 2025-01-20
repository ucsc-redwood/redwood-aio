#pragma once

#include "base_appdata.hpp"

namespace cifar_sparse {

// note this pointer may came from USM vector
struct CSRMatrix {
  const float* values;
  const int* row_ptr;
  const int* col_idx;
  int rows;
  int cols;
  int nnz;
};

// Maximum sizes for static arrays
constexpr int MAX_NNZ_CONV1 = 1728;    // 3*3*3*64
constexpr int MAX_NNZ_CONV2 = 110592;  // 3*3*64*192
constexpr int MAX_NNZ_CONV3 = 663552;  // 3*3*192*384
constexpr int MAX_NNZ_CONV4 = 884736;  // 3*3*384*256
constexpr int MAX_NNZ_CONV5 = 589824;  // 3*3*256*256
constexpr int MAX_NNZ_LINEAR = 40960;  // 256*4*4*10

struct AppData : public BaseAppData {
  UsmVector<float> u_image_data;  // initial input

  UsmVector<float> u_conv1_values;
  UsmVector<int> u_conv1_row_ptr;
  UsmVector<int> u_conv1_col_idx;

  UsmVector<float> u_conv2_values;
  UsmVector<int> u_conv2_row_ptr;
  UsmVector<int> u_conv2_col_idx;

  UsmVector<float> u_conv3_values;
  UsmVector<int> u_conv3_row_ptr;
  UsmVector<int> u_conv3_col_idx;

  UsmVector<float> u_conv4_values;
  UsmVector<int> u_conv4_row_ptr;
  UsmVector<int> u_conv4_col_idx;

  UsmVector<float> u_conv5_values;
  UsmVector<int> u_conv5_row_ptr;
  UsmVector<int> u_conv5_col_idx;

  UsmVector<float> u_linear_values;
  UsmVector<int> u_linear_row_ptr;
  UsmVector<int> u_linear_col_idx;

  UsmVector<float> u_conv1_output;
  UsmVector<float> u_pool1_output;
  UsmVector<float> u_conv2_output;
  UsmVector<float> u_pool2_output;
  UsmVector<float> u_conv3_output;
  UsmVector<float> u_conv4_output;
  UsmVector<float> u_conv5_output;
  UsmVector<float> u_pool3_output;
  UsmVector<float> u_linear_output;  // final output

  UsmVector<float> u_conv1_bias;
  UsmVector<float> u_conv2_bias;
  UsmVector<float> u_conv3_bias;
  UsmVector<float> u_conv4_bias;
  UsmVector<float> u_conv5_bias;
  UsmVector<float> u_linear_bias;

  CSRMatrix conv1_weights;
  CSRMatrix conv2_weights;
  CSRMatrix conv3_weights;
  CSRMatrix conv4_weights;
  CSRMatrix conv5_weights;
  CSRMatrix linear_weights;

  explicit AppData(std::pmr::memory_resource* mr);
};

}  // namespace cifar_sparse
