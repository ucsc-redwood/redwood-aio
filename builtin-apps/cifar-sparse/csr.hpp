#pragma once

// note this pointer may came from USM vector
struct CSRMatrix {
  const float* values;
  const int* row_ptr;
  const int* col_idx;
  int rows;
  int cols;
  int nnz;
};
