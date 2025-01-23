#pragma once

#include <cstdint>

namespace tree {

namespace omp {

void bucket_sort(uint32_t *input_array, int dim, int n_buckets = 16);

}  // namespace omp

}  // namespace tree
