#pragma once

#include <cuda_runtime_api.h>

// #include <stdexcept>
// #include <string>
#include <iostream>

// ----------------------------------------------------------------------------
// Math
// ----------------------------------------------------------------------------

constexpr size_t div_up(const size_t a, const size_t b) { return (a + b - 1) / b; }

// ----------------------------------------------------------------------------
// Helper function to handle CUDA errors
// ----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    cudaError_t _status = call;                                                        \
    if (_status != cudaSuccess) {                                                      \
      std::cerr << "Error: " << cudaGetErrorString(_status) << " at line " << __LINE__ \
                << std::endl;                                                          \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  } while (0)

// inline void cudaCheck(cudaError_t err, const char *file, int line) {
//   if (err != cudaSuccess) {
//     throw ::std::runtime_error(::std::string("CUDA Error: ") +
//                                cudaGetErrorString(err) + " at " + file + ":" +
//                                ::std::to_string(line));
//   }
// }

// #define CUDA_CHECK(call) cuda::cudaCheck((call), __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// Simplify launch parameters
// Need to define TOTAL_ITER (e.g., 'total_iter' = 10000), and then write some
// number for BLOCK_SIZE (e.g., 256)
// ----------------------------------------------------------------------------

#define SETUP_DEFAULT_LAUNCH_PARAMS(TOTAL_ITER, BLOCK_SIZE)     \
  static const auto block_dim = dim3{BLOCK_SIZE, 1, 1};         \
  static const auto grid_dim = div_up(TOTAL_ITER, block_dim.x); \
  static constexpr auto shared_mem = 0;
