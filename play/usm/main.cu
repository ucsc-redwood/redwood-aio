
#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <thread>

inline void checkCuda(cudaError_t result, const char* file, int line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result) << "("
              << cudaGetErrorString(result) << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(x) checkCuda(x, __FILE__, __LINE__)

namespace omp {

void process_stage_1(float* a, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    a[i] += 1.0f;
  }
}

}  // namespace omp

namespace cuda {

__global__ void kernel_process_stage_1(float* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] *= 2.0f;
  }
}

void process_stage_1(float* a, int n) {
  constexpr auto threads_per_block = 256;
  const auto blocks = (n + threads_per_block - 1) / threads_per_block;

  kernel_process_stage_1<<<blocks, threads_per_block>>>(a, n);
  CHECK_CUDA(cudaDeviceSynchronize());
}

}  // namespace cuda

int main() {
  constexpr auto n = 1'000'000;

  float* u_input;
  CHECK_CUDA(cudaMallocManaged(&u_input, n * sizeof(float)));

  float* u_output;
  CHECK_CUDA(cudaMallocManaged(&u_output, n * sizeof(float)));

  std::fill_n(u_input, n, 1.0f);

  std::thread t_omp([&]() { omp::process_stage_1(u_input, n); });
  std::thread t_cuda([&]() { cuda::process_stage_1(u_input, n); });

  t_omp.join();
  t_cuda.join();

  // Verify results
  bool correct = true;
  for (int i = 0; i < n; i++) {
    if (u_input[i] != 4.0f) {  // Should be 1.0 * (1.0 + 1.0) * 2.0 = 4.0
      correct = false;
      std::cout << "Mismatch at position " << i << ": " << u_input[i]
                << " != 4.0" << std::endl;
      break;
    }
  }

  if (correct) {
    std::cout << "All results are correct!" << std::endl;
  }

  // Cleanup
  CHECK_CUDA(cudaFree(u_input));
  CHECK_CUDA(cudaFree(u_output));

  return 0;
}