#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>  // For std::cerr

// Simple check macro
#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t _status = call;                                            \
    if (_status != cudaSuccess) {                                          \
      std::cerr << "Error: " << cudaGetErrorString(_status) << " at line " \
                << __LINE__ << std::endl;                                  \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

/**
 * A simple helper to fill two matrices with random integers.
 */
void randFill(int *p, int *q, int hp, int wp, int hq, int wq) {
  // Fill p (hp x wp)
  for (int i = 0; i < hp * wp; i++) {
    p[i] = rand() % 10;
  }
  // Fill q (hq x wq)
  for (int i = 0; i < hq * wq; i++) {
    q[i] = rand() % 10;
  }
}

/**
 * Kernel for matrix multiplication of:
 *     p is (hp x wp),
 *     q is (hq x wq),
 * requiring wp == hq for valid multiplication.
 * The result r is (hp x wq).
 */
__global__ void matrixMul(
    int *p, int *q, int *r, int hp, int hq, int wp, int wq) {
  // Compute row and column of the element
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < hp && col < wq) {
    int sum = 0;
    // wp == hq for valid multiplication
    for (int k = 0; k < wp; k++) {
      sum += p[row * wp + k] * q[k * wq + col];
    }
    r[row * wq + col] = sum;
  }
}

/**
 * MatrixMul function
 * Demonstrates usage of Unified Memory and prefetch using
 * cudaStreamAttachMemAsync.
 */
void MatrixMul(int hp, int hq, int wp, int wq) {
  // Unified memory pointers
  int *p, *q, *r;

  // Calculate sizes in bytes
  size_t sizeP = hp * wp * sizeof(int);
  size_t sizeQ = hq * wq * sizeof(int);
  size_t sizeR = hp * wq * sizeof(int);

  // Allocate in Unified Memory
  // p and q are primarily used on the CPU (host),
  // r is primarily used on the GPU by default.
  CHECK_CUDA(cudaMallocManaged(&p, sizeP, cudaMemAttachHost));
  CHECK_CUDA(cudaMallocManaged(&q, sizeQ, cudaMemAttachHost));
  CHECK_CUDA(cudaMallocManaged(&r, sizeR));

  // Initialize with random values on the CPU
  srand((unsigned)time(NULL));
  randFill(p, q, hp, wp, hq, wq);

  // Prefetch / attach 'p' and 'q' to GPU as they are needed for computation
  CHECK_CUDA(cudaStreamAttachMemAsync(NULL, p, 0, cudaMemAttachGlobal));
  CHECK_CUDA(cudaStreamAttachMemAsync(NULL, q, 0, cudaMemAttachGlobal));

  // Define execution configuration
  dim3 block(16, 16);
  dim3 grid((wq + block.x - 1) / block.x, (hp + block.y - 1) / block.y);

  // Launch the matrix multiplication kernel
  matrixMul<<<grid, block>>>(p, q, r, hp, hq, wp, wq);

  // Check for any kernel launch errors
  CHECK_CUDA(cudaGetLastError());

  // After kernel execution, we want 'r' on the CPU
  CHECK_CUDA(cudaStreamAttachMemAsync(NULL, r, 0, cudaMemAttachHost));

  // Wait for all operations to finish
  CHECK_CUDA(cudaStreamSynchronize(NULL));

  // Print the result matrix 'r'
  printf("\nResult matrix (hp x wq = %d x %d):\n", hp, wq);
  for (int i = 0; i < hp; i++) {
    for (int j = 0; j < wq; j++) {
      printf("%d ", r[i * wq + j]);
    }
    printf("\n");
  }
  printf("\n");

  // Free the Unified Memory
  CHECK_CUDA(cudaFree(p));
  CHECK_CUDA(cudaFree(q));
  CHECK_CUDA(cudaFree(r));
}

int main() {
  /**
   * Let's define:
   *   p is (hp x wp) = (2 x 3),
   *   q is (hq x wq) = (3 x 2).
   * Then r is (hp x wq) = (2 x 2).
   */
  int hp = 2;
  int wp = 3;
  int hq = 3;
  int wq = 2;

  // Ensure wp == hq for valid matrix multiplication
  if (wp != hq) {
    printf("Error: wp (%d) must match hq (%d)!\n", wp, hq);
    return -1;
  }

  // Perform the matrix multiplication
  MatrixMul(hp, hq, wp, wq);

  return 0;
}
