/******************************************************************************
 * Compile with:  nvcc -o choose_kernels choose_kernels.cu
 *
 * Usage examples:
 *   ./choose_kernels         (all GPU)
 *   ./choose_kernels 1       (A=CPU, B=GPU, C=GPU)
 *   ./choose_kernels 1 0 1   (A=CPU, B=GPU, C=CPU)
 *   ./choose_kernels 1 2 3   (A=CPU, B=CPU, C=CPU) etc.
 ******************************************************************************/

#include <algorithm>  // for std::min
#include <cstdio>
#include <cstdlib>
#include <iostream>

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

// ---------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------
__global__ void kernelA_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = idx;
  }
}

__global__ void kernelB_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] += 10;
  }
}

__global__ void kernelC_GPU(int* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] *= 2;
  }
}

// ---------------------------------------------------------------------
// CPU (host) versions of the same "kernels."
// We'll call them from a Host Node in the CUDA Graph.
// ---------------------------------------------------------------------
struct HostKernelParams {
  int* data;  // Points to pinned host memory
  int N;
};

static void kernelA_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] = i;
  }
}

static void kernelB_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] += 10;
  }
}

static void kernelC_CPU(void* userData) {
  HostKernelParams* p = reinterpret_cast<HostKernelParams*>(userData);
  for (int i = 0; i < p->N; i++) {
    p->data[i] *= 2;
  }
}

int main(int argc, char* argv[]) {
  // Decide which versions are CPU vs GPU, based on command line.
  // We'll parse up to 3 integers; nonzero => CPU, zero => GPU.
  // If fewer than 3 arguments, the missing kernels default to GPU.
  int modeA = (argc > 1) ? std::atoi(argv[1]) : 0;  // 0=GPU, nonzero=CPU
  int modeB = (argc > 2) ? std::atoi(argv[2]) : 0;
  int modeC = (argc > 3) ? std::atoi(argv[3]) : 0;

  std::cout << "Modes: A=" << (modeA ? "CPU" : "GPU")
            << ", B=" << (modeB ? "CPU" : "GPU")
            << ", C=" << (modeC ? "CPU" : "GPU") << std::endl;

  // Problem size
  const int N = 16;
  size_t size = N * sizeof(int);

  // 1. Allocate pinned (page-locked) host memory, mapped for GPU
  int* h_data = nullptr;
  CHECK_CUDA(cudaHostAlloc(&h_data, size, cudaHostAllocMapped));

  // 2. Get the device pointer that refers to the same pinned host buffer
  int* d_data = nullptr;
  CHECK_CUDA(cudaHostGetDevicePointer(&d_data, h_data, 0));

  // We'll keep a small struct to pass into the CPU host nodes
  HostKernelParams hostParams;
  hostParams.data = h_data;
  hostParams.N = N;

  // 3. Create an empty CUDA graph
  cudaGraph_t graph;
  CHECK_CUDA(cudaGraphCreate(&graph, 0));

  // Prepare grid/block for GPU kernels
  dim3 blockDim(8);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

  // We'll create an array of node handles for each step: A, B, C
  cudaGraphNode_t nodeA, nodeB, nodeC;

  // -------------- Node A --------------
  if (modeA == 0) {
    // GPU kernel node
    cudaKernelNodeParams aParams = {0};
    aParams.func = (void*)kernelA_GPU;
    aParams.gridDim = gridDim;
    aParams.blockDim = blockDim;
    aParams.sharedMemBytes = 0;
    void* kernelAArgs[2] = {(void*)&d_data, (void*)&hostParams.N};
    aParams.kernelParams = kernelAArgs;
    aParams.extra = nullptr;

    CHECK_CUDA(cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &aParams));
  } else {
    // CPU (host) node
    cudaHostNodeParams hostAParams = {0};
    hostAParams.fn = kernelA_CPU;
    hostAParams.userData = &hostParams;

    CHECK_CUDA(cudaGraphAddHostNode(&nodeA, graph, nullptr, 0, &hostAParams));
  }

  // -------------- Node B --------------
  if (modeB == 0) {
    // GPU kernel node
    cudaKernelNodeParams bParams = {0};
    bParams.func = (void*)kernelB_GPU;
    bParams.gridDim = gridDim;
    bParams.blockDim = blockDim;
    bParams.sharedMemBytes = 0;
    void* kernelBArgs[2] = {(void*)&d_data, (void*)&hostParams.N};
    bParams.kernelParams = kernelBArgs;
    bParams.extra = nullptr;

    // B depends on nodeA
    CHECK_CUDA(cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &bParams));
  } else {
    // CPU node
    cudaHostNodeParams hostBParams = {0};
    hostBParams.fn = kernelB_CPU;
    hostBParams.userData = &hostParams;

    // B depends on nodeA
    CHECK_CUDA(cudaGraphAddHostNode(&nodeB, graph, &nodeA, 1, &hostBParams));
  }

  // -------------- Node C --------------
  if (modeC == 0) {
    // GPU kernel node
    cudaKernelNodeParams cParams = {0};
    cParams.func = (void*)kernelC_GPU;
    cParams.gridDim = gridDim;
    cParams.blockDim = blockDim;
    cParams.sharedMemBytes = 0;
    void* kernelCArgs[2] = {(void*)&d_data, (void*)&hostParams.N};
    cParams.kernelParams = kernelCArgs;
    cParams.extra = nullptr;

    // C depends on nodeB
    CHECK_CUDA(cudaGraphAddKernelNode(&nodeC, graph, &nodeB, 1, &cParams));
  } else {
    // CPU node
    cudaHostNodeParams hostCParams = {0};
    hostCParams.fn = kernelC_CPU;
    hostCParams.userData = &hostParams;

    // C depends on nodeB
    CHECK_CUDA(cudaGraphAddHostNode(&nodeC, graph, &nodeB, 1, &hostCParams));
  }

  // 4. Instantiate the graph
  cudaGraphExec_t graphExec;
  CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // 5. Launch the graph
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 6. Check results. We always expect data[i] == 2 * (i + 10),
  // because it's "i → i+10 → multiply by 2."
  bool correct = true;
  for (int i = 0; i < N; i++) {
    int expected = 2 * (i + 10);
    if (h_data[i] != expected) {
      std::cerr << "Mismatch at index " << i << ": " << h_data[i]
                << " (got) vs " << expected << " (expected)\n";
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "All results are correct!\n";
  } else {
    std::cout << "Results are incorrect.\n";
  }

  // Print the first 10 elements
  std::cout << "First 10 elements: ";
  for (int i = 0; i < std::min(10, N); i++) {
    std::cout << h_data[i] << " ";
  }
  std::cout << std::endl;

  // 7. Cleanup
  CHECK_CUDA(cudaGraphExecDestroy(graphExec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_CUDA(cudaFreeHost(h_data));

  return 0;
}
