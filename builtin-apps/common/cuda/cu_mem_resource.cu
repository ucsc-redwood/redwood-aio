#include <cuda_runtime.h>

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

void* CudaMemoryResource::do_allocate(std::size_t bytes, std::size_t) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
  return ptr;
}

void CudaMemoryResource::do_deallocate(void* p, std::size_t, std::size_t) {
  CUDA_CHECK(cudaFree(p));
}

bool CudaMemoryResource::do_is_equal(const memory_resource& other) const noexcept {
  return dynamic_cast<const CudaMemoryResource*>(&other) != nullptr;
}

}  // namespace cuda
