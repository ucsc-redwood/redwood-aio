#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "cu_mem_resource.cuh"
#include "helpers.cuh"

namespace cuda {

// ----------------------------------------------------------------------------
// Unified memory
// ----------------------------------------------------------------------------

void* CudaMemoryResource::do_allocate(std::size_t bytes, std::size_t) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMallocManaged(&ptr, bytes));
  spdlog::trace("CudaMemoryResource::do_allocate allocating {} bytes, ptr = {}", bytes, ptr);
  return ptr;
}

void CudaMemoryResource::do_deallocate(void* p, std::size_t, std::size_t) {
  spdlog::trace("CudaMemoryResource::do_deallocate  ptr = {}", p);

  CUDA_CHECK(cudaFree(p));
}

bool CudaMemoryResource::do_is_equal(const memory_resource& other) const noexcept {
  return dynamic_cast<const CudaMemoryResource*>(&other) != nullptr;
}

// ----------------------------------------------------------------------------
// Pinned host memory 
// ----------------------------------------------------------------------------

void* CudaMemoryResource_PinnedHost::do_allocate(std::size_t bytes, std::size_t) {
  void* h_ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_ptr, bytes, cudaHostAllocMapped));

  void* d_ptr = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&d_ptr, h_ptr, 0));

  spdlog::trace(
      "CudaMemoryResource_PinnedHost::do_allocate allocating {} bytes, h_ptr = {}, d_ptr = {}",
      bytes,
      h_ptr,
      d_ptr);

  return d_ptr;
}

void CudaMemoryResource_PinnedHost::do_deallocate(void* p, std::size_t, std::size_t) {
  spdlog::trace("CudaMemoryResource_PinnedHost::do_deallocate ptr = {}", p);

  CUDA_CHECK(cudaFreeHost(p));
}

bool CudaMemoryResource_PinnedHost::do_is_equal(const memory_resource& other) const noexcept {
  return dynamic_cast<const CudaMemoryResource_PinnedHost*>(&other) != nullptr;
}

}  // namespace cuda
