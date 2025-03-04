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

// #include <cuda_runtime.h>
// #include <iostream>

// int main() {
//     float *h_ptr, *d_ptr;
//     size_t size = 1024 * sizeof(float);

//     // Allocate pinned memory
//     cudaHostAlloc((void **)&h_ptr, size, cudaHostAllocMapped);

//     // Get device pointer to the same host memory
//     cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);

//     // Initialize data on host
//     for (int i = 0; i < 1024; i++) {
//         h_ptr[i] = static_cast<float>(i);
//     }

//     // Kernel can use d_ptr, which is mapped to the same memory as h_ptr
//     // kernel<<<blocks, threads>>>(d_ptr);

//     // Free pinned memory
//     cudaFreeHost(h_ptr);

//     std::cout << "Pinned memory allocated and freed successfully!" << std::endl;
//     return 0;
// }

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

  // auto it = d_h_map_.find(p);
  // if (it != d_h_map_.end()) {
  //   CUDA_CHECK(cudaFreeHost(it->second));
  //   d_h_map_.erase(it);
  // }
}

bool CudaMemoryResource_PinnedHost::do_is_equal(const memory_resource& other) const noexcept {
  return dynamic_cast<const CudaMemoryResource_PinnedHost*>(&other) != nullptr;
}

}  // namespace cuda
