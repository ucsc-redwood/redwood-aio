#pragma once

#include <spdlog/spdlog.h>

#include <memory_resource>
#include <string>

#include "helpers.cuh"

namespace cuda {

// Helper function to format bytes into human readable string
std::string format_bytes(std::size_t bytes) {
  constexpr std::size_t KB = 1024;
  constexpr std::size_t MB = KB * 1024;
  constexpr std::size_t GB = MB * 1024;

  if (bytes >= GB) {
    return fmt::format("{:.2f} GB", static_cast<double>(bytes) / GB);
  } else if (bytes >= MB) {
    return fmt::format("{:.2f} MB", static_cast<double>(bytes) / MB);
  } else if (bytes >= KB) {
    return fmt::format("{:.2f} KB", static_cast<double>(bytes) / KB);
  }
  return fmt::format("{} bytes", bytes);
}

// ----------------------------------------------------------------------------
// CudaManagedResource
// ----------------------------------------------------------------------------

// Custom memory resource that uses cudaMallocManaged and cudaFree
class CudaManagedResource final : public std::pmr::memory_resource {
 protected:
  // Allocate memory using cudaMallocManaged. Alignment is ignored.
  void *do_allocate(std::size_t bytes, std::size_t /*alignment*/) override {
    void *ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, bytes, cudaMemAttachHost);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }

    spdlog::trace(
        "CudaManagedResource::do_allocate: {}, {}", std::to_address(ptr), format_bytes(bytes));

    return ptr;
  }

  // Deallocate memory using cudaFree
  void do_deallocate(void *p, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
    spdlog::trace("CudaManagedResource::do_deallocate: {}", std::to_address(p));

    CheckCuda(cudaFree(p));
  }

  // Compares memory resources: here we simply use pointer equality
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
};

// ----------------------------------------------------------------------------
// CudaPinnedResource
// ----------------------------------------------------------------------------

// Custom memory resource that uses cudaHostAlloc and cudaHostGetDevicePointer
class CudaPinnedResource final : public std::pmr::memory_resource {
 protected:
  // Allocate memory using cudaHostAlloc. Alignment is ignored.
  void *do_allocate(std::size_t bytes, std::size_t /*alignment*/) override {
    void *h_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&h_ptr, bytes, cudaHostAllocMapped);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }

    void *d_ptr = nullptr;
    err = cudaHostGetDevicePointer(&d_ptr, h_ptr, 0);
    if (err != cudaSuccess) {
      throw std::bad_alloc();
    }

    spdlog::trace(
        "CudaPinnedResource::do_allocate: {}, {}", std::to_address(d_ptr), format_bytes(bytes));

    return d_ptr;
  }

  // Deallocate memory using cudaFree
  void do_deallocate(void *p, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
    spdlog::trace("CudaPinnedResource::do_deallocate: {}", std::to_address(p));

    CheckCuda(cudaFreeHost(p));
  }

  // Compares memory resources: here we simply use pointer equality
  bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    return this == &other;
  }
};

}  // namespace cuda
