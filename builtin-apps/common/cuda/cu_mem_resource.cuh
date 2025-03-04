#pragma once

#include <memory_resource>
#include <unordered_map>

namespace cuda {

// ----------------------------------------------------------------------------
// CudaMemoryResource implementation
// ----------------------------------------------------------------------------

class CudaMemoryResource : public std::pmr::memory_resource {
 protected:
  void* do_allocate(std::size_t bytes, std::size_t) override;

  void do_deallocate(void* p, std::size_t, std::size_t) override;

  bool do_is_equal(const memory_resource& other) const noexcept override;
};

// ----------------------------------------------------------------------------
// CudaMemoryResource_PinnedHost implementation
// ----------------------------------------------------------------------------

class CudaMemoryResource_PinnedHost : public std::pmr::memory_resource {
 protected:
  void* do_allocate(std::size_t bytes, std::size_t) override;

  void do_deallocate(void* p, std::size_t, std::size_t) override;

  bool do_is_equal(const memory_resource& other) const noexcept override;
};

}  // namespace cuda
