#pragma once

#include <omp.h>
#include <spdlog/spdlog.h>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

enum class LogKernelType {
  kOMP,
  kCUDA,
  kVK,
};

template <LogKernelType kernel_type>
inline void log_kernel(const int stage, const void *appdata_addr) {
  if constexpr (kernel_type == LogKernelType::kOMP) {
    uint64_t thread_id = 0;

#if defined(_WIN32) || defined(_WIN64)
    thread_id = (uint64_t)GetCurrentThreadId();
#else
    thread_id = (uint64_t)pthread_self();
#endif
    spdlog::debug("[omp][{}][thread {}/{}] process_stage_{}, app_data: {:p}",
                  thread_id,
                  omp_get_thread_num(),
                  omp_get_num_threads(),
                  stage,
                  appdata_addr);
  } else if constexpr (kernel_type == LogKernelType::kCUDA) {
    spdlog::debug("[cuda] process_stage_{}, app_data: {:p}", stage, appdata_addr);
  } else if constexpr (kernel_type == LogKernelType::kVK) {
    spdlog::debug("[vk] process_stage_{}, app_data: {:p}", stage, appdata_addr);
  }
}

// #ifdef NDEBUG
// #define LOG_KERNEL(kernel_type, stage, appdata) ((void)0)
// #else
#define LOG_KERNEL(kernel_type, stage, appdata) log_kernel<kernel_type>(stage, appdata)
// #endif

// Example usage:
// In your kernel code:
//
// void process_stage_1(cifar_dense::AppData &app_data) {
//   // Log the start of kernel execution
//   LOG_KERNEL<LogKernelType::kOMP>(1, &app_data);
//
//   // Kernel code here...
// }
//
// This will output something like:
// [omp][12345][thread 0/4] process_stage_1, app_data: 0x7fff1234
//
// For CUDA:
// LOG_KERNEL<LogKernelType::kCUDA>(1, &app_data);
// [cuda] process_stage_1, app_data: 0x7fff1234
//
// For Vulkan:
// LOG_KERNEL<LogKernelType::kVK>(1, &app_data);
// [vk] process_stage_1, app_data: 0x7fff1234
