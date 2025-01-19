#pragma once

#include <omp.h>
#include <sched.h>

#include <cstdlib>
#include <stdexcept>

// Function to pin the current thread to a given core
inline void bind_thread_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    throw std::runtime_error("Failed to pin thread to core " +
                             std::to_string(core_id));
  }
}

inline void simulate_heavy_work() {
  double start_time = omp_get_wtime();
  volatile double dummy =
      0.0;  // volatile to prevent the compiler from optimizing it out
  while (omp_get_wtime() - start_time < 10) {
    dummy += 1.0;
  }
}