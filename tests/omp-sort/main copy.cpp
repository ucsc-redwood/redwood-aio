#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

// Parallel Radix Sort for unsigned integer types
template <typename T>
void parallel_radix_sort(std::vector<T>& data) {
  static_assert(std::is_unsigned<T>::value, "Radix sort requires unsigned integer type");

  const size_t n = data.size();
  std::vector<T> temp(n);

  constexpr size_t RADIX_BITS = 8;
  constexpr size_t RADIX = 1 << RADIX_BITS;  // 256
  constexpr size_t MASK = RADIX - 1;         // 0xFF
  constexpr size_t NUM_PASSES = sizeof(T) * 8 / RADIX_BITS;

  // Get number of threads
  int max_threads = omp_get_max_threads();

  // Allocate per-thread histograms
  std::vector<std::vector<size_t>> thread_histograms(max_threads, std::vector<size_t>(RADIX));

  // For each byte
  for (size_t pass = 0; pass < NUM_PASSES; pass++) {
    const size_t shift = pass * RADIX_BITS;

// Reset all histograms
#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      std::fill(thread_histograms[thread_id].begin(), thread_histograms[thread_id].end(), 0);

// Build local histogram
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++) {
        size_t bin = (data[i] >> shift) & MASK;
        thread_histograms[thread_id][bin]++;
      }
    }

    // Combine histograms and compute global prefix sum
    std::vector<size_t> global_histogram(RADIX, 0);
    std::vector<size_t> prefix_sum(RADIX, 0);

    for (int t = 0; t < max_threads; t++) {
      for (size_t i = 0; i < RADIX; i++) {
        global_histogram[i] += thread_histograms[t][i];
      }
    }

    // Compute prefix sum
    prefix_sum[0] = 0;
    for (size_t i = 1; i < RADIX; i++) {
      prefix_sum[i] = prefix_sum[i - 1] + global_histogram[i - 1];
    }

    // Compute per-thread offsets
    std::vector<std::vector<size_t>> thread_offsets(max_threads, std::vector<size_t>(RADIX));
    for (size_t bin = 0; bin < RADIX; bin++) {
      size_t offset = prefix_sum[bin];
      for (int t = 0; t < max_threads; t++) {
        thread_offsets[t][bin] = offset;
        offset += thread_histograms[t][bin];
      }
    }

// Distribute elements to correct positions
#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();

#pragma omp for schedule(static)
      for (size_t i = 0; i < n; i++) {
        size_t bin = (data[i] >> shift) & MASK;
        size_t pos = thread_offsets[thread_id][bin]++;
        temp[pos] = data[i];
      }
    }

    // Swap buffers for next iteration
    std::swap(data, temp);
  }
}

int main() {
  // Get the number of threads
  int num_threads = omp_get_max_threads();
  std::cout << "Running with " << num_threads << " threads\n";

  // Create test data
  size_t N = 1 << 20;  // 1 million elements
  std::vector<uint32_t> buffer(N);

  // Initialize buffer with reversed values for demonstration
  for (size_t i = 0; i < N; ++i) {
    buffer[i] = static_cast<uint32_t>(N - i - 1);
  }

  // Time the sort
  auto start = std::chrono::high_resolution_clock::now();

  parallel_radix_sort(buffer);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  // Print first 10 elements
  std::cout << "First 10 elements: ";
  for (size_t i = 0; i < 10 && i < N; ++i) {
    std::cout << buffer[i] << " ";
  }
  std::cout << "\n";

  // Print last 10 elements
  std::cout << "Last 10 elements: ";
  for (size_t i = N - 10; i < N; ++i) {
    std::cout << buffer[i] << " ";
  }
  std::cout << "\n";

  // Verify sorting
  bool is_sorted = std::is_sorted(buffer.begin(), buffer.end());
  std::cout << "Sorting " << (is_sorted ? "successful" : "failed") << "!\n";
  std::cout << "Time: " << elapsed.count() << " seconds\n";

  return 0;
}
