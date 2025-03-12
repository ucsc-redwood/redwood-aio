#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "builtin-apps/affinity.hpp"

// Structure to hold all temporary storage needed for radix sort
template <typename T>
struct RadixSortTemp {
  const size_t n_elements;              // Number of elements to sort
  const int n_threads;                  // Number of threads to use
  static constexpr size_t RADIX = 256;  // Radix size (2^8)

  std::vector<T> temp_buffer;                          // Temporary buffer for elements
  std::vector<std::vector<size_t>> thread_histograms;  // Per-thread histograms
  std::vector<std::vector<size_t>> thread_offsets;     // Per-thread offsets
  std::vector<size_t> global_histogram;                // Global histogram
  std::vector<size_t> prefix_sum;                      // Prefix sum array

  // Constructor allocates all temporary storage
  RadixSortTemp(size_t n, int threads)
      : n_elements(n),
        n_threads(threads),
        temp_buffer(n),
        thread_histograms(threads, std::vector<size_t>(RADIX)),
        thread_offsets(threads, std::vector<size_t>(RADIX)),
        global_histogram(RADIX),
        prefix_sum(RADIX) {}

  // Calculate memory usage
  size_t get_memory_usage() const {
    size_t temp_buffer_size = n_elements * sizeof(T);
    size_t thread_histograms_size = n_threads * RADIX * sizeof(size_t);
    size_t thread_offsets_size = n_threads * RADIX * sizeof(size_t);
    size_t global_histogram_size = RADIX * sizeof(size_t);
    size_t prefix_sum_size = RADIX * sizeof(size_t);

    return temp_buffer_size + thread_histograms_size + thread_offsets_size + global_histogram_size +
           prefix_sum_size;
  }

  // Print memory usage breakdown
  void print_memory_usage() const {
    size_t temp_buffer_size = n_elements * sizeof(T);
    size_t thread_histograms_size = n_threads * RADIX * sizeof(size_t);
    size_t thread_offsets_size = n_threads * RADIX * sizeof(size_t);
    size_t global_histogram_size = RADIX * sizeof(size_t);
    size_t prefix_sum_size = RADIX * sizeof(size_t);
    size_t total = get_memory_usage();

    std::cout << "Temporary storage breakdown:\n"
              << "  Temp buffer: " << temp_buffer_size / 1024.0 << " KB\n"
              << "  Thread histograms: " << thread_histograms_size / 1024.0 << " KB\n"
              << "  Thread offsets: " << thread_offsets_size / 1024.0 << " KB\n"
              << "  Global histogram: " << global_histogram_size / 1024.0 << " KB\n"
              << "  Prefix sum: " << prefix_sum_size / 1024.0 << " KB\n"
              << "Total temporary storage: " << total / 1024.0 << " KB\n";
  }
};

// Parallel Radix Sort for unsigned integer types
template <typename T>
void parallel_radix_sort(const std::vector<T>& input,
                         std::vector<T>& output,
                         RadixSortTemp<T>& temp) {
  static_assert(std::is_unsigned<T>::value, "Radix sort requires unsigned integer type");

  const size_t n = input.size();
  std::copy(input.begin(), input.end(), output.begin());

  constexpr size_t RADIX_BITS = 8;
  constexpr size_t RADIX = 1 << RADIX_BITS;  // 256
  constexpr size_t MASK = RADIX - 1;         // 0xFF
  constexpr size_t NUM_PASSES = sizeof(T) * 8 / RADIX_BITS;

  // Get thread ID
  int thread_id = omp_get_thread_num();
  int num_threads = omp_get_num_threads();

  // For each byte
  for (size_t pass = 0; pass < NUM_PASSES; pass++) {
    const size_t shift = pass * RADIX_BITS;

    // Reset histograms for this thread
    std::fill(
        temp.thread_histograms[thread_id].begin(), temp.thread_histograms[thread_id].end(), 0);

// Barrier to ensure all threads have reset their histograms
#pragma omp barrier

// Build local histogram
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      size_t bin = (output[i] >> shift) & MASK;
      temp.thread_histograms[thread_id][bin]++;
    }

// Barrier to ensure all histograms are complete
#pragma omp barrier

// Thread 0 combines histograms and computes prefix sums
#pragma omp single
    {
      // Reset global histogram
      std::fill(temp.global_histogram.begin(), temp.global_histogram.end(), 0);

      // Combine histograms
      for (int t = 0; t < num_threads; t++) {
        for (size_t i = 0; i < RADIX; i++) {
          temp.global_histogram[i] += temp.thread_histograms[t][i];
        }
      }

      // Compute prefix sum
      temp.prefix_sum[0] = 0;
      for (size_t i = 1; i < RADIX; i++) {
        temp.prefix_sum[i] = temp.prefix_sum[i - 1] + temp.global_histogram[i - 1];
      }

      // Compute per-thread offsets
      for (size_t bin = 0; bin < RADIX; bin++) {
        size_t offset = temp.prefix_sum[bin];
        for (int t = 0; t < num_threads; t++) {
          temp.thread_offsets[t][bin] = offset;
          offset += temp.thread_histograms[t][bin];
        }
      }
    }

// Barrier to ensure prefix sums and offsets are ready
#pragma omp barrier

// Distribute elements to correct positions
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      size_t bin = (output[i] >> shift) & MASK;
      size_t pos = temp.thread_offsets[thread_id][bin]++;
      temp.temp_buffer[pos] = output[i];
    }

// Barrier to ensure all elements are distributed
#pragma omp barrier

// Copy back to output buffer
#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      output[i] = temp.temp_buffer[i];
    }

// Barrier before next pass
#pragma omp barrier
  }
}

int main(int argc, char** argv) {
  std::vector<int> cores = {1, 2, 3, 4};

  int num_threads = cores.size();

  //   int num_threads = 2;
  //   if (argc > 1) {
  //     num_threads = std::stoi(argv[1]);
  //   }

  // Get the number of threads
  //   int num_threads = omp_get_max_threads();

  std::cout << "Running with " << num_threads << " threads\n";

  // Create test data
  size_t N = 1 << 20;  // 1 million elements
  std::vector<uint32_t> buffer_input(N);
  std::vector<uint32_t> buffer_output(N);

  // Initialize buffer with reversed values for demonstration
  for (size_t i = 0; i < N; ++i) {
    buffer_input[i] = static_cast<uint32_t>(N - i - 1);
  }

  // Create temporary storage structure
  RadixSortTemp<uint32_t> temp_storage(N, num_threads);

  // Print memory usage information
  temp_storage.print_memory_usage();

  // Time the sort
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
  {
    // You can do thread binding or other setup here
    bind_thread_to_cores(cores);

    parallel_radix_sort(buffer_input, buffer_output, temp_storage);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // Print first 10 elements
  std::cout << "First 10 elements: ";
  for (size_t i = 0; i < 10 && i < N; ++i) {
    std::cout << buffer_output[i] << " ";
  }
  std::cout << "\n";

  // Print last 10 elements
  std::cout << "Last 10 elements: ";
  for (size_t i = N - 10; i < N; ++i) {
    std::cout << buffer_output[i] << " ";
  }
  std::cout << "\n";

  // Verify sorting
  bool is_sorted = std::ranges::is_sorted(buffer_output);
  std::cout << "Sorting " << (is_sorted ? "successful" : "failed") << "!\n";
  std::cout << "Time: " << elapsed.count() << " milliseconds\n";

  return 0;
}
