#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "builtin-apps/affinity.hpp"

// Helper function to merge sorted segments
void merge_segments(const std::vector<uint32_t>& input,
                    std::vector<uint32_t>& output,
                    size_t start1,
                    size_t end1,
                    size_t start2,
                    size_t end2,
                    size_t output_start) {
  size_t i = start1, j = start2, k = output_start;
  while (i < end1 && j < end2) {
    if (input[i] <= input[j])
      output[k++] = input[i++];
    else
      output[k++] = input[j++];
  }
  while (i < end1) output[k++] = input[i++];
  while (j < end2) output[k++] = input[j++];
}

void parallel_sort(std::vector<uint32_t>& buffer_input,
                   std::vector<uint32_t>& buffer_output,
                   int thread_id,
                   int num_threads) {
  const size_t N = buffer_input.size();
  const size_t segment_size = (N + num_threads - 1) / num_threads;

  size_t start = thread_id * segment_size;
  size_t end = std::min(start + segment_size, N);

  // Step 1: Each thread sorts its segment
  if (start < end) {
    std::sort(buffer_input.begin() + start, buffer_input.begin() + end);
  }

#pragma omp barrier

  // Step 2: Iterative merging performed by a single thread after sorting
  if (thread_id == 0) {
    std::vector<uint32_t>* src = &buffer_input;
    std::vector<uint32_t>* dst = &buffer_output;

    for (size_t width = segment_size; width < N; width *= 2) {
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < N; i += 2 * width) {
        size_t start1 = i;
        size_t end1 = std::min(start1 + width, N);
        size_t start2 = end1;
        size_t end2 = std::min(start2 + width, N);

        merge_segments(*src, *dst, start1, end1, start2, end2, start1);
      }
      std::swap(src, dst);
    }

    if (src != &buffer_output) {
      buffer_output = *src;
    }
  }

#pragma omp barrier
}

// Example usage:
int main() {
  const int num_threads = 4;
  omp_set_num_threads(num_threads);

  const size_t N = 640 * 480;
  std::vector<uint32_t> buffer_input(N);
  std::vector<uint32_t> buffer_output(N);

  // Initialize input data (example: random data)
  std::generate(buffer_input.begin(), buffer_input.end(), []() { return rand(); });

  std::vector<int> cores = {0, 1, 2, 3};

  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
  {
    // Custom binding or thread setup (platform-specific), if needed
    bind_thread_to_cores(cores);

    const int tid = omp_get_thread_num();
    parallel_sort(buffer_input, buffer_output, tid, num_threads);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Time taken: " << elapsed.count() << " ms\n";

  // buffer_output now contains the sorted data

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

  return 0;
}
