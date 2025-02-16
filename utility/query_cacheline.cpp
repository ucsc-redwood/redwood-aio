#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>

int main() {
    // Allocate a large array (16 MB)
    const size_t arraySize = 16 * 1024 * 1024; // 16 MB
    std::vector<char> data(arraySize, 1);

    // Number of iterations to average the timing.
    const size_t outerIterations = 1000;

    std::cout << "Stride (bytes), Total Time (ns)\n";

    // Try different strides. (Stride below cache line should have lower access cost.)
    // We start at 4 bytes and go up to 512 bytes in steps of 4.
    for (size_t stride = 4; stride <= 512; stride += 4) {
        volatile uint64_t sum = 0; // volatile prevents optimizing the loop away

        auto startTime = std::chrono::high_resolution_clock::now();

        // Outer loop to accumulate timing over many iterations.
        for (size_t iter = 0; iter < outerIterations; ++iter) {
            // Walk through the array using the given stride.
            for (size_t i = 0; i < arraySize; i += stride) {
                sum += data[i];
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

        std::cout << stride << ", " << duration << "\n";

    std::cout << "Dummy sum: " << sum << "\n";

    }

    // Print sum to prevent optimizing away the loop completely.

    return 0;
}
