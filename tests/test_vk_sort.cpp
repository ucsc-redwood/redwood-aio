#include <gtest/gtest.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>

#include "base_appdata.hpp"
#include "common/vulkan/engine.hpp"

// ----------------------------------------------------------------------------
// globals
// ----------------------------------------------------------------------------

std::string device_id;

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  vulkan::Engine &get_engine() { return engine; }
  const vulkan::Engine &get_engine() const { return engine; }

 private:
  Singleton() { spdlog::info("Singleton instance created."); }
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  vulkan::Engine engine;
};

struct PushConstants {
  uint32_t g_num_elements;
};

// ----------------------------------------------------------------------------
// Test Fixtures and Helpers
// ----------------------------------------------------------------------------

class VulkanTestFixture : public ::testing::Test {
 protected:
  vulkan::Engine &engine = Singleton::getInstance().get_engine();

  std::string get_shader_name() const {
    if (device_id == "3A021JEHN02756") {
      return "tmp_single_radixsort_warp16.comp";
    } else if (device_id == "9b034f1b") {
      return "tmp_single_radixsort_warp64.comp";
    } else if (device_id == "pc" || device_id == "jetson") {
      return "tmp_single_radixsort_warp32.comp";
    }
    throw std::runtime_error("Invalid device ID");
  }
};

class VulkanSortTest : public VulkanTestFixture,
                       public testing::WithParamInterface<unsigned int> {
 protected:
  void verify_sort(unsigned int n) {
    auto mr = engine.get_mr();

    UsmVector<uint32_t> u_elements_in(n, mr);
    UsmVector<uint32_t> u_elements_out(n, mr);

    // Initialize with shuffled sequence
    std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

    // Keep CPU copy for verification
    std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                         u_elements_in.end());

    auto algo = engine
                    .algorithm(get_shader_name(),
                               {
                                   engine.get_buffer(u_elements_in.data()),
                                   engine.get_buffer(u_elements_out.data()),
                               })
                    ->set_push_constants<PushConstants>({
                        .g_num_elements = n,
                    })
                    ->build();

    auto seq = engine.sequence();
    seq->record_commands_with_blocks(algo.get(), 1);
    seq->launch_kernel_async();
    seq->sync();

    // Verify results
    EXPECT_TRUE(std::ranges::is_sorted(u_elements_out));
    std::ranges::sort(h_cpu_elements);
    EXPECT_TRUE(std::ranges::equal(h_cpu_elements, u_elements_out));
  }
};

class VulkanSortIterationTest : public VulkanTestFixture,
                                public testing::WithParamInterface<
                                    std::tuple<unsigned int, unsigned int>> {
 protected:
  void verify_sort_iterations(unsigned int n, unsigned int iterations) {
    auto mr = engine.get_mr();

    UsmVector<uint32_t> u_elements_in(n, mr);
    UsmVector<uint32_t> u_elements_out(n, mr);

    // Initialize with shuffled sequence
    std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

    // Keep CPU copy for verification
    std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                         u_elements_in.end());

    auto algo = engine
                    .algorithm(get_shader_name(),
                               {
                                   engine.get_buffer(u_elements_in.data()),
                                   engine.get_buffer(u_elements_out.data()),
                               })
                    ->set_push_constants<PushConstants>({
                        .g_num_elements = n,
                    })
                    ->build();

    auto seq = engine.sequence();

    // Run the sort multiple times
    for (unsigned int i = 0; i < iterations; ++i) {
      seq->record_commands_with_blocks(algo.get(), 1);
      seq->launch_kernel_async();
      seq->sync();
    }

    // Verify results
    bool all_zeros =
        std::ranges::all_of(u_elements_out, [](uint32_t x) { return x == 0; });
    EXPECT_FALSE(all_zeros);

    EXPECT_TRUE(std::ranges::is_sorted(u_elements_out));
    std::ranges::sort(h_cpu_elements);
    EXPECT_TRUE(std::ranges::equal(h_cpu_elements, u_elements_out));
  }
};

class VulkanSortEdgeCasesTest : public VulkanTestFixture {
 protected:
  void verify_sort_with_data(const std::vector<uint32_t> &input_data) {
    auto mr = engine.get_mr();
    const size_t n = input_data.size();

    UsmVector<uint32_t> u_elements_in(input_data.begin(), input_data.end(), mr);
    UsmVector<uint32_t> u_elements_out(n, mr);

    // Keep CPU copy for verification
    std::vector<uint32_t> h_cpu_elements = input_data;

    auto algo = engine
                    .algorithm(get_shader_name(),
                               {
                                   engine.get_buffer(u_elements_in.data()),
                                   engine.get_buffer(u_elements_out.data()),
                               })
                    ->set_push_constants<PushConstants>({
                        .g_num_elements = static_cast<uint32_t>(n),
                    })
                    ->build();

    auto seq = engine.sequence();
    seq->record_commands_with_blocks(algo.get(), 1);
    seq->launch_kernel_async();
    seq->sync();

    // Verify results
    EXPECT_TRUE(std::ranges::is_sorted(u_elements_out));
    std::ranges::sort(h_cpu_elements);
    EXPECT_TRUE(std::ranges::equal(h_cpu_elements, u_elements_out));
  }
};

// ----------------------------------------------------------------------------
// Vulkan Sort Tests
// ----------------------------------------------------------------------------

TEST_F(VulkanTestFixture, RadixSortCorrectlySortsRandomData) {
  constexpr unsigned int n = 640 * 480;
  auto mr = engine.get_mr();

  UsmVector<uint32_t> u_elements_in(n, mr);
  UsmVector<uint32_t> u_elements_out(n, mr);

  // Initialize with shuffled sequence
  std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

  // Keep CPU copy for verification
  std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                       u_elements_in.end());

  auto algo = engine
                  .algorithm(get_shader_name(),
                             {
                                 engine.get_buffer(u_elements_in.data()),
                                 engine.get_buffer(u_elements_out.data()),
                             })
                  ->set_push_constants<PushConstants>({
                      .g_num_elements = n,
                  })
                  ->build();

  auto seq = engine.sequence();
  seq->record_commands_with_blocks(algo.get(), 1);
  seq->launch_kernel_async();
  seq->sync();

  // Verify results
  EXPECT_TRUE(std::ranges::is_sorted(u_elements_out));
  std::ranges::sort(h_cpu_elements);
  EXPECT_TRUE(std::ranges::equal(h_cpu_elements, u_elements_out));
}

// Test with different input sizes
TEST_P(VulkanSortTest, SortsCorrectlyWithDifferentSizes) {
  verify_sort(GetParam());
}

INSTANTIATE_TEST_SUITE_P(VaryingSizes,
                         VulkanSortTest,
                         testing::Values(1024,         // Small dataset
                                         64 * 1024,    // Medium dataset
                                         640 * 480,    // Original test size
                                         1920 * 1080,  // Full HD size
                                         2048 * 2048   // Large power of 2
                                         ),
                         [](const testing::TestParamInfo<unsigned int> &info) {
                           return "Size" + std::to_string(info.param);
                         });

// Test with different input sizes and iteration counts
TEST_P(VulkanSortIterationTest, SortsCorrectlyWithMultipleIterations) {
  auto [size, iterations] = GetParam();
  verify_sort_iterations(size, iterations);
}

INSTANTIATE_TEST_SUITE_P(
    VaryingSizesAndIterations,
    VulkanSortIterationTest,
    testing::Combine(testing::Values(1024, 64 * 1024, 640 * 480),  // Sizes
                     testing::Values(1, 2, 5, 10, 32)  // Number of iterations
                     ),
    [](const testing::TestParamInfo<std::tuple<unsigned int, unsigned int>>
           &info) {
      return "Size" + std::to_string(std::get<0>(info.param)) + "_Iterations" +
             std::to_string(std::get<1>(info.param));
    });

// Test edge cases
TEST_F(VulkanSortEdgeCasesTest, HandlesAllZeros) {
  std::vector<uint32_t> all_zeros(1024, 0);
  verify_sort_with_data(all_zeros);
}

TEST_F(VulkanSortEdgeCasesTest, HandlesAllSameValue) {
  std::vector<uint32_t> all_same(1024, 42);
  verify_sort_with_data(all_same);
}

TEST_F(VulkanSortEdgeCasesTest, HandlesAlreadySorted) {
  std::vector<uint32_t> sorted(1024);
  std::iota(sorted.begin(), sorted.end(), 0);
  verify_sort_with_data(sorted);
}

TEST_F(VulkanSortEdgeCasesTest, HandlesReverseSorted) {
  std::vector<uint32_t> reverse_sorted(1024);
  std::iota(reverse_sorted.rbegin(), reverse_sorted.rend(), 0);
  verify_sort_with_data(reverse_sorted);
}

TEST_F(VulkanSortEdgeCasesTest, HandlesAlternatingValues) {
  std::vector<uint32_t> alternating(1024);
  for (size_t i = 0; i < alternating.size(); i++) {
    alternating[i] = (i % 2) ? 1 : 0;
  }
  verify_sort_with_data(alternating);
}

// TEST_F(VulkanSortEdgeCasesTest, HandlesMaxValues) {
//   std::vector<uint32_t> with_max_values{
//       0, UINT32_MAX, 1, UINT32_MAX - 1, UINT32_MAX / 2, UINT32_MAX};
//   verify_sort_with_data(with_max_values);
// }

// // Add stress test for memory patterns
// TEST_F(VulkanSortEdgeCasesTest, StressTestWithBitPatterns) {
//   std::vector<uint32_t> data(1024);

//   // Test various bit patterns
//   std::vector<uint32_t> patterns = {
//       0xAAAAAAAA,  // Alternating bits
//       0x55555555,  // Alternating bits (inverse)
//       0xFF00FF00,  // Alternating bytes
//       0x00FF00FF,  // Alternating bytes (inverse)
//       0xFFFF0000,  // Half and half
//       0x0000FFFF   // Half and half (inverse)
//   };

//   for (auto pattern : patterns) {
//     std::fill(data.begin(), data.end(), pattern);
//     // Add some random variations
//     std::mt19937 rng(42);
//     for (size_t i = 0; i < data.size(); i += 4) {
//       data[i] ^= rng() & 0xFF;
//     }
//     verify_sort_with_data(data);
//   }
// }

// Main function for running tests
int main(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  spdlog::set_level(spdlog::level::off);

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    spdlog::error("Device ID is required");
    return 1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
