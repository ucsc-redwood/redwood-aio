#include <gtest/gtest.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdint>
#include <numeric>

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

struct LocalPushConstants {
  uint32_t g_num_elements;
};

struct GlobalPushConstants {
  uint32_t g_num_blocks;
};

// ----------------------------------------------------------------------------
// Test Fixtures and Helpers
// ----------------------------------------------------------------------------

class VulkanTestFixture : public ::testing::Test {
 protected:
  vulkan::Engine &engine = Singleton::getInstance().get_engine();
};

class VulkanPrefixSumTest : public VulkanTestFixture,
                            public testing::WithParamInterface<unsigned int> {
 protected:
  void verify_prefix_sum(unsigned int n) {
    const auto n_blocks = (n + 255) / 256;

    auto mr = engine.get_mr();

    UsmVector<uint32_t> u_elements_in(n, mr);
    UsmVector<uint32_t> u_elements_out(n, mr);
    UsmVector<uint32_t> u_sums(n_blocks, mr);
    UsmVector<uint32_t> u_prefix_sums(n_blocks, mr);

    std::ranges::fill(u_elements_in, 1);

    auto local_inclusive_scan = engine
                                    .algorithm("tmp_local_inclusive_scan.comp",
                                               {
                                                   engine.get_buffer(u_elements_in.data()),
                                                   engine.get_buffer(u_elements_out.data()),
                                                   engine.get_buffer(u_sums.data()),
                                               })
                                    ->set_push_constants<LocalPushConstants>({
                                        .g_num_elements = n,
                                    })
                                    ->build();

    auto global_exclusive_scan = engine
                                     .algorithm("tmp_global_exclusive_scan.comp",
                                                {
                                                    engine.get_buffer(u_sums.data()),
                                                    engine.get_buffer(u_prefix_sums.data()),
                                                })
                                     ->set_push_constants<GlobalPushConstants>({
                                         .g_num_blocks = n_blocks,
                                     })
                                     ->build();

    auto add_base = engine
                        .algorithm("tmp_add_base.comp",
                                   {
                                       engine.get_buffer(u_elements_out.data()),
                                       engine.get_buffer(u_prefix_sums.data()),
                                   })
                        ->set_push_constants<LocalPushConstants>({
                            .g_num_elements = n,
                        })
                        ->build();

    auto seq = engine.sequence();

    seq->record_commands_with_blocks(local_inclusive_scan.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(global_exclusive_scan.get(), 1);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(add_base.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();

    // // Print first 10 elements of output
    // std::cout << "First 10 output elements: ";
    // for (size_t i = 0; i < std::min(size_t(10), u_elements_out.size()); ++i)
    // {
    //   std::cout << u_elements_out[i] << " ";
    // }
    // std::cout << std::endl;

    std::vector<uint32_t> h_cpu_elements(n, 1);

    // Verify results
    std::vector<uint32_t> h_cpu_prefix_sums(n);
    std::partial_sum(h_cpu_elements.begin(), h_cpu_elements.end(), h_cpu_prefix_sums.begin());

    // std::cout << "First 10 CPU prefix sums: ";
    // for (size_t i = 0; i < std::min(size_t(10), h_cpu_prefix_sums.size());
    //      ++i) {
    //   std::cout << h_cpu_prefix_sums[i] << " ";
    // }
    // std::cout << std::endl;

    EXPECT_TRUE(std::ranges::equal(h_cpu_prefix_sums, u_elements_out));
  }
};

TEST_P(VulkanPrefixSumTest, VerifyPrefixSum) { verify_prefix_sum(GetParam()); }

INSTANTIATE_TEST_SUITE_P(VaryingSizes,
                         VulkanPrefixSumTest,
                         testing::Values(1024, 64 * 1024, 640 * 480),
                         [](const testing::TestParamInfo<unsigned int> &info) {
                           return "Size" + std::to_string(info.param);
                         });

class VulkanPrefixSumIterationTest
    : public VulkanTestFixture,
      public testing::WithParamInterface<std::tuple<unsigned int, unsigned int>> {
 protected:
  void verify_prefix_sum_iterations(unsigned int n, unsigned int iterations) {
    const auto n_blocks = (n + 255) / 256;
    auto mr = engine.get_mr();

    UsmVector<uint32_t> u_elements_in(n, mr);
    UsmVector<uint32_t> u_elements_out(n, mr);
    UsmVector<uint32_t> u_sums(n_blocks, mr);
    UsmVector<uint32_t> u_prefix_sums(n_blocks, mr);

    std::ranges::fill(u_elements_in, 1);

    auto local_inclusive_scan = engine
                                    .algorithm("tmp_local_inclusive_scan.comp",
                                               {
                                                   engine.get_buffer(u_elements_in.data()),
                                                   engine.get_buffer(u_elements_out.data()),
                                                   engine.get_buffer(u_sums.data()),
                                               })
                                    ->set_push_constants<LocalPushConstants>({
                                        .g_num_elements = n,
                                    })
                                    ->build();

    auto global_exclusive_scan = engine
                                     .algorithm("tmp_global_exclusive_scan.comp",
                                                {
                                                    engine.get_buffer(u_sums.data()),
                                                    engine.get_buffer(u_prefix_sums.data()),
                                                })
                                     ->set_push_constants<GlobalPushConstants>({
                                         .g_num_blocks = n_blocks,
                                     })
                                     ->build();

    auto add_base = engine
                        .algorithm("tmp_add_base.comp",
                                   {
                                       engine.get_buffer(u_elements_out.data()),
                                       engine.get_buffer(u_prefix_sums.data()),
                                   })
                        ->set_push_constants<LocalPushConstants>({
                            .g_num_elements = n,
                        })
                        ->build();

    auto seq = engine.sequence();

    // Run multiple iterations
    for (unsigned int i = 0; i < iterations; ++i) {
      seq->record_commands_with_blocks(local_inclusive_scan.get(), n_blocks);
      seq->launch_kernel_async();
      seq->sync();

      seq->record_commands_with_blocks(global_exclusive_scan.get(), 1);
      seq->launch_kernel_async();
      seq->sync();

      seq->record_commands_with_blocks(add_base.get(), n_blocks);
      seq->launch_kernel_async();
      seq->sync();
    }

    std::vector<uint32_t> h_cpu_elements(n, 1);
    std::vector<uint32_t> h_cpu_prefix_sums(n);
    std::partial_sum(h_cpu_elements.begin(), h_cpu_elements.end(), h_cpu_prefix_sums.begin());

    EXPECT_TRUE(std::ranges::equal(h_cpu_prefix_sums, u_elements_out));
  }
};

class VulkanPrefixSumEdgeCasesTest : public VulkanTestFixture {
 protected:
  void verify_prefix_sum_with_data(const std::vector<uint32_t> &input_data) {
    const uint32_t n = input_data.size();
    const uint32_t n_blocks = (n + 255) / 256;
    auto mr = engine.get_mr();

    UsmVector<uint32_t> u_elements_in(input_data.begin(), input_data.end(), mr);
    UsmVector<uint32_t> u_elements_out(n, mr);
    UsmVector<uint32_t> u_sums(n_blocks, mr);
    UsmVector<uint32_t> u_prefix_sums(n_blocks, mr);

    auto local_inclusive_scan = engine
                                    .algorithm("tmp_local_inclusive_scan.comp",
                                               {
                                                   engine.get_buffer(u_elements_in.data()),
                                                   engine.get_buffer(u_elements_out.data()),
                                                   engine.get_buffer(u_sums.data()),
                                               })
                                    ->set_push_constants<LocalPushConstants>({
                                        .g_num_elements = static_cast<uint32_t>(n),
                                    })
                                    ->build();

    auto global_exclusive_scan = engine
                                     .algorithm("tmp_global_exclusive_scan.comp",
                                                {
                                                    engine.get_buffer(u_sums.data()),
                                                    engine.get_buffer(u_prefix_sums.data()),
                                                })
                                     ->set_push_constants<GlobalPushConstants>({
                                         .g_num_blocks = n_blocks,
                                     })
                                     ->build();

    auto add_base = engine
                        .algorithm("tmp_add_base.comp",
                                   {
                                       engine.get_buffer(u_elements_out.data()),
                                       engine.get_buffer(u_prefix_sums.data()),
                                   })
                        ->set_push_constants<LocalPushConstants>({
                            .g_num_elements = static_cast<uint32_t>(n),
                        })
                        ->build();

    auto seq = engine.sequence();
    seq->record_commands_with_blocks(local_inclusive_scan.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(global_exclusive_scan.get(), 1);
    seq->launch_kernel_async();
    seq->sync();

    seq->record_commands_with_blocks(add_base.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();

    std::vector<uint32_t> h_cpu_prefix_sums(n);
    std::partial_sum(input_data.begin(), input_data.end(), h_cpu_prefix_sums.begin());

    EXPECT_TRUE(std::ranges::equal(h_cpu_prefix_sums, u_elements_out));
  }
};

// Test with different input sizes and iteration counts
TEST_P(VulkanPrefixSumIterationTest, ComputesCorrectlyWithMultipleIterations) {
  auto [size, iterations] = GetParam();
  verify_prefix_sum_iterations(size, iterations);
}

INSTANTIATE_TEST_SUITE_P(
    VaryingSizesAndIterations,
    VulkanPrefixSumIterationTest,
    testing::Combine(testing::Values(1024, 64 * 1024, 640 * 480),  // Sizes
                     testing::Values(1, 2, 5, 10, 32)              // Iterations
                     ),
    [](const testing::TestParamInfo<std::tuple<unsigned int, unsigned int>> &info) {
      return "Size" + std::to_string(std::get<0>(info.param)) + "_Iterations" +
             std::to_string(std::get<1>(info.param));
    });

// Edge case tests
TEST_F(VulkanPrefixSumEdgeCasesTest, HandlesAllZeros) {
  std::vector<uint32_t> all_zeros(1024, 0);
  verify_prefix_sum_with_data(all_zeros);
}

TEST_F(VulkanPrefixSumEdgeCasesTest, HandlesAllOnes) {
  std::vector<uint32_t> all_ones(1024, 1);
  verify_prefix_sum_with_data(all_ones);
}

TEST_F(VulkanPrefixSumEdgeCasesTest, HandlesAlternatingValues) {
  std::vector<uint32_t> alternating(1024);
  for (size_t i = 0; i < alternating.size(); i++) {
    alternating[i] = (i % 2) ? 1 : 0;
  }
  verify_prefix_sum_with_data(alternating);
}

TEST_F(VulkanPrefixSumEdgeCasesTest, HandlesIncreasingValues) {
  std::vector<uint32_t> increasing(1024);
  std::iota(increasing.begin(), increasing.end(), 0);
  verify_prefix_sum_with_data(increasing);
}

TEST_F(VulkanPrefixSumEdgeCasesTest, HandlesDecreasingValues) {
  std::vector<uint32_t> decreasing(1024);
  std::iota(decreasing.rbegin(), decreasing.rend(), 0);
  verify_prefix_sum_with_data(decreasing);
}

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
