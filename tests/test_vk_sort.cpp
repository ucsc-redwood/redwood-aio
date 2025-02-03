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

// Test sorting a random vector
TEST(SortTest, SortsAscending) {
  std::vector<int> data = {5, 3, 8, 1, 2};
  std::vector<int> expected = {1, 2, 3, 5, 8};

  std::sort(data.begin(), data.end());

  EXPECT_EQ(data, expected);
}

struct PushConstants {
  uint32_t g_num_elements;
};

TEST(VulkanSortTest, SingleRadixSort) {
  constexpr unsigned int n = 640 * 480;

  auto &engine = Singleton::getInstance().get_engine();

  auto mr = engine.get_mr();

  //   auto engine = vulkan::Engine();

  //   auto mr = engine.get_mr();

  UsmVector<uint32_t> u_elements_in(n, mr);
  UsmVector<uint32_t> u_elements_out(n, mr);

  std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

  std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                       u_elements_in.end());

  //   // Peek at first 10 elements before sorting
  //   std::cout << "First 10 elements before sorting:\n";
  //   for (auto i = 0u; i < std::min(10u, n); i++) {
  //     std::cout << u_elements_in[i] << " ";
  //   }
  //   std::cout << "\n";

  std::string shader_name;

  if (device_id == "3A021JEHN02756") {
    shader_name = "tmp_single_radixsort_warp16.comp";
  } else if (device_id == "9b034f1b") {
    shader_name = "tmp_single_radixsort_warp64.comp";
  } else {
    throw std::runtime_error("Invalid device ID");
  }

  //   printf("Shader name: %s\n", shader_name.c_str());

  auto algo = engine
                  .algorithm(shader_name,
                             {
                                 engine.get_buffer(u_elements_in.data()),
                                 engine.get_buffer(u_elements_out.data()),
                             })
                  ->set_push_constants<PushConstants>({
                      .g_num_elements = n,
                  })
                  ->build();

  auto seq = engine.sequence();

  //   constexpr auto n_iterations = 100;

  //   for (auto i = 0u; i < n_iterations; i++) {
  seq->record_commands_with_blocks(algo.get(), 1);
  seq->launch_kernel_async();
  seq->sync();
  //   }
  //   // Peek at first 10 elements after sorting
  //   std::cout << "First 10 elements after sorting:\n";
  //   for (auto i = 0u; i < std::min(10u, n); i++) {
  //     std::cout << u_elements_out[i] << " ";
  //   }
  //   std::cout << "\n";

  EXPECT_TRUE(std::ranges::is_sorted(u_elements_out));

  std::ranges::sort(h_cpu_elements);
  EXPECT_TRUE(std::ranges::equal(h_cpu_elements, u_elements_out));
}

// Main function for running tests
int main(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    spdlog::error("Device ID is required");
    return 1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
