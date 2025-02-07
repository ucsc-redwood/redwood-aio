
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>

#include "base_appdata.hpp"
#include "common/vulkan/engine.hpp"

struct PushConstants {
  uint32_t g_num_elements;
};

int main(int argc, char** argv) {
  std::string device_id;
  uint32_t g_num_elements;

  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.add_option("-n,--num-elements", g_num_elements, "Number of elements")
      ->default_val(640 * 480);
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    throw std::runtime_error("Device ID is required");
  }

  printf("Device ID: %s\n", device_id.c_str());

  std::cout << "Number of elements: " << g_num_elements << "\n";

  auto engine = vulkan::Engine();

  auto mr = engine.get_mr();

  UsmVector<uint32_t> u_elements_in(g_num_elements, mr);
  UsmVector<uint32_t> u_elements_out(g_num_elements, mr);

  std::ranges::fill(u_elements_in, 1);
  std::ranges::fill(u_elements_out, 666);

  std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                       u_elements_in.end());

  // Peek at first 10 elements before sorting
  std::cout << "First 10 elements before prefix sum:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_in[i] << " ";
  }
  std::cout << "\n";

  // layout(push_constant) uniform PushConstants {
  //     uint num_elements;
  // } pc;

  // layout(set = 0, binding = 0) buffer DataIn {
  //     uint data_in[];
  // };

  // layout(set = 0, binding = 1) buffer DataOut {
  //     uint data_out[];
  // };

  uint32_t n_blocks = (g_num_elements + 255) / 256;

  UsmVector<uint32_t> u_block_sums(n_blocks, mr);

  auto algo1 = engine
                   .algorithm("tmp_prefix_sum.comp",
                              {
                                  engine.get_buffer(u_elements_in.data()),
                                  engine.get_buffer(u_elements_out.data()),
                                  engine.get_buffer(u_block_sums.data()),
                              })
                   ->set_push_constants<PushConstants>({
                       .g_num_elements = g_num_elements,
                   })
                   ->build();

  auto algo2 = engine
                   .algorithm("tmp_prefix_sum_second_pass.comp",
                              {engine.get_buffer(u_elements_out.data()),
                               engine.get_buffer(u_block_sums.data())})
                   ->set_push_constants<PushConstants>({
                       .g_num_elements = g_num_elements,
                   })
                   ->build();

  auto seq = engine.sequence();

  auto start = std::chrono::high_resolution_clock::now();

  constexpr auto n_iterations = 1;

  for (auto i = 0u; i < n_iterations; i++) {
    seq->record_commands_with_blocks(algo1.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();
    seq->record_commands_with_blocks(algo2.get(), n_blocks);
    seq->launch_kernel_async();
    seq->sync();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "Average time per iteration: "
            << (duration.count() / static_cast<float>(n_iterations))
            << " milliseconds" << std::endl;

  // Peek at first 10 elements after sorting
  std::cout << "First 10 elements after prefix sum:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_out[i] << " ";
  }
  std::cout << "\n";

  // std::inclusive_scan(
  //     h_cpu_elements.begin(), h_cpu_elements.end(), h_cpu_elements.begin());

  std::partial_sum(
      h_cpu_elements.begin(), h_cpu_elements.end(), h_cpu_elements.begin());

  std::cout << "First 10 elements after CPU prefix sum:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << h_cpu_elements[i] << " ";
  }
  std::cout << "\n";

  bool is_equal = std::ranges::equal(h_cpu_elements, u_elements_out);
  std::cout << "Matches CPU sort: " << (is_equal ? "true" : "false") << "\n";

  if (!is_equal) {
    std::cout << "\nMismatches found (showing up to 100):\n";
    size_t mismatch_count = 0;
    for (size_t i = 0; i < g_num_elements && mismatch_count < 100; i++) {
      if (h_cpu_elements[i] != u_elements_out[i]) {
        std::cout << "Index " << i << ": CPU=" << h_cpu_elements[i]
                  << ", GPU=" << u_elements_out[i] << "\n";
        mismatch_count++;
      }
    }
    if (mismatch_count == 100) {
      std::cout << "... more mismatches exist (showing first 100 only)\n";
    }
  }

  return 0;
}