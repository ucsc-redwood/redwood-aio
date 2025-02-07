
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "base_appdata.hpp"
#include "common/vulkan/engine.hpp"
// #include "conf.hpp"

// layout (push_constant, std430) uniform PushConstants {
//     uint g_num_elements;
// };

// layout (std430, set = 0, binding = 0) buffer elements_in {
//     uint g_elements_in[];
// };

// layout (std430, set = 0, binding = 1) buffer elements_out {
//     uint g_elements_out[];
// };

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

  std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

  std::vector<uint32_t> h_cpu_elements(u_elements_in.begin(),
                                       u_elements_in.end());

  // Peek at first 10 elements before sorting
  std::cout << "First 10 elements before sorting:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_in[i] << " ";
  }
  std::cout << "\n";

  std::string shader_name;

  if (device_id == "3A021JEHN02756") {
    shader_name = "tmp_single_radixsort_warp16.comp";
  } else if (device_id == "9b034f1b") {
    shader_name = "tmp_single_radixsort_warp64.comp";
  } else if (device_id == "pc" || device_id == "jetson") {
    shader_name = "tmp_single_radixsort_warp32.comp";
  } else {
    throw std::runtime_error("Invalid device ID");
  }

  printf("Shader name: %s\n", shader_name.c_str());

  auto algo = engine
                  .algorithm(shader_name,
                             {
                                 engine.get_buffer(u_elements_in.data()),
                                 engine.get_buffer(u_elements_out.data()),
                             })
                  ->set_push_constants<PushConstants>({
                      .g_num_elements = g_num_elements,
                  })
                  ->build();

  auto seq = engine.sequence();

  auto start = std::chrono::high_resolution_clock::now();

  constexpr auto n_iterations = 100;

  for (auto i = 0u; i < n_iterations; i++) {
    seq->record_commands_with_blocks(algo.get(), 1);
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
  std::cout << "First 10 elements after sorting:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_out[i] << " ";
  }
  std::cout << "\n";

  bool is_sorted = std::ranges::is_sorted(u_elements_out);
  std::cout << "Is sorted: " << (is_sorted ? "true" : "false") << "\n";

  std::ranges::sort(h_cpu_elements);
  bool is_equal = std::ranges::equal(h_cpu_elements, u_elements_out);
  std::cout << "Matches CPU sort: " << (is_equal ? "true" : "false") << "\n";

  return 0;
}