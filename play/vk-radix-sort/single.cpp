
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

#include "../../builtin-apps/base_appdata.hpp"
#include "../../builtin-apps/common/vulkan/engine.hpp"

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
  uint32_t g_num_elements = 640 * 480;
  if (argc > 1) {
    g_num_elements = std::stoul(argv[1]);
  }

  std::cout << "Number of elements: " << g_num_elements << "\n";

  auto engine = vulkan::Engine();

  auto mr = engine.get_mr();

  UsmVector<uint32_t> u_elements_in(g_num_elements, mr);
  UsmVector<uint32_t> u_elements_out(g_num_elements, mr);

  std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
  std::mt19937 rng(42);
  std::shuffle(u_elements_in.begin(), u_elements_in.end(), rng);

  // Peek at first 10 elements before sorting
  std::cout << "First 10 elements before sorting:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_in[i] << " ";
  }
  std::cout << "\n";

  auto algo = engine
                  .algorithm("tmp_single_radixsort_warp32.comp",
                             {
                                 engine.get_buffer(u_elements_in.data()),
                                 engine.get_buffer(u_elements_out.data()),
                             })
                  ->set_push_constants<PushConstants>({
                      .g_num_elements = g_num_elements,
                  })
                  ->build();

  auto seq = engine.sequence();

  seq->record_commands(algo.get(), g_num_elements);
  seq->launch_kernel_async();
  seq->sync();

  // Peek at first 10 elements after sorting
  std::cout << "First 10 elements after sorting:\n";
  for (auto i = 0u; i < std::min(10u, g_num_elements); i++) {
    std::cout << u_elements_out[i] << " ";
  }
  std::cout << "\n";

  bool is_sorted = std::ranges::is_sorted(u_elements_out);
  std::cout << "Is sorted: " << (is_sorted ? "true" : "false") << "\n";

  return 0;
}