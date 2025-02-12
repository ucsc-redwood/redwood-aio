#include <random>

#include "builtin-apps/app.hpp"
#include "builtin-apps/base_appdata.hpp"
#include "builtin-apps/common/vulkan/engine.hpp"

[[nodiscard]] static std::string get_shader_name() {
  if (g_device_id == "3A021JEHN02756") {
    return "tmp_single_radixsort_warp16";
  } else if (g_device_id == "9b034f1b") {
    return "tmp_single_radixsort_warp64";
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    return "tmp_single_radixsort_warp32";
  } else if (g_device_id == "amd-minipc") {
    return "tmp_single_radixsort_warp64";
  } else if (g_device_id == "pc" || g_device_id == "jetson") {
    return "tmp_single_radixsort_warp32";
  }
  throw std::runtime_error("Invalid device ID");
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  constexpr auto n = 1024;

  vulkan::Engine engine;

  UsmVector<uint32_t> u_elements_in(n, engine.get_mr());
  UsmVector<uint32_t> u_elements_out(n, engine.get_mr());

  std::iota(u_elements_in.begin(), u_elements_in.end(), 0);
  std::mt19937 rng(42);
  std::ranges::shuffle(u_elements_in, rng);

  for (uint32_t i = 0; i < 10; ++i) {
    std::cout << "in [" << i << "] " << u_elements_in[i] << std::endl;
  }

  struct PushConstants {
    uint32_t g_num_elements;
  };

  auto algo = engine.make_algo(get_shader_name())
                  ->work_group_size(256, 1, 1)
                  ->num_sets(1)
                  ->num_buffers(2)
                  ->push_constant<PushConstants>()
                  ->build();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(u_elements_in),   // input
                                  engine.get_buffer_info(u_elements_out),  // output
                              });
  algo->update_push_constant(PushConstants{n});

  auto seq = engine.make_seq();

  seq->cmd_begin();

  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(), {1, 1, 1});

  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();

  for (uint32_t i = 0; i < n; ++i) {
    std::cout << "out [" << i << "] " << u_elements_out[i] << std::endl;
  }

  return 0;
}