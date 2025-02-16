
#include "../../common/vulkan/engine.hpp"
#include "../sparse_appdata.hpp"

namespace cifar_sparse {

namespace vulkan {

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  ::vulkan::VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

  void sync() { seq->sync(); }

  void process_stage_1(cifar_sparse::AppData &app_data);
  void process_stage_2(cifar_sparse::AppData &app_data);
  void process_stage_3(cifar_sparse::AppData &app_data);
  void process_stage_4(cifar_sparse::AppData &app_data);
  void process_stage_5(cifar_sparse::AppData &app_data);
  void process_stage_6(cifar_sparse::AppData &app_data);
  void process_stage_7(cifar_sparse::AppData &app_data);
  void process_stage_8(cifar_sparse::AppData &app_data);
  void process_stage_9(cifar_sparse::AppData &app_data);

  template <int stage>
  void run_stage(cifar_sparse::AppData &app_data) {
    if constexpr (stage == 1) {
      process_stage_1(app_data);
    } else if constexpr (stage == 2) {
      process_stage_2(app_data);
    } else if constexpr (stage == 3) {
      process_stage_3(app_data);
    } else if constexpr (stage == 4) {
      process_stage_4(app_data);
    } else if constexpr (stage == 5) {
      process_stage_5(app_data);
    } else if constexpr (stage == 6) {
      process_stage_6(app_data);
    } else if constexpr (stage == 7) {
      process_stage_7(app_data);
    } else if constexpr (stage == 8) {
      process_stage_8(app_data);
    } else if constexpr (stage == 9) {
      process_stage_9(app_data);
    }
  }

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  ::vulkan::Engine engine;
  std::shared_ptr<::vulkan::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<::vulkan::Algorithm>> algorithms;

  struct Conv2dPushConstants {
    uint32_t input_height;
    uint32_t input_width;
    uint32_t weight_output_channels;
    uint32_t weight_input_channels;
    uint32_t weight_height;
    uint32_t weight_width;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    bool relu;
  };

  struct MaxpoolPushConstants {
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t pool_size;
    uint32_t stride;
  };

  struct LinearPushConstants {
    uint32_t weight_matrix_rows;
    uint32_t weight_matrix_cols;
  };
};

}  // namespace vulkan

}  // namespace cifar_sparse