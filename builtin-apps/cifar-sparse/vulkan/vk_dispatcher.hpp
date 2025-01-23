
#include "../../common/vulkan/engine.hpp"
#include "../sparse_appdata.hpp"

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

  VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

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

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  Engine engine;
  std::shared_ptr<Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<Algorithm>> algorithms;

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
