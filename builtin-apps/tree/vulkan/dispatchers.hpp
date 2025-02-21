#include "../../common/kiss-vk/engine.hpp"
#include "../tree_appdata.hpp"
#include "tmp_storage.hpp"

namespace tree::vulkan {

class Singleton {
 public:
  // Delete copy constructor and assignment operator to prevent copies
  Singleton(const Singleton &) = delete;
  Singleton &operator=(const Singleton &) = delete;

  static Singleton &getInstance() {
    static Singleton instance;
    return instance;
  }

  kiss_vk::VulkanMemoryResource::memory_resource *get_mr() { return engine.get_mr(); }

  void process_stage_1(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_2(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_3(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_4(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_5(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_6(AppData &appdata, TmpStorage &tmp_storage);
  void process_stage_7(AppData &appdata, TmpStorage &tmp_storage);

  template <int Stage>
    requires(Stage >= 1 && Stage <= 7)
  void run_stage(AppData &appdata, TmpStorage &tmp_storage) {
    if constexpr (Stage == 1) {
      process_stage_1(appdata, tmp_storage);
    } else if constexpr (Stage == 2) {
      process_stage_2(appdata, tmp_storage);
    } else if constexpr (Stage == 3) {
      process_stage_3(appdata, tmp_storage);
    } else if constexpr (Stage == 4) {
      process_stage_4(appdata, tmp_storage);
    } else if constexpr (Stage == 5) {
      process_stage_5(appdata, tmp_storage);
    } else if constexpr (Stage == 6) {
      process_stage_6(appdata, tmp_storage);
    } else if constexpr (Stage == 7) {
      process_stage_7(appdata, tmp_storage);
    }
  }

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  kiss_vk::Engine engine;
  std::shared_ptr<kiss_vk::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<kiss_vk::Algorithm>> cached_algorithms;

  // --------------------------------------------------------------------------
  // Temporary storages
  // --------------------------------------------------------------------------

  // (n + 255) / 256;
  UsmVector<uint32_t> tmp_u_sums;
  UsmVector<uint32_t> tmp_u_prefix_sums;
};

}  // namespace tree::vulkan
