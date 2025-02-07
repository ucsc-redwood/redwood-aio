#include "../../common/vulkan/engine.hpp"
#include "../tree_appdata.hpp"
#include "tmp_storage.hpp"

namespace tree {

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

  ::vulkan::VulkanMemoryResource::memory_resource *get_mr() {
    return engine.get_mr();
  }

  void sync() { seq->sync(); }

  void process_stage_1(tree::AppData &app_data_ref);
  void process_stage_2(tree::AppData &app_data_ref);
  void process_stage_3(tree::AppData &app_data_ref, TmpStorage &tmp_storage);
  void process_stage_4(tree::AppData &app_data_ref);
  void process_stage_5(tree::AppData &app_data_ref);
  void process_stage_6(tree::AppData &app_data_ref);
  void process_stage_7(tree::AppData &app_data_ref);

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  ::vulkan::Engine engine;
  std::shared_ptr<::vulkan::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<::vulkan::Algorithm>>
      cached_algorithms;

  // --------------------------------------------------------------------------
  // Temporary storages
  // --------------------------------------------------------------------------

  // (n + 255) / 256;
  UsmVector<uint32_t> tmp_u_sums;
  UsmVector<uint32_t> tmp_u_prefix_sums;

  struct LocalPushConstants {
    uint32_t n_elements;
  };

  struct GlobalPushConstants {
    uint32_t n_blocks;
  };

  // uint32_t warp_size;

  // --------------------------------------------------------------------------
  // Stage 1
  // --------------------------------------------------------------------------

  struct MortonPushConstants {
    uint32_t n;
    float min_coord;
    float range;
  };

  // --------------------------------------------------------------------------
  // Stage 2 - 6
  // --------------------------------------------------------------------------

  struct InputSizePushConstantsUnsigned {
    uint32_t n;
  };

  struct InputSizePushConstantsSigned {
    int32_t n;
  };

  // --------------------------------------------------------------------------
  // Stage 7
  // --------------------------------------------------------------------------

  struct OctreePushConstants {
    float min_coord;
    float range;
    int32_t n_brt_nodes;
  };
};

}  // namespace vulkan

}  // namespace tree
