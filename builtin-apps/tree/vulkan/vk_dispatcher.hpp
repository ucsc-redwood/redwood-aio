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

  void process_stage_1(tree::AppData &app_data);
  void process_stage_2(tree::AppData &app_data);
  void process_stage_3(tree::AppData &app_data);
  void process_stage_4(tree::AppData &app_data);
  void process_stage_5(tree::AppData &app_data);
  void process_stage_6(tree::AppData &app_data);
  void process_stage_7(tree::AppData &app_data);

 private:
  Singleton();
  ~Singleton() { spdlog::info("Singleton instance destroyed."); }

  ::vulkan::Engine engine;
  std::shared_ptr<::vulkan::Sequence> seq;
  std::unordered_map<std::string, std::shared_ptr<::vulkan::Algorithm>>
      cached_algorithms;

  // ::vulkan::TmpStorage tmp_storage;

  // --------------------------------------------------------------------------
  // Stage 1
  // --------------------------------------------------------------------------

  struct MortonPushConstants {
    uint32_t n;
    float min_coord;
    float range;
  };

  // --------------------------------------------------------------------------
  // Stage 2
  // --------------------------------------------------------------------------

  struct MergeSortPushConstants {
    uint32_t n_logical_blocks;
    uint32_t n;
    uint32_t width;
    uint32_t num_pairs;
  };

  // --------------------------------------------------------------------------
  // Stage 3
  // --------------------------------------------------------------------------

  struct FindDupsPushConstants {
    int32_t n;
  };

  struct MoveDupsPushConstants {
    uint32_t n;
  };

  // --------------------------------------------------------------------------
  // Stage 4
  // --------------------------------------------------------------------------

  struct BuildTreePushConstants {
    int32_t n;
  };

  // --------------------------------------------------------------------------
  // Stage 5
  // --------------------------------------------------------------------------

  struct EdgeCountPushConstants {
    int32_t n_brt_nodes;
  };

  // --------------------------------------------------------------------------
  // Stage 6
  // --------------------------------------------------------------------------

  struct PrefixSumPushConstants {
    uint32_t inputSize;
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
