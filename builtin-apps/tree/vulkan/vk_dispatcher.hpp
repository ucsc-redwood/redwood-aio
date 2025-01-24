#include "../../common/vulkan/engine.hpp"
#include "../tree_appdata.hpp"

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
      algorithms;
};

}  // namespace vulkan

}  // namespace tree
