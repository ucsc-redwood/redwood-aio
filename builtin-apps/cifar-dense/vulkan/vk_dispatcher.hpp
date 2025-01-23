#pragma once

#include "../../common/vulkan/engine.hpp"
#include "../dense_appdata.hpp"

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

  void process_stage_1(cifar_dense::AppData &app_data);
  void process_stage_2(cifar_dense::AppData &app_data);
  void process_stage_3(cifar_dense::AppData &app_data);
  void process_stage_4(cifar_dense::AppData &app_data);
  void process_stage_5(cifar_dense::AppData &app_data);
  void process_stage_6(cifar_dense::AppData &app_data);
  void process_stage_7(cifar_dense::AppData &app_data);
  void process_stage_8(cifar_dense::AppData &app_data);
  void process_stage_9(cifar_dense::AppData &app_data);

 private:
  Singleton() : engine(Engine()), seq(engine.sequence()) {
    spdlog::info("Singleton instance created.");

    // tmp
    cifar_dense::AppData app_data(engine.get_mr());

    auto conv2d_algo =
        engine
            .algorithm("cifar_conv2d.comp",
                       {
                           // We still need the buffer here, because we need to
                           // know the size to setup the vk::Pipeline. While the
                           // values here does not matter yet.
                           engine.get_buffer(app_data.u_conv1_out.data()),
                           engine.get_buffer(app_data.u_conv2_weights.data()),
                           engine.get_buffer(app_data.u_conv2_bias.data()),
                           engine.get_buffer(app_data.u_conv2_out.data()),
                       })
            ->set_push_constants<Conv2dPushConstants>({
                // Similarly here, we need to know how many elements we have in
                .input_height = cifar_dense::kInputHeight,
                .input_width = cifar_dense::kInputWidth,
                .weight_output_channels = cifar_dense::kConv1OutChannels,
                .weight_input_channels = cifar_dense::kInputChannels,
                .weight_height = cifar_dense::kKernelSize,
                .weight_width = cifar_dense::kKernelSize,
                .bias_number_of_elements = cifar_dense::kConv1BiasSize,
                .kernel_size = cifar_dense::kKernelSize,
                .stride = cifar_dense::kStride,
                .padding = cifar_dense::kPadding,
                .output_height = cifar_dense::kConv1OutHeight,
                .output_width = cifar_dense::kConv1OutWidth,
                .relu = cifar_dense::kRelu,
            })
            ->build();

    algorithms.try_emplace("conv2d", std::move(conv2d_algo));

    auto maxpool2d_algo =
        engine
            .algorithm(
                "cifar_maxpool2d.comp",
                {
                    engine.get_buffer(app_data.u_conv1_out.data()),  // input
                    engine.get_buffer(app_data.u_pool1_out.data()),  // output
                })
            ->set_push_constants<MaxpoolPushConstants>({
                .input_channels = cifar_dense::kConv1OutChannels,
                .input_height = cifar_dense::kConv1OutHeight,
                .input_width = cifar_dense::kConv1OutWidth,
                .pool_size = cifar_dense::kPoolSize,
                .stride = cifar_dense::kPoolStride,
                .output_height = cifar_dense::kPool1OutHeight,
                .output_width = cifar_dense::kPool1OutWidth,
            })
            ->build();

    algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

    auto linear_algo =
        engine
            .algorithm(
                "cifar_linear.comp",
                {
                    engine.get_buffer(app_data.u_pool3_out.data()),  // input
                    engine.get_buffer(
                        app_data.u_linear_weights.data()),  // weights
                    engine.get_buffer(app_data.u_linear_bias.data()),  // bias
                    engine.get_buffer(app_data.u_linear_out.data()),   // output
                })
            ->set_push_constants<LinearPushConstants>({
                .in_features = cifar_dense::kLinearInFeatures,
                .out_features = cifar_dense::kLinearOutFeatures,
            })
            ->build();

    algorithms.try_emplace("linear", std::move(linear_algo));
  }

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
    uint32_t bias_number_of_elements;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t output_height;
    uint32_t output_width;
    bool relu;
  };

  struct MaxpoolPushConstants {
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t pool_size;
    uint32_t stride;
    uint32_t output_height;
    uint32_t output_width;
  };

  struct LinearPushConstants {
    uint32_t in_features;
    uint32_t out_features;
  };
};

struct Dispatcher {
  explicit Dispatcher(Engine &engine, cifar_dense::AppData &app_data);

  void run_stage1(Sequence *seq);
  void run_stage2(Sequence *seq);
  void run_stage3(Sequence *seq);
  void run_stage4(Sequence *seq);
  void run_stage5(Sequence *seq);
  void run_stage6(Sequence *seq);
  void run_stage7(Sequence *seq);
  void run_stage8(Sequence *seq);
  void run_stage9(Sequence *seq);

  Engine &engine_ref;
  cifar_dense::AppData &app_data_ref;
  std::unordered_map<std::string, std::shared_ptr<Algorithm>> algorithms;

  struct Conv2dPushConstants {
    uint32_t input_height;
    uint32_t input_width;
    uint32_t weight_output_channels;
    uint32_t weight_input_channels;
    uint32_t weight_height;
    uint32_t weight_width;
    uint32_t bias_number_of_elements;
    uint32_t kernel_size;
    uint32_t stride;
    uint32_t padding;
    uint32_t output_height;
    uint32_t output_width;
    bool relu;
  };

  struct MaxpoolPushConstants {
    uint32_t input_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t pool_size;
    uint32_t stride;
    uint32_t output_height;
    uint32_t output_width;
  };

  struct LinearPushConstants {
    uint32_t in_features;
    uint32_t out_features;
  };
};

}  // namespace vulkan
