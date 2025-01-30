#include "vk_dispatcher.hpp"

namespace cifar_dense {

namespace vulkan {

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.sequence()) {
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
                      app_data.u_linear_weights.data()),             // weights
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

void Singleton::process_stage_1(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv1OutChannels *
                               cifar_dense::kConv1OutHeight *
                               cifar_dense::kConv1OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv1_out.data()),
      engine.get_buffer(app_data.u_conv2_weights.data()),
      engine.get_buffer(app_data.u_conv2_bias.data()),
      engine.get_buffer(app_data.u_conv2_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
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
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_2(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv1OutChannels *
                               cifar_dense::kPool1OutHeight *
                               cifar_dense::kPool1OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv1_out.data()),
      engine.get_buffer(app_data.u_pool1_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv1OutChannels,
      .input_height = cifar_dense::kConv1OutHeight,
      .input_width = cifar_dense::kConv1OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool1OutHeight,
      .output_width = cifar_dense::kPool1OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_3(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv2OutChannels *
                               cifar_dense::kConv2OutHeight *
                               cifar_dense::kConv2OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_pool1_out.data()),
      engine.get_buffer(app_data.u_conv2_weights.data()),
      engine.get_buffer(app_data.u_conv2_bias.data()),
      engine.get_buffer(app_data.u_conv2_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = cifar_dense::kPool1OutHeight,
      .input_width = cifar_dense::kPool1OutWidth,
      .weight_output_channels = cifar_dense::kConv2OutChannels,
      .weight_input_channels = cifar_dense::kConv1OutChannels,
      .weight_height = cifar_dense::kKernelSize,
      .weight_width = cifar_dense::kKernelSize,
      .bias_number_of_elements = cifar_dense::kConv2BiasSize,
      .kernel_size = cifar_dense::kKernelSize,
      .stride = cifar_dense::kStride,
      .padding = cifar_dense::kPadding,
      .output_height = cifar_dense::kConv2OutHeight,
      .output_width = cifar_dense::kConv2OutWidth,
      .relu = cifar_dense::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_4(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv2OutChannels *
                               cifar_dense::kPool2OutHeight *
                               cifar_dense::kPool2OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv2_out.data()),
      engine.get_buffer(app_data.u_pool2_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv2OutChannels,
      .input_height = cifar_dense::kConv2OutHeight,
      .input_width = cifar_dense::kConv2OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool2OutHeight,
      .output_width = cifar_dense::kPool2OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_5(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv3OutChannels *
                               cifar_dense::kConv3OutHeight *
                               cifar_dense::kConv3OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_pool2_out.data()),
      engine.get_buffer(app_data.u_conv3_weights.data()),
      engine.get_buffer(app_data.u_conv3_bias.data()),
      engine.get_buffer(app_data.u_conv3_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = cifar_dense::kPool2OutHeight,
      .input_width = cifar_dense::kPool2OutWidth,
      .weight_output_channels = cifar_dense::kConv3OutChannels,
      .weight_input_channels = cifar_dense::kConv2OutChannels,
      .weight_height = cifar_dense::kKernelSize,
      .weight_width = cifar_dense::kKernelSize,
      .bias_number_of_elements = cifar_dense::kConv3BiasSize,
      .kernel_size = cifar_dense::kKernelSize,
      .stride = cifar_dense::kStride,
      .padding = cifar_dense::kPadding,
      .output_height = cifar_dense::kConv3OutHeight,
      .output_width = cifar_dense::kConv3OutWidth,
      .relu = cifar_dense::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_6(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv4OutChannels *
                               cifar_dense::kConv4OutHeight *
                               cifar_dense::kConv4OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv3_out.data()),
      engine.get_buffer(app_data.u_conv4_weights.data()),
      engine.get_buffer(app_data.u_conv4_bias.data()),
      engine.get_buffer(app_data.u_conv4_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = cifar_dense::kConv3OutHeight,
      .input_width = cifar_dense::kConv3OutWidth,
      .weight_output_channels = cifar_dense::kConv4OutChannels,
      .weight_input_channels = cifar_dense::kConv3OutChannels,
      .weight_height = cifar_dense::kKernelSize,
      .weight_width = cifar_dense::kKernelSize,
      .bias_number_of_elements = cifar_dense::kConv4BiasSize,
      .kernel_size = cifar_dense::kKernelSize,
      .stride = cifar_dense::kStride,
      .padding = cifar_dense::kPadding,
      .output_height = cifar_dense::kConv4OutHeight,
      .output_width = cifar_dense::kConv4OutWidth,
      .relu = cifar_dense::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_7(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv5OutChannels *
                               cifar_dense::kConv5OutHeight *
                               cifar_dense::kConv5OutWidth;

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv4_out.data()),
      engine.get_buffer(app_data.u_conv5_weights.data()),
      engine.get_buffer(app_data.u_conv5_bias.data()),
      engine.get_buffer(app_data.u_conv5_out.data()),
  });

  algo->update_push_constants(Conv2dPushConstants{
      .input_height = cifar_dense::kConv4OutHeight,
      .input_width = cifar_dense::kConv4OutWidth,
      .weight_output_channels = cifar_dense::kConv5OutChannels,
      .weight_input_channels = cifar_dense::kConv4OutChannels,
      .weight_height = cifar_dense::kKernelSize,
      .weight_width = cifar_dense::kKernelSize,
      .bias_number_of_elements = cifar_dense::kConv5BiasSize,
      .kernel_size = cifar_dense::kKernelSize,
      .stride = cifar_dense::kStride,
      .padding = cifar_dense::kPadding,
      .output_height = cifar_dense::kConv5OutHeight,
      .output_width = cifar_dense::kConv5OutWidth,
      .relu = cifar_dense::kRelu,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_8(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kConv5OutChannels *
                               cifar_dense::kPool3OutHeight *
                               cifar_dense::kPool3OutWidth;

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_conv5_out.data()),
      engine.get_buffer(app_data.u_pool3_out.data()),
  });

  algo->update_push_constants(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv5OutChannels,
      .input_height = cifar_dense::kConv5OutHeight,
      .input_width = cifar_dense::kConv5OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool3OutHeight,
      .output_width = cifar_dense::kPool3OutWidth,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

void Singleton::process_stage_9(cifar_dense::AppData &app_data) {
  const int total_iterations = cifar_dense::kLinearOutFeatures;

  auto algo = algorithms.at("linear").get();

  algo->update_descriptor_sets({
      engine.get_buffer(app_data.u_pool3_out.data()),
      engine.get_buffer(app_data.u_linear_weights.data()),
      engine.get_buffer(app_data.u_linear_bias.data()),
      engine.get_buffer(app_data.u_linear_out.data()),
  });

  algo->update_push_constants(LinearPushConstants{
      .in_features = cifar_dense::kLinearInFeatures,
      .out_features = cifar_dense::kLinearOutFeatures,
  });

  seq->record_commands(algo, total_iterations);

  seq->launch_kernel_async();

  // tmp
  seq->sync();
}

}  // namespace vulkan

}  // namespace cifar_dense
