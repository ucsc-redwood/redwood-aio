#include "vk_dispatcher.hpp"

namespace cifar_dense {

namespace vulkan {

Singleton::Singleton() : engine(::vulkan::Engine()), seq(engine.make_seq()) {
  spdlog::info("Singleton instance created.");

  auto conv2d_algo = engine.make_algo("cifar_conv2d")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(4)
                         ->push_constant<Conv2dPushConstants>()
                         ->build();

  algorithms.try_emplace("conv2d", std::move(conv2d_algo));

  auto maxpool2d_algo = engine.make_algo("cifar_maxpool2d")
                            ->work_group_size(256, 1, 1)
                            ->num_sets(1)
                            ->num_buffers(2)
                            ->push_constant<MaxpoolPushConstants>()
                            ->build();

  algorithms.try_emplace("maxpool2d", std::move(maxpool2d_algo));

  auto linear_algo = engine.make_algo("cifar_linear")
                         ->work_group_size(256, 1, 1)
                         ->num_sets(1)
                         ->num_buffers(4)
                         ->push_constant<LinearPushConstants>()
                         ->build();

  algorithms.try_emplace("linear", std::move(linear_algo));
}

void Singleton::process_stage_1(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv1OutChannels * cifar_dense::kConv1OutHeight * cifar_dense::kConv1OutWidth;

  spdlog::trace("[VK] process_stage_1, (conv2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_image),
                                  engine.get_buffer_info(app_data.u_conv1_weights),
                                  engine.get_buffer_info(app_data.u_conv1_bias),
                                  engine.get_buffer_info(app_data.u_conv1_out),
                              });

  algo->update_push_constant(Conv2dPushConstants{
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_2(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv1OutChannels * cifar_dense::kPool1OutHeight * cifar_dense::kPool1OutWidth;

  spdlog::trace("[VK] process_stage_2, (maxpool2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv1_out),
                                  engine.get_buffer_info(app_data.u_pool1_out),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv1OutChannels,
      .input_height = cifar_dense::kConv1OutHeight,
      .input_width = cifar_dense::kConv1OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool1OutHeight,
      .output_width = cifar_dense::kPool1OutWidth,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_3(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv2OutChannels * cifar_dense::kConv2OutHeight * cifar_dense::kConv2OutWidth;

  spdlog::trace("[VK] process_stage_3, (conv2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool1_out),
                                  engine.get_buffer_info(app_data.u_conv2_weights),
                                  engine.get_buffer_info(app_data.u_conv2_bias),
                                  engine.get_buffer_info(app_data.u_conv2_out),
                              });

  algo->update_push_constant(Conv2dPushConstants{
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_4(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv2OutChannels * cifar_dense::kPool2OutHeight * cifar_dense::kPool2OutWidth;

  spdlog::trace("[VK] process_stage_4, (maxpool2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv2_out),
                                  engine.get_buffer_info(app_data.u_pool2_out),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv2OutChannels,
      .input_height = cifar_dense::kConv2OutHeight,
      .input_width = cifar_dense::kConv2OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool2OutHeight,
      .output_width = cifar_dense::kPool2OutWidth,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_5(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv3OutChannels * cifar_dense::kConv3OutHeight * cifar_dense::kConv3OutWidth;

  spdlog::trace("[VK] process_stage_5, (conv2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool2_out),
                                  engine.get_buffer_info(app_data.u_conv3_weights),
                                  engine.get_buffer_info(app_data.u_conv3_bias),
                                  engine.get_buffer_info(app_data.u_conv3_out),
                              });

  algo->update_push_constant(Conv2dPushConstants{
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_6(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv4OutChannels * cifar_dense::kConv4OutHeight * cifar_dense::kConv4OutWidth;

  spdlog::trace("[VK] process_stage_6, (conv2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv3_out),
                                  engine.get_buffer_info(app_data.u_conv4_weights),
                                  engine.get_buffer_info(app_data.u_conv4_bias),
                                  engine.get_buffer_info(app_data.u_conv4_out),
                              });

  algo->update_push_constant(Conv2dPushConstants{
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_7(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv5OutChannels * cifar_dense::kConv5OutHeight * cifar_dense::kConv5OutWidth;

  spdlog::trace("[VK] process_stage_7, (conv2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("conv2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv4_out),
                                  engine.get_buffer_info(app_data.u_conv5_weights),
                                  engine.get_buffer_info(app_data.u_conv5_bias),
                                  engine.get_buffer_info(app_data.u_conv5_out),
                              });

  algo->update_push_constant(Conv2dPushConstants{
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

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_8(cifar_dense::AppData &app_data) {
  const int total_iterations =
      cifar_dense::kConv5OutChannels * cifar_dense::kPool3OutHeight * cifar_dense::kPool3OutWidth;

  spdlog::trace("[VK] process_stage_8, (maxpool2d), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("maxpool2d").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_conv5_out),
                                  engine.get_buffer_info(app_data.u_pool3_out),
                              });

  algo->update_push_constant(MaxpoolPushConstants{
      .input_channels = cifar_dense::kConv5OutChannels,
      .input_height = cifar_dense::kConv5OutHeight,
      .input_width = cifar_dense::kConv5OutWidth,
      .pool_size = cifar_dense::kPoolSize,
      .stride = cifar_dense::kPoolStride,
      .output_height = cifar_dense::kPool3OutHeight,
      .output_width = cifar_dense::kPool3OutWidth,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

void Singleton::process_stage_9(cifar_dense::AppData &app_data) {
  constexpr int total_iterations = cifar_dense::kLinearOutFeatures;  // 10

  spdlog::trace("[VK] process_stage_9, (linear), total_iterations: {}", total_iterations);

  auto algo = algorithms.at("linear").get();

  algo->update_descriptor_set(0,
                              {
                                  engine.get_buffer_info(app_data.u_pool3_out),
                                  engine.get_buffer_info(app_data.u_linear_weights),
                                  engine.get_buffer_info(app_data.u_linear_bias),
                                  engine.get_buffer_info(app_data.u_linear_out),
                              });

  algo->update_push_constant(LinearPushConstants{
      .in_features = cifar_dense::kLinearInFeatures,
      .out_features = cifar_dense::kLinearOutFeatures,
  });

  seq->cmd_begin();
  algo->record_bind_core(seq->get_handle(), 0);
  algo->record_bind_push(seq->get_handle());
  algo->record_dispatch(seq->get_handle(),
                        {static_cast<uint32_t>(::vulkan::div_ceil(total_iterations, 256)), 1, 1});
  seq->cmd_end();

  seq->launch_kernel_async();
  seq->sync();
}

}  // namespace vulkan

}  // namespace cifar_dense
