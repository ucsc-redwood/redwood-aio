// #pragma once

// #include "builtin-apps/affinity.hpp"
// #include "builtin-apps/app.hpp"
// #include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
// #include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
// #include "builtin-apps/common/cuda/manager.cuh"

// // template <int N>
// // concept AllowedStage = (N >= 1 && N <= 9);

// // ---------------------------------------------------------------------
// // CPU stages
// // ---------------------------------------------------------------------

// namespace omp {

// constexpr std::array<void (*)(cifar_dense::AppData &), 9> cpu_stages = {
//     cifar_dense::omp::process_stage_1,
//     cifar_dense::omp::process_stage_2,
//     cifar_dense::omp::process_stage_3,
//     cifar_dense::omp::process_stage_4,
//     cifar_dense::omp::process_stage_5,
//     cifar_dense::omp::process_stage_6,
//     cifar_dense::omp::process_stage_7,
//     cifar_dense::omp::process_stage_8,
//     cifar_dense::omp::process_stage_9,
// };

// void run_stage_impl(int start, int end, cifar_dense::AppData &data) {
//   for (int i = start - 1; i < end; ++i) {
//     cpu_stages[i](data);
//   }
// }

// // template <ProcessorType PT>
// // void run_multiple_stages(const int start_stage,
// //                          const int end_stage,
// //                          cifar_dense::AppData &data,
// //                          [[maybe_unused]] cuda::CudaManager &mgr,
// //                          const int num_threads) {
// // #pragma omp parallel num_threads(num_threads)
// //   {
// //     // Bind to core
// //     if constexpr (PT == ProcessorType::kLittleCore) {
// //       bind_thread_to_cores(g_little_cores);
// //     } else if constexpr (PT == ProcessorType::kMediumCore) {
// //       bind_thread_to_cores(g_medium_cores);
// //     } else if constexpr (PT == ProcessorType::kBigCore) {
// //       bind_thread_to_cores(g_big_cores);
// //     }

// //     run_stage_impl(start_stage, end_stage, data);
// //   }
// // }

// void run_multiple_stages(const int start_stage,
//                          const int end_stage,
//                          ProcessorType pt,
//                          cifar_dense::AppData &data,
//                          [[maybe_unused]] cuda::CudaManager &mgr,
//                          const int num_threads) {
// #pragma omp parallel num_threads(num_threads)
//   {
//     // Bind to core
//     if (pt == ProcessorType::kLittleCore) {
//       bind_thread_to_cores(g_little_cores);
//     } else if (pt == ProcessorType::kMediumCore) {
//       bind_thread_to_cores(g_medium_cores);
//     } else if (pt == ProcessorType::kBigCore) {
//       bind_thread_to_cores(g_big_cores);
//     }

//     run_stage_impl(start_stage, end_stage, data);
//   }
// }

// // template <int Start, int End, ProcessorType PT, int NThreads>
// //   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// // void run_multiple_stages(cifar_dense::AppData &data, cuda::CudaManager &) {
// // #pragma omp parallel num_threads(NThreads)
// //   {
// //     // Bind to core
// //     if constexpr (PT == ProcessorType::kLittleCore) {
// //       bind_thread_to_cores(g_little_cores);
// //     } else if constexpr (PT == ProcessorType::kMediumCore) {
// //       bind_thread_to_cores(g_medium_cores);
// //     } else if constexpr (PT == ProcessorType::kBigCore) {
// //       bind_thread_to_cores(g_big_cores);
// //     }

// //     for (int s = Start; s <= End; ++s) {
// //       cpu_stages[s - 1](data);
// //     }
// //   }
// // }

// }  // namespace omp

// // ---------------------------------------------------------------------
// // GPU stages
// // ---------------------------------------------------------------------

// namespace cuda {

// constexpr std::array<void (*)(cifar_dense::AppData &), 9> gpu_stages = {
//     cifar_dense::cuda::process_stage_1,
//     cifar_dense::cuda::process_stage_2,
//     cifar_dense::cuda::process_stage_3,
//     cifar_dense::cuda::process_stage_4,
//     cifar_dense::cuda::process_stage_5,
//     cifar_dense::cuda::process_stage_6,
//     cifar_dense::cuda::process_stage_7,
//     cifar_dense::cuda::process_stage_8,
//     cifar_dense::cuda::process_stage_9,
// };

// void run_multiple_stages(const int start_stage,
//                          const int end_stage,
//                          cifar_dense::AppData &data,
//                          cuda::CudaManager &mgr) {
//   for (int s = start_stage; s <= end_stage; ++s) {
//     gpu_stages[s - 1](data);
//   }

//   CheckCuda(cudaStreamSynchronize(mgr.get_stream()));
// }

// // // Input
// //   std::pmr::vector<float> u_image;

// //   // Conv1
// //   std::pmr::vector<float> u_conv1_weights;
// //   std::pmr::vector<float> u_conv1_bias;
// //   std::pmr::vector<float> u_conv1_out;

// //   // Pool1
// //   std::pmr::vector<float> u_pool1_out;

// //   // Conv2
// //   std::pmr::vector<float> u_conv2_weights;
// //   std::pmr::vector<float> u_conv2_bias;
// //   std::pmr::vector<float> u_conv2_out;

// //   // Pool2
// //   std::pmr::vector<float> u_pool2_out;

// //   // Conv3
// //   std::pmr::vector<float> u_conv3_weights;
// //   std::pmr::vector<float> u_conv3_bias;
// //   std::pmr::vector<float> u_conv3_out;

// //   // Conv4
// //   std::pmr::vector<float> u_conv4_weights;
// //   std::pmr::vector<float> u_conv4_bias;
// //   std::pmr::vector<float> u_conv4_out;

// //   // Conv5
// //   std::pmr::vector<float> u_conv5_weights;
// //   std::pmr::vector<float> u_conv5_bias;
// //   std::pmr::vector<float> u_conv5_out;

// //   // Pool3 (also used as flattened)
// //   std::pmr::vector<float> u_pool3_out;

// //   // Linear
// //   std::pmr::vector<float> u_linear_weights;
// //   std::pmr::vector<float> u_linear_bias;
// //   std::pmr::vector<float> u_linear_out;

// // #define CudaAttachSingle(ptr) \
// //   (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0, cudaMemAttachSingle))
// // #define CudaAttachHost(ptr) (cudaStreamAttachMemAsync(mgr.get_stream(), ptr, 0,
// // cudaMemAttachHost))

// // template <int Start, int End>
// //   requires AllowedStage<Start> && AllowedStage<End> && (Start <= End)
// // void run_multiple_stages(cifar_dense::AppData &data, cuda::CudaManager &mgr) {
// //   CudaAttachSingle(data.u_conv1_bias.data());
// //   CudaAttachSingle(data.u_conv1_weights.data());
// //   CudaAttachSingle(data.u_conv1_out.data());
// //   CudaAttachSingle(data.u_pool1_out.data());
// //   CudaAttachSingle(data.u_conv2_bias.data());
// //   CudaAttachSingle(data.u_conv2_weights.data());
// //   CudaAttachSingle(data.u_conv2_out.data());
// //   CudaAttachSingle(data.u_pool2_out.data());
// //   CudaAttachSingle(data.u_conv3_bias.data());
// //   CudaAttachSingle(data.u_conv3_weights.data());
// //   CudaAttachSingle(data.u_conv3_out.data());
// //   CudaAttachSingle(data.u_conv4_bias.data());
// //   CudaAttachSingle(data.u_conv4_weights.data());
// //   CudaAttachSingle(data.u_conv4_out.data());
// //   CudaAttachSingle(data.u_conv5_bias.data());
// //   CudaAttachSingle(data.u_conv5_weights.data());
// //   CudaAttachSingle(data.u_conv5_out.data());
// //   CudaAttachSingle(data.u_pool3_out.data());
// //   CudaAttachSingle(data.u_linear_bias.data());
// //   CudaAttachSingle(data.u_linear_weights.data());
// //   CudaAttachSingle(data.u_linear_out.data());

// //   for (int s = Start; s <= End; ++s) {
// //     gpu_stages[s - 1](data);
// //   }

// //   CheckCuda(cudaStreamSynchronize(mgr.get_stream()));

// //   CudaAttachHost(data.u_conv1_bias.data());
// //   CudaAttachHost(data.u_conv1_weights.data());
// //   CudaAttachHost(data.u_conv1_out.data());
// //   CudaAttachHost(data.u_pool1_out.data());
// //   CudaAttachHost(data.u_conv2_bias.data());
// //   CudaAttachHost(data.u_conv2_weights.data());
// //   CudaAttachHost(data.u_conv2_out.data());
// //   CudaAttachHost(data.u_pool2_out.data());
// //   CudaAttachHost(data.u_conv3_bias.data());
// //   CudaAttachHost(data.u_conv3_weights.data());
// //   CudaAttachHost(data.u_conv3_out.data());
// //   CudaAttachHost(data.u_conv4_bias.data());
// //   CudaAttachHost(data.u_conv4_weights.data());
// //   CudaAttachHost(data.u_conv4_out.data());
// //   CudaAttachHost(data.u_conv5_bias.data());
// //   CudaAttachHost(data.u_conv5_weights.data());
// //   CudaAttachHost(data.u_conv5_out.data());
// //   CudaAttachHost(data.u_pool3_out.data());
// //   CudaAttachHost(data.u_linear_bias.data());
// //   CudaAttachHost(data.u_linear_weights.data());
// //   CudaAttachHost(data.u_linear_out.data());
// // }

// }  // namespace cuda
