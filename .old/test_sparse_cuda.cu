#include "../common/cuda/cu_mem_resource.cuh"
#include "arg_max.hpp"
#include "cuda/cu_dispatcher.cuh"
#include "sparse_appdata.hpp"

int main() {
  cuda::CudaMemoryResource mr;

  cifar_sparse::AppData appdata(&mr);

  cifar_sparse::cuda::process_stage_1(appdata);
  cifar_sparse::cuda::process_stage_2(appdata);
  cifar_sparse::cuda::process_stage_3(appdata);
  cifar_sparse::cuda::process_stage_4(appdata);
  cifar_sparse::cuda::process_stage_5(appdata);
  cifar_sparse::cuda::process_stage_6(appdata);
  cifar_sparse::cuda::process_stage_7(appdata);
  cifar_sparse::cuda::process_stage_8(appdata);
  cifar_sparse::cuda::process_stage_9(appdata);

  cifar_sparse::cuda::device_sync();

  auto arg_max_index = cifar_sparse::arg_max(appdata.u_linear_output.data());
  cifar_sparse::print_prediction(arg_max_index);

  return 0;
}