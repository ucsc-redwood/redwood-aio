#include <concurrentqueue.h>

#include "../builtin-apps/cifar-dense/arg_max.hpp"
#include "../builtin-apps/cifar-dense/cuda/cu_dense_kernel.cuh"
#include "../builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "../builtin-apps/common/cuda/cu_mem_resource.cuh"

int main() {
  cuda::CudaMemoryResource mr;

  cifar_dense::AppData appdata(&mr);

  cifar_dense::cuda::process_stage_1(&appdata);
  cifar_dense::cuda::process_stage_2(&appdata);
  cifar_dense::cuda::process_stage_3(&appdata);
  cifar_dense::cuda::process_stage_4(&appdata);
  cifar_dense::cuda::process_stage_5(&appdata);
  cifar_dense::cuda::device_sync();

  auto arg_max_index = arg_max(appdata.u_linear_out.data());
  print_prediction(arg_max_index);

  return 0;
}
