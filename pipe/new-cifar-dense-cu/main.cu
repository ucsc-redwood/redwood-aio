#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "../templates_cu.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/manager.cuh"
#include "run_stages.hpp"
#include "task.hpp"

__global__ void kernel_test() {}

void warmup() {
  kernel_test<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}

namespace device_test {

static void BM_schedule_test_CifarDense_schedule_001() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 20;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  moodycamel::ConcurrentQueue<Task *> q_0_1;
  moodycamel::ConcurrentQueue<Task *> q_1_2;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, cuda::run_multiple_stages<3, 7>, mgr);
  });
  std::thread t3([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kBigCore, 2>, mgr);
  });

  t1.join();
  t2.join();
  t3.join();
}

}  // namespace device_test

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::debug);

  warmup();

  device_test::BM_schedule_test_CifarDense_schedule_001();

  return 0;
}
