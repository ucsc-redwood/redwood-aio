#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "generated_code_non_bm.cuh"

__global__ void kernel_test() {}

void warmup() {
  kernel_test<<<1, 1>>>();
  CheckCuda(cudaDeviceSynchronize());
}

struct DeviceInfo {
  const generated_schedules::ScheduleRecord *table;
  size_t count;
};

// Now create a lookup map from the string device ID to the device's table info
static std::map<std::string, DeviceInfo> g_device_map = {
    {"jetson", {device_jetson::schedule_table, device_jetson::schedule_count}},
    {"jetsonlowpower",
     {device_jetsonlowpower::schedule_table, device_jetsonlowpower::schedule_count}},
};

void run_single_schedule(const std::string &device_id, int schedule_index) {
  // Look up the device
  auto it = g_device_map.find(device_id);
  if (it == g_device_map.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  // Confirm the schedule index is valid
  const DeviceInfo &dev_info = it->second;
  if (schedule_index < 0 || schedule_index >= static_cast<int>(dev_info.count)) {
    throw std::runtime_error("Invalid schedule index " + std::to_string(schedule_index) +
                             " for device " + device_id + ". Must be between 0 and " +
                             std::to_string(dev_info.count - 1));
  }

  // Grab the chosen schedule record
  const auto &rec = dev_info.table[schedule_index];

  // rec.name is the schedule_id string
  // rec.func is the function pointer
  rec.func();
}

static void schedule_jetson_CifarDense_cpu_baseline() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  auto start_time = std::chrono::high_resolution_clock::now();

  // ---------------------------------------------------------------------

  chunk<Task, cifar_dense::AppData>(
      q_input, nullptr, omp::run_multiple_stages<1, 9, ProcessorType::kLittleCore, 6>, mgr);

  // ---------------------------------------------------------------------

  auto end_time = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
  double avg_task_time = elapsed / num_tasks;
  std::cout << "CPU baseline time per task: " << avg_task_time << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_gpu_baseline() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  //   auto start_time = std::chrono::high_resolution_clock::now();

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------

  chunk<Task, cifar_dense::AppData>(q_input, nullptr, cuda::run_multiple_stages<1, 9>, mgr);

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "GPU baseline time per task: " << avg_task_time << " ms" << std::endl;
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  warmup();

  //   schedule_jetson_CifarDense_cpu_baseline();
  schedule_jetson_CifarDense_gpu_baseline();

  spdlog::info("Running schedule {} for device {}", schedule_index, g_device_id);
  run_single_schedule(g_device_id, schedule_index);

  return 0;
}
