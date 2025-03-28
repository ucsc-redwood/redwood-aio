#pragma once

#include <cuda_runtime.h>
#include <nvToolsExt.h>

#include <numeric>
#include <thread>

#include "../templates.hpp"
#include "../templates_cu.hpp"
#include "run_stages.hpp"
#include "task.hpp"

namespace generated_schedules {
using run_func_t = void (*)();
struct ScheduleRecord {
  const char *name;
  run_func_t func;
};
}  // namespace generated_schedules

namespace device_jetson {
static void schedule_jetson_CifarDense_schedule_001() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 7>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_001: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_002() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 8>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_002: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_003() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_003: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_004() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_004: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_005() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, nullptr, cuda::run_multiple_stages<1, 9>, mgr);
  });

  t1.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_005: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_006() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_006: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_007() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_007: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_008() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_008: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_009() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_009: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_010() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<6, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_010: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_011() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_011: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_012() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<5, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_012: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_013() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<4, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_013: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_014() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<3, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_014: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_015() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<2, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_015: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_016() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<8, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_016: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_017() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 8, ProcessorType::kLittleCore, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<9, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_017: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetson_CifarDense_schedule_018() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, nullptr, omp::run_multiple_stages<1, 9, ProcessorType::kLittleCore, 6>, mgr);
  });

  t1.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetson_CifarDense_schedule_018: " << avg_task_time
            << " ms" << std::endl;
}

// Table of schedules for this device:
static generated_schedules::ScheduleRecord schedule_table[] = {
    {"jetson_CifarDense_schedule_001", &schedule_jetson_CifarDense_schedule_001},
    {"jetson_CifarDense_schedule_002", &schedule_jetson_CifarDense_schedule_002},
    {"jetson_CifarDense_schedule_003", &schedule_jetson_CifarDense_schedule_003},
    {"jetson_CifarDense_schedule_004", &schedule_jetson_CifarDense_schedule_004},
    {"jetson_CifarDense_schedule_005", &schedule_jetson_CifarDense_schedule_005},
    {"jetson_CifarDense_schedule_006", &schedule_jetson_CifarDense_schedule_006},
    {"jetson_CifarDense_schedule_007", &schedule_jetson_CifarDense_schedule_007},
    {"jetson_CifarDense_schedule_008", &schedule_jetson_CifarDense_schedule_008},
    {"jetson_CifarDense_schedule_009", &schedule_jetson_CifarDense_schedule_009},
    {"jetson_CifarDense_schedule_010", &schedule_jetson_CifarDense_schedule_010},
    {"jetson_CifarDense_schedule_011", &schedule_jetson_CifarDense_schedule_011},
    {"jetson_CifarDense_schedule_012", &schedule_jetson_CifarDense_schedule_012},
    {"jetson_CifarDense_schedule_013", &schedule_jetson_CifarDense_schedule_013},
    {"jetson_CifarDense_schedule_014", &schedule_jetson_CifarDense_schedule_014},
    {"jetson_CifarDense_schedule_015", &schedule_jetson_CifarDense_schedule_015},
    {"jetson_CifarDense_schedule_016", &schedule_jetson_CifarDense_schedule_016},
    {"jetson_CifarDense_schedule_017", &schedule_jetson_CifarDense_schedule_017},
    {"jetson_CifarDense_schedule_018", &schedule_jetson_CifarDense_schedule_018},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_jetson

namespace device_jetsonlowpower {
static void schedule_jetsonlowpower_CifarDense_schedule_001() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 7>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_001: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_002() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 8>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_002: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_003() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_003: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_004() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_004: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_005() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, nullptr, cuda::run_multiple_stages<1, 9>, mgr);
  });

  t1.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_005: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_006() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_006: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_007() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_007: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_008() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_008: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_009() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_009: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_010() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<6, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_010: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_011() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_011: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_012() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<5, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_012: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_013() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<4, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_013: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_014() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<3, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_014: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_015() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_0_1, nullptr, omp::run_multiple_stages<2, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_015: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_016() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<8, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_016: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_017() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");
  moodycamel::ConcurrentQueue<Task *> q_0_1;

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, &q_0_1, omp::run_multiple_stages<1, 8, ProcessorType::kLittleCore, 4>, mgr);
  });
  std::thread t2([&]() {
    chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<9, 9>, mgr);
  });

  t1.join();
  t2.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_017: " << avg_task_time
            << " ms" << std::endl;
}

static void schedule_jetsonlowpower_CifarDense_schedule_018() {
  cuda::CudaManager mgr;

  constexpr size_t num_tasks = 100;

  auto mr = &mgr.get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // ---------------------------------------------------------------------
  // Automatically generated from schedule JSON

  nvtxRangePushA("Compute");

  std::thread t1([&]() {
    chunk<Task, cifar_dense::AppData>(
        q_input, nullptr, omp::run_multiple_stages<1, 9, ProcessorType::kLittleCore, 4>, mgr);
  });

  t1.join();
  nvtxRangePop();

  // ---------------------------------------------------------------------

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  double avg_task_time = milliseconds / num_tasks;
  std::cout << "Average time per task for jetsonlowpower_CifarDense_schedule_018: " << avg_task_time
            << " ms" << std::endl;
}

// Table of schedules for this device:
static generated_schedules::ScheduleRecord schedule_table[] = {
    {"jetsonlowpower_CifarDense_schedule_001", &schedule_jetsonlowpower_CifarDense_schedule_001},
    {"jetsonlowpower_CifarDense_schedule_002", &schedule_jetsonlowpower_CifarDense_schedule_002},
    {"jetsonlowpower_CifarDense_schedule_003", &schedule_jetsonlowpower_CifarDense_schedule_003},
    {"jetsonlowpower_CifarDense_schedule_004", &schedule_jetsonlowpower_CifarDense_schedule_004},
    {"jetsonlowpower_CifarDense_schedule_005", &schedule_jetsonlowpower_CifarDense_schedule_005},
    {"jetsonlowpower_CifarDense_schedule_006", &schedule_jetsonlowpower_CifarDense_schedule_006},
    {"jetsonlowpower_CifarDense_schedule_007", &schedule_jetsonlowpower_CifarDense_schedule_007},
    {"jetsonlowpower_CifarDense_schedule_008", &schedule_jetsonlowpower_CifarDense_schedule_008},
    {"jetsonlowpower_CifarDense_schedule_009", &schedule_jetsonlowpower_CifarDense_schedule_009},
    {"jetsonlowpower_CifarDense_schedule_010", &schedule_jetsonlowpower_CifarDense_schedule_010},
    {"jetsonlowpower_CifarDense_schedule_011", &schedule_jetsonlowpower_CifarDense_schedule_011},
    {"jetsonlowpower_CifarDense_schedule_012", &schedule_jetsonlowpower_CifarDense_schedule_012},
    {"jetsonlowpower_CifarDense_schedule_013", &schedule_jetsonlowpower_CifarDense_schedule_013},
    {"jetsonlowpower_CifarDense_schedule_014", &schedule_jetsonlowpower_CifarDense_schedule_014},
    {"jetsonlowpower_CifarDense_schedule_015", &schedule_jetsonlowpower_CifarDense_schedule_015},
    {"jetsonlowpower_CifarDense_schedule_016", &schedule_jetsonlowpower_CifarDense_schedule_016},
    {"jetsonlowpower_CifarDense_schedule_017", &schedule_jetsonlowpower_CifarDense_schedule_017},
    {"jetsonlowpower_CifarDense_schedule_018", &schedule_jetsonlowpower_CifarDense_schedule_018},
};
static const size_t schedule_count = sizeof(schedule_table) / sizeof(schedule_table[0]);
}  // namespace device_jetsonlowpower
