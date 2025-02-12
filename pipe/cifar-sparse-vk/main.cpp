
// #include <concurrentqueue.h>
#include "third-party/concurrentqueue.h"

#include <cstdint>
#include <memory_resource>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"
#include "cifar-sparse/vulkan/vk_dispatcher.hpp"
#include "spdlog/common.h"

enum class ProcessorType {
  kLittleCore,
  kMediumCore,
  kBigCore,
};

template <int start_stage, int end_stage, ProcessorType processor_type, int num_threads>
void run_stages(cifar_sparse::AppData* app_data) {
  static_assert(start_stage >= 1 && end_stage <= 9, "Stage range out of bounds");
  static_assert(start_stage <= end_stage, "start_stage must be <= end_stage");

#pragma omp parallel num_threads(num_threads)
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    }

    // Generate a compile-time sequence for the range [start_stage, end_stage]
    []<std::size_t... I>(std::index_sequence<I...>, cifar_sparse::AppData& data) {
      // Each I is offset by (start_stage - 1)
      ((cifar_sparse::omp::run_stage<start_stage + I>(data)), ...);
    }(std::make_index_sequence<end_stage - start_stage + 1>{}, *app_data);
  }
}

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  // uint32_t uid;
  cifar_sparse::AppData* app_data;
};

std::atomic<bool> done(false);

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();

  std::vector<Task> tasks(num_tasks);

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks[i] = Task{
        // .uid = i,                                   //
        .app_data = new cifar_sparse::AppData(mr),  //
    };
  }

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  for (auto& task : tasks) {
    delete task.app_data;
  }
}

// ---------------------------------------------------------------------
// Device-specific pipeline stages (3A021JEHN02756)
// ---------------------------------------------------------------------

namespace device_3A021JEHN02756 {

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB,
                   moodycamel::ConcurrentQueue<Task>& q_BC) {
  while (!done) {
    Task task;
    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<4, 5, ProcessorType::kMediumCore, 2>(task.app_data);
      // ---------

      q_BC.enqueue(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

void stage_group_C(moodycamel::ConcurrentQueue<Task>& q_BC, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;
    if (q_BC.try_dequeue(task)) {
      // ---------
      run_stages<6, 9, ProcessorType::kBigCore, 2>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

}  // namespace device_3A021JEHN02756

// ---------------------------------------------------------------------
// Device-specific pipeline stages (9b034f1b)
// ---------------------------------------------------------------------

namespace device_9b034f1b {

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 4, ProcessorType::kLittleCore, 3>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;
    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<5, 9, ProcessorType::kMediumCore, 2>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

}  // namespace device_9b034f1b

// ---------------------------------------------------------------------
// Device-specific pipeline stages (ce0717178d7758b00b7e)
// ---------------------------------------------------------------------

namespace device_ce0717178d7758b00b7e {

void stage_group_A(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& q_AB) {
  for (auto& task : in_tasks) {
    // ---------
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    // ---------

    q_AB.enqueue(task);
  }

  // Signal consumer to stop
  done = true;
}

void stage_group_B(moodycamel::ConcurrentQueue<Task>& q_AB, std::vector<Task>& out_tasks) {
  while (!done) {
    Task task;
    if (q_AB.try_dequeue(task)) {
      // ---------
      run_stages<5, 9, ProcessorType::kMediumCore, 4>(task.app_data);
      // ---------

      out_tasks.push_back(task);
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

}  // namespace device_ce0717178d7758b00b7e

// ---------------------------------------------------------------------
// Pipeline Instance
// ---------------------------------------------------------------------

void tmp() {
  auto tasks = init_tasks(10);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  if (g_device_id == "3A021JEHN02756") {
    // Little cores: 0 1 2 3
    // Mid cores: 4 5
    // Big cores: 6 7

    moodycamel::ConcurrentQueue<Task> q_AB;
    moodycamel::ConcurrentQueue<Task> q_BC;

    std::thread t_A(device_3A021JEHN02756::stage_group_A, std::ref(tasks), std::ref(q_AB));
    std::thread t_B(device_3A021JEHN02756::stage_group_B, std::ref(q_AB), std::ref(q_BC));
    std::thread t_C(device_3A021JEHN02756::stage_group_C, std::ref(q_BC), std::ref(out_tasks));

    t_A.join();
    t_B.join();
    t_C.join();

  } else if (g_device_id == "9b034f1b") {
    // Little cores: 0 1 2
    // Mid cores: 3 4
    // Big cores: (un pinnable)

    moodycamel::ConcurrentQueue<Task> q_AB;

    std::thread t1(device_9b034f1b::stage_group_A, std::ref(tasks), std::ref(q_AB));
    std::thread t2(device_9b034f1b::stage_group_B, std::ref(q_AB), std::ref(out_tasks));

    t1.join();
    t2.join();

  } else if (g_device_id == "ce0717178d7758b00b7e") {
    // Little cores: 4 5 6 7
    // Mid cores: 0 1 2 3
    // Big cores:

    moodycamel::ConcurrentQueue<Task> q_AB;

    std::thread t1(device_ce0717178d7758b00b7e::stage_group_A, std::ref(tasks), std::ref(q_AB));
    std::thread t2(device_ce0717178d7758b00b7e::stage_group_B, std::ref(q_AB), std::ref(out_tasks));

    t1.join();
    t2.join();
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "Pipeline: Average time per iteration: " << avg_time << " us" << std::endl;

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Baseline
// ---------------------------------------------------------------------

void run_baseline() {
  auto tasks = init_tasks(10);
  std::vector<Task> out_tasks;
  out_tasks.reserve(tasks.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (auto& task : tasks) {
#pragma omp parallel
    {
      cifar_sparse::omp::run_stage<1>(*task.app_data);
      cifar_sparse::omp::run_stage<2>(*task.app_data);
      cifar_sparse::omp::run_stage<3>(*task.app_data);
      cifar_sparse::omp::run_stage<4>(*task.app_data);
      cifar_sparse::omp::run_stage<5>(*task.app_data);
      cifar_sparse::omp::run_stage<6>(*task.app_data);
      cifar_sparse::omp::run_stage<7>(*task.app_data);
      cifar_sparse::omp::run_stage<8>(*task.app_data);
      cifar_sparse::omp::run_stage<9>(*task.app_data);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double avg_time = duration.count() / static_cast<double>(tasks.size());
  std::cout << "Baseline: Average time per iteration: " << avg_time << " us" << std::endl;

  cleanup(tasks);
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  tmp();
  run_baseline();

  return 0;
}