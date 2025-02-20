// Auto-generated aggregated source for device: ce0717178d7758b00b7e
// Contains all 'Tree' schedules for device_ce0717178d7758b00b7e
#include "device_ce0717178d7758b00b7e.hpp"

#include <atomic>
#include <thread>
#include "../run_stages.hpp"

namespace device_ce0717178d7758b00b7e {

namespace Tree_schedule_008 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 2, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 2, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_008_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_008

namespace Tree_schedule_021 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_021_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_021_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_021_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_021_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_021

namespace Tree_schedule_040 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_040_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 6, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_040_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<7, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_040_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_040_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_040

namespace Tree_schedule_014 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<5, 5>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<5, 5>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_014_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_014

namespace Tree_schedule_042 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_042_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_042_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_042_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_042_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_042

namespace Tree_schedule_011 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 5, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 5, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_011_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_011

namespace Tree_schedule_017 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<7, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_017_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_017

namespace Tree_schedule_041 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_041_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 6, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_041_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_041_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_041_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_041

namespace Tree_schedule_016 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_016_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_016

namespace Tree_schedule_013 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_013_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_013_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_013_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_013_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_013

namespace Tree_schedule_033 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_033_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_033

namespace Tree_schedule_036 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_036_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_036

namespace Tree_schedule_018 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_018_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 5, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_018_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_018_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_018_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_018

namespace Tree_schedule_022 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_022_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_022_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<7, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_022_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_022_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_022

namespace Tree_schedule_004 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_004_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_004

namespace Tree_schedule_034 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_034_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_034

namespace Tree_schedule_031 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_031_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_031

namespace Tree_schedule_026 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_026_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_026_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_026_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_026_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_026

namespace Tree_schedule_015 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_015_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_015

namespace Tree_schedule_020 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_020_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_020_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_020_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_020_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_020

namespace Tree_schedule_025 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 2, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 2, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_025_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_025

namespace Tree_schedule_002 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<4, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<4, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_002_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_002

namespace Tree_schedule_005 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_005_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_005_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_005_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_005_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_005

namespace Tree_schedule_027 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<5, 5>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<5, 5>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_027_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_027

namespace Tree_schedule_032 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_032_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_032

namespace Tree_schedule_043 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_043_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_043_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_043_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_043_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_043

namespace Tree_schedule_009 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 4, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_009_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_009

namespace Tree_schedule_038 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_038_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 5, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_038_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_038_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_038_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_038

namespace Tree_schedule_037 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 5, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_037_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_037

namespace Tree_schedule_024 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_024_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks) {
  for (auto& task : in_tasks) {
    run_stages<1, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_024_chunk1(tasks, out_tasks); });

  t_chunk1.join();
}

}  // end namespace Tree_schedule_024

namespace Tree_schedule_003 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<4, 5>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<4, 5>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_003_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_003

namespace Tree_schedule_019 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 5, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_019_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_019

namespace Tree_schedule_006 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 3, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_006_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_006

namespace Tree_schedule_028 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_028_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_028_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_028_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_028_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_028

namespace Tree_schedule_012 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 3, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 3, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_012_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_012

namespace Tree_schedule_023 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_023_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_023_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_023_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_023_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_023

namespace Tree_schedule_029 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_029_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_029

namespace Tree_schedule_035 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<5, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_035_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_035

namespace Tree_schedule_030 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_030_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_030

namespace Tree_schedule_010 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_gpu_stages<1, 1>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 2, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 2, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kLittleCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kLittleCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_010_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_010

namespace Tree_schedule_039 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 5, ProcessorType::kBigCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 6, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<6, 6, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_039_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_039

namespace Tree_schedule_001 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 6, ProcessorType::kBigCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 6, ProcessorType::kBigCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_gpu_stages<7, 7>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_gpu_stages<7, 7>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_001_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace Tree_schedule_001

namespace Tree_schedule_007 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_ce0717178d7758b00b7e_Tree_schedule_007_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_ce0717178d7758b00b7e_Tree_schedule_007_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
      out_tasks.push_back(task);
      int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
      if (r == 0) done.store(true, std::memory_order_release);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kBigCore, 4>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_007_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { stage_group_ce0717178d7758b00b7e_Tree_schedule_007_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace Tree_schedule_007

}  // namespace device_ce0717178d7758b00b7e