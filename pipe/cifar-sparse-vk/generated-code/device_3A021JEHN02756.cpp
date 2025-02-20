// Auto-generated aggregated source for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756
#include "device_3A021JEHN02756.hpp"

#include <atomic>
#include <thread>

#include "../run_stages.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_013 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 8, ProcessorType::kMediumCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 8, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<9, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<9, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_013

namespace CifarSparse_schedule_004 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 3, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 3, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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
    run_stages<4, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_004

namespace CifarSparse_schedule_001 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 5, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 5, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_001

namespace CifarSparse_schedule_015 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 2, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 2, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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
    run_stages<3, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_015

namespace CifarSparse_schedule_003 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 6, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 6, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_003

namespace CifarSparse_schedule_008 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 4, ProcessorType::kLittleCore, 4>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 4, ProcessorType::kLittleCore, 4>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<5, 9, ProcessorType::kBigCore, 2>(task.app_data);
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
    run_stages<5, 9, ProcessorType::kBigCore, 2>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_008

namespace CifarSparse_schedule_014 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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
    run_stages<3, 9, ProcessorType::kMediumCore, 2>(task.app_data);
    out_tasks.push_back(task);
    int r = tasks_in_flight.fetch_sub(1, std::memory_order_relaxed) - 1;
    if (r == 0) done.store(true, std::memory_order_release);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  tasks_in_flight.store(0, std::memory_order_relaxed);
  done.store(false, std::memory_order_relaxed);

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // end namespace CifarSparse_schedule_014

namespace CifarSparse_schedule_012 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_012

namespace CifarSparse_schedule_002 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 1, ProcessorType::kLittleCore, 4>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<2, 3, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<2, 3, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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
    run_stages<4, 9, ProcessorType::kMediumCore, 2>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_002

namespace CifarSparse_schedule_005 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 5, ProcessorType::kMediumCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 5, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_005

namespace CifarSparse_schedule_007 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 7, ProcessorType::kMediumCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 7, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_007

namespace CifarSparse_schedule_006 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 2, ProcessorType::kBigCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<3, 6, ProcessorType::kMediumCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<3, 6, ProcessorType::kMediumCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_006

namespace CifarSparse_schedule_010 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 6, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 6, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<7, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_010

namespace CifarSparse_schedule_011 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 7, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 7, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<8, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_011

namespace CifarSparse_schedule_009 {

static std::atomic<int> tasks_in_flight{0};
static std::atomic<bool> done(false);

void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    run_stages<1, 3, ProcessorType::kMediumCore, 2>(task.app_data);
    tasks_in_flight.fetch_add(1, std::memory_order_relaxed);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<4, 5, ProcessorType::kBigCore, 2>(task.app_data);
      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
  while (true) {
    Task task;
    if (!in_q.try_dequeue(task)) break;
    run_stages<4, 5, ProcessorType::kBigCore, 2>(task.app_data);
    out_q.enqueue(task);
  }
}

void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (!done.load(std::memory_order_acquire)) {
    Task task;
    if (in_q.try_dequeue(task)) {
      run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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
    run_stages<6, 9, ProcessorType::kLittleCore, 4>(task.app_data);
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

  std::thread t_chunk1(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk1(tasks, q_01); });
  std::thread t_chunk2(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk2(q_01, q_12); });
  std::thread t_chunk3(
      [&]() { stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // end namespace CifarSparse_schedule_009

}  // namespace device_3A021JEHN02756