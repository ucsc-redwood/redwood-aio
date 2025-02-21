// Auto-generated aggregated source for device: ce0717178d7758b00b7e
// Contains all 'Tree' schedules for device_ce0717178d7758b00b7e
#include "device_ce0717178d7758b00b7e.hpp"

#include "../run_stages.hpp"

namespace device_ce0717178d7758b00b7e {

namespace Tree_schedule_008 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 2, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<3, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_008

namespace Tree_schedule_021 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<2, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_021

namespace Tree_schedule_040 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 6, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<7, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_040

namespace Tree_schedule_014 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<5, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_014

namespace Tree_schedule_042 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<2, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_042

namespace Tree_schedule_011 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 5, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_011

namespace Tree_schedule_017 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<7, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_017

namespace Tree_schedule_041 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 6, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_041

namespace Tree_schedule_016 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_016

namespace Tree_schedule_013 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_013

namespace Tree_schedule_033 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_033

namespace Tree_schedule_036 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_036

namespace Tree_schedule_018 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 5, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_018

namespace Tree_schedule_022 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 6, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<7, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_022

namespace Tree_schedule_004 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_004

namespace Tree_schedule_034 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_034

namespace Tree_schedule_031 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<3, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_031

namespace Tree_schedule_026 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_026

namespace Tree_schedule_015 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_015

namespace Tree_schedule_020 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<2, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_020

namespace Tree_schedule_025 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 2, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<3, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_025

namespace Tree_schedule_002 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<4, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_002

namespace Tree_schedule_005 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_005

namespace Tree_schedule_027 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<5, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_027

namespace Tree_schedule_032 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<3, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_032

namespace Tree_schedule_043 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<2, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_043

namespace Tree_schedule_009 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 4, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_009

namespace Tree_schedule_038 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 5, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_038

namespace Tree_schedule_037 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 5, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_037

namespace Tree_schedule_024 {

void chunk_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_tasks.push_back(task);
      continue;
    }

    run_cpu_stages<1, 7, ProcessorType::kLittleCore, 4>(task);

    out_tasks.push_back(task);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, out_tasks); });

  t_chunk1.join();
}

}  // namespace Tree_schedule_024

namespace Tree_schedule_003 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<4, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_003

namespace Tree_schedule_019 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 5, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_019

namespace Tree_schedule_006 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 3, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_006

namespace Tree_schedule_028 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 6, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_028

namespace Tree_schedule_012 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 3, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_012

namespace Tree_schedule_023 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 6, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_023

namespace Tree_schedule_029 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_029

namespace Tree_schedule_035 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 4, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_035

namespace Tree_schedule_030 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_030

namespace Tree_schedule_010 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 2, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<3, 7, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_010

namespace Tree_schedule_039 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 5, ProcessorType::kBigCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 6, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_039

namespace Tree_schedule_001 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<3, 6, ProcessorType::kBigCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_gpu_stages<7, 7>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace Tree_schedule_001

namespace Tree_schedule_007 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<3, 7, ProcessorType::kBigCore, 4>(task);

      out_tasks.push_back(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace Tree_schedule_007

}  // namespace device_ce0717178d7758b00b7e