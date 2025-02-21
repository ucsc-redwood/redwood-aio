// Auto-generated aggregated source for device: ce0717178d7758b00b7e
// Contains all 'CifarSparse' schedules for device_ce0717178d7758b00b7e
#include "device_ce0717178d7758b00b7e.hpp"

#include "../run_stages.hpp"

namespace device_ce0717178d7758b00b7e {

namespace CifarSparse_schedule_005 {

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

      run_cpu_stages<6, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_005

namespace CifarSparse_schedule_003 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 8, ProcessorType::kBigCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_003

namespace CifarSparse_schedule_002 {

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

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_002

namespace CifarSparse_schedule_013 {

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

      run_cpu_stages<4, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_013

namespace CifarSparse_schedule_006 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 8, ProcessorType::kBigCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_006

namespace CifarSparse_schedule_014 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 7, ProcessorType::kLittleCore, 4>(task);

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

      run_cpu_stages<8, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_014

namespace CifarSparse_schedule_020 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_020

namespace CifarSparse_schedule_011 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<7, 8, ProcessorType::kBigCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_011

namespace CifarSparse_schedule_015 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 7, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<8, 8, ProcessorType::kBigCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_015

namespace CifarSparse_schedule_019 {

void chunk_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_tasks.push_back(task);
      continue;
    }

    run_cpu_stages<1, 9, ProcessorType::kLittleCore, 4>(task);

    out_tasks.push_back(task);
  }
}

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks) {
  std::thread t_chunk1([&]() { chunk_chunk1(tasks, out_tasks); });

  t_chunk1.join();
}

}  // namespace CifarSparse_schedule_019

namespace CifarSparse_schedule_004 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 8, ProcessorType::kLittleCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_004

namespace CifarSparse_schedule_012 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 8, ProcessorType::kBigCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_012

namespace CifarSparse_schedule_008 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push_back(task);
        break;
      }

      run_cpu_stages<3, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_008

namespace CifarSparse_schedule_010 {

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

      run_cpu_stages<7, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_010

namespace CifarSparse_schedule_009 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<3, 8, ProcessorType::kLittleCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_009

namespace CifarSparse_schedule_016 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 7, ProcessorType::kLittleCore, 4>(task);

    out_q.enqueue(task);
  }
}

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<8, 8>(task);

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

      run_cpu_stages<9, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_016

namespace CifarSparse_schedule_018 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 8, ProcessorType::kLittleCore, 4>(task);

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

      run_gpu_stages<9, 9>(task);

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

}  // namespace CifarSparse_schedule_018

namespace CifarSparse_schedule_001 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<2, 2>(task);

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

      run_cpu_stages<3, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_001

namespace CifarSparse_schedule_007 {

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

      run_cpu_stages<2, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_007

namespace CifarSparse_schedule_017 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 8, ProcessorType::kLittleCore, 4>(task);

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

      run_cpu_stages<9, 9, ProcessorType::kBigCore, 4>(task);

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

}  // namespace CifarSparse_schedule_017

}  // namespace device_ce0717178d7758b00b7e