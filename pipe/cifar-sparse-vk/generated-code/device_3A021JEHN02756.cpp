// Auto-generated aggregated source for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756
#include "device_3A021JEHN02756.hpp"

#include "../run_stages.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_013 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 8, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<9, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_013

namespace CifarSparse_schedule_004 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 3, ProcessorType::kLittleCore, 4>(task);

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

      run_cpu_stages<4, 9, ProcessorType::kMediumCore, 2>(task);

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

namespace CifarSparse_schedule_001 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<3, 5, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<6, 9, ProcessorType::kLittleCore, 4>(task);

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

namespace CifarSparse_schedule_015 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 9, ProcessorType::kMediumCore, 2>(task);

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

namespace CifarSparse_schedule_003 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<3, 6, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

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

namespace CifarSparse_schedule_008 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<4, 4, ProcessorType::kLittleCore, 4>(task);

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

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 2>(task);

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

}  // namespace CifarSparse_schedule_008

namespace CifarSparse_schedule_014 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 9, ProcessorType::kMediumCore, 2>(task);

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

namespace CifarSparse_schedule_012 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<3, 7, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

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

namespace CifarSparse_schedule_002 {

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

void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<2, 3, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<4, 9, ProcessorType::kMediumCore, 2>(task);

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

}  // namespace CifarSparse_schedule_002

namespace CifarSparse_schedule_005 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 5, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<6, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_005

namespace CifarSparse_schedule_007 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 7, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_007

namespace CifarSparse_schedule_006 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 2, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<3, 6, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

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

namespace CifarSparse_schedule_010 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<4, 6, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace CifarSparse_schedule_010

namespace CifarSparse_schedule_011 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<4, 7, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

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

namespace CifarSparse_schedule_009 {

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  for (auto& task : in_tasks) {
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      continue;
    }

    run_cpu_stages<1, 3, ProcessorType::kMediumCore, 2>(task);

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

      run_cpu_stages<4, 5, ProcessorType::kBigCore, 2>(task);

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

      run_cpu_stages<6, 9, ProcessorType::kLittleCore, 4>(task);

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

}  // namespace device_3A021JEHN02756