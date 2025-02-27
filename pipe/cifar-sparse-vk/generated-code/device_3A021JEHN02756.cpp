// Auto-generated aggregated source for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756
#include "device_3A021JEHN02756.hpp"

#include "../run_stages.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_036 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 4, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_036

namespace CifarSparse_schedule_020 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_020

namespace CifarSparse_schedule_019 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 6, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_019

namespace CifarSparse_schedule_013 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 3>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_013

namespace CifarSparse_schedule_028 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 3>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_028

namespace CifarSparse_schedule_004 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 5, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_004

namespace CifarSparse_schedule_001 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
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

namespace CifarSparse_schedule_031 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<4, 7>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_031

namespace CifarSparse_schedule_034 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 4, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace CifarSparse_schedule_034

namespace CifarSparse_schedule_016 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
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

namespace CifarSparse_schedule_033 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 4, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace CifarSparse_schedule_033

namespace CifarSparse_schedule_045 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_045

namespace CifarSparse_schedule_015 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 3>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_015

namespace CifarSparse_schedule_003 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 5, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_003

namespace CifarSparse_schedule_032 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_032

namespace CifarSparse_schedule_040 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_040

namespace CifarSparse_schedule_021 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_021

namespace CifarSparse_schedule_049 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 4, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 5, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_049

namespace CifarSparse_schedule_008 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 8, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<9, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_008

namespace CifarSparse_schedule_044 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_044

namespace CifarSparse_schedule_014 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 3>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_014

namespace CifarSparse_schedule_022 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_022

namespace CifarSparse_schedule_046 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 3, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_046

namespace CifarSparse_schedule_038 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 5, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_038

namespace CifarSparse_schedule_035 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace CifarSparse_schedule_035

namespace CifarSparse_schedule_017 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
}

}  // namespace CifarSparse_schedule_017

namespace CifarSparse_schedule_048 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 4, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 5, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_048

namespace CifarSparse_schedule_012 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_012

namespace CifarSparse_schedule_002 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
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

namespace CifarSparse_schedule_025 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_025

namespace CifarSparse_schedule_018 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_018

namespace CifarSparse_schedule_037 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 4, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<5, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_037

namespace CifarSparse_schedule_024 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kLittleCore, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_024

namespace CifarSparse_schedule_050 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 4, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_050

namespace CifarSparse_schedule_041 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_041

namespace CifarSparse_schedule_042 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 3, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_042

namespace CifarSparse_schedule_023 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 7, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_023

namespace CifarSparse_schedule_026 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 8, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<9, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_026

namespace CifarSparse_schedule_030 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 1>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_030

namespace CifarSparse_schedule_043 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 3, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_043

namespace CifarSparse_schedule_005 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_005

namespace CifarSparse_schedule_047 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_cpu_stages<3, 3, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<4, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_047

namespace CifarSparse_schedule_039 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_gpu_stages<1, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<6, 9, ProcessorType::kMediumCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_039

namespace CifarSparse_schedule_027 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 5>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<6, 8, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<9, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_027

namespace CifarSparse_schedule_007 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 8, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<9, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_007

namespace CifarSparse_schedule_006 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 7, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<8, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_006

namespace CifarSparse_schedule_010 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kBigCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 6, ProcessorType::kMediumCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_010

namespace CifarSparse_schedule_011 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
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

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kBigCore, 2>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_011

namespace CifarSparse_schedule_009 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

      run_gpu_stages<2, 4>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_cpu_stages<5, 6, ProcessorType::kBigCore, 2>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_009

namespace CifarSparse_schedule_029 {

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q) {
  while (!in_tasks.empty()) {
    auto& task = in_tasks.front();
    if (task.is_sentinel()) {
      out_q.enqueue(task);
      in_tasks.pop();
      continue;
    }

    run_cpu_stages<1, 1, ProcessorType::kMediumCore, 2>(task);

    out_q.enqueue(task);
    in_tasks.pop();
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

void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_q.enqueue(task);
        break;
      }

      run_gpu_stages<4, 6>(task);

      out_q.enqueue(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void chunk_chunk4(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks) {
  while (true) {
    Task task;
    if (in_q.try_dequeue(task)) {
      if (task.is_sentinel()) {
        out_tasks.push(task);
        break;
      }

      run_cpu_stages<7, 9, ProcessorType::kLittleCore, 4>(task);

      out_tasks.push(task);
    } else {
      std::this_thread::yield();
    }
  }
}

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks) {
  moodycamel::ConcurrentQueue<Task> q_01;
  moodycamel::ConcurrentQueue<Task> q_12;
  moodycamel::ConcurrentQueue<Task> q_23;

  std::thread t_chunk1([&]() { chunk_chunk1(tasks, q_01); });
  std::thread t_chunk2([&]() { chunk_chunk2(q_01, q_12); });
  std::thread t_chunk3([&]() { chunk_chunk3(q_12, q_23); });
  std::thread t_chunk4([&]() { chunk_chunk4(q_23, out_tasks); });

  t_chunk1.join();
  t_chunk2.join();
  t_chunk3.join();
  t_chunk4.join();
}

}  // namespace CifarSparse_schedule_029

}  // namespace device_3A021JEHN02756