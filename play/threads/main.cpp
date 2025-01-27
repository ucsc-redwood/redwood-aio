#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "concurrentqueue.h"  // Include Moodycamel's ConcurrentQueue

struct Task {
  int uid;
  float* data;
};

// Global atomic flag to control threads
std::atomic<bool> done(false);

std::mutex mtx;

void producer(moodycamel::ConcurrentQueue<Task>& queue, int num_tasks) {
  for (int i = 0; i < num_tasks; ++i) {
    float* data = new float[10];  // Dynamically allocate some data
    for (int j = 0; j < 10; ++j) {
      data[j] = static_cast<float>(i + j) / 2.0f;
    }

    Task task{i, data};
    queue.enqueue(task);

    {
      std::lock_guard<std::mutex> lock(mtx);
      std::cout << "Produced Task ID: " << i << "\n";
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(100));  // Simulate work
  }

  done = true;  // Signal consumer to stop
}

void consumer(moodycamel::ConcurrentQueue<Task>& queue) {
  while (!done) {
    Task task;
    if (queue.try_dequeue(task)) {
      {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Consumed Task ID: " << task.uid << "\n";
      }

      // Clean up dynamically allocated data
      delete[] task.data;
    } else {
      // No task available, yield to avoid busy-waiting
      std::this_thread::yield();
    }
  }
}

int main() {
  moodycamel::ConcurrentQueue<Task> queue;
  const int num_tasks = 20;

  // Start producer and consumer threads
  std::thread producer_thread(producer, std::ref(queue), num_tasks);
  std::thread consumer_thread(consumer, std::ref(queue));

  // Join threads
  producer_thread.join();
  consumer_thread.join();

  std::cout << "All tasks processed." << std::endl;
  return 0;
}