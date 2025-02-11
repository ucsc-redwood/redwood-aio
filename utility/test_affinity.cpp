#include <iostream>
#include <thread>
#include <vector>

#include "../builtin-apps/affinity.hpp"

int main() {
  auto num_cores = std::thread::hardware_concurrency();
  std::vector<bool> success_status(num_cores, false);
  std::vector<std::string> error_messages(num_cores);

  std::vector<std::thread> threads;
  for (int i = 0; i < num_cores; i++) {
    threads.emplace_back([i, &success_status, &error_messages]() {
      try {
        bind_thread_to_core({i});
        success_status[i] = true;
      } catch (const std::exception& e) {
        success_status[i] = false;
        error_messages[i] = e.what();
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // Print report
  std::cout << "\n=== Thread Binding Report ===\n";
  for (int i = 0; i < num_cores; i++) {
    std::cout << "Core " << i << ": ";
    if (success_status[i]) {
      std::cout << "Successfully bound\n";
    } else {
      std::cout << "Failed to bind - " << error_messages[i] << "\n";
    }
  }

  int success_count = std::count(success_status.begin(), success_status.end(), true);
  std::cout << "\nSummary:\n";
  std::cout << "Total cores tested: " << num_cores << "\n";
  std::cout << "Successfully bound: " << success_count << "\n";
  std::cout << "Failed to bind: " << (num_cores - success_count) << "\n";

  return 0;
}