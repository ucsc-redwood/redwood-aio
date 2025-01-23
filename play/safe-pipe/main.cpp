#include "concurrentqueue.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

#include "../../builtin-apps/cifar-sparse/arg_max.hpp"
#include "../../builtin-apps/cifar-sparse/omp/sparse_kernel.hpp"
#include "../../builtin-apps/cifar-sparse/sparse_appdata.hpp"
#include "../../builtin-apps/cifar-sparse/vulkan/vk_dispatcher.hpp"
#include "../../builtin-apps/common/vulkan/engine.hpp"
#include "../../pipe/affinity.hpp"
#include "../../pipe/app.hpp"
#include "../../pipe/conf.hpp"

// void run_2_stage() {
//   auto mr = vulkan::Singleton::getInstance().get_mr();

//   std::vector double_buffer = {
//       cifar_sparse::AppData(mr),
//       cifar_sparse::AppData(mr),
//   };

//   constexpr auto n_iterations = 10;

//   for (auto i = 0; i < n_iterations; ++i) {
//     auto cur = i % 2;
//     auto next = (i + 1) % 2;

//     if (cur == 0) {
//       // process first half of the buffer
//       vulkan::Singleton::getInstance().process_stage_1(double_buffer[cur]);
//       vulkan::Singleton::getInstance().process_stage_2(double_buffer[cur]);
//       vulkan::Singleton::getInstance().process_stage_3(double_buffer[cur]);
//       vulkan::Singleton::getInstance().process_stage_4(double_buffer[cur]);
//       vulkan::Singleton::getInstance().sync();

// //       // process first half of the buffer
// // #pragma omp parallel num_threads(4)
// //       {
// //         bind_thread_to_core<4, 5, 6, 7>();
// //         cifar_sparse::omp::process_stage_1(double_buffer[next]);
// //         cifar_sparse::omp::process_stage_2(double_buffer[next]);
// //         cifar_sparse::omp::process_stage_3(double_buffer[next]);
// //         cifar_sparse::omp::process_stage_4(double_buffer[next]);
// //       }

//     } else {
//       // process first half of the buffer
//       vulkan::Singleton::getInstance().process_stage_1(double_buffer[next]);
//       vulkan::Singleton::getInstance().process_stage_2(double_buffer[next]);
//       vulkan::Singleton::getInstance().process_stage_3(double_buffer[next]);
//       vulkan::Singleton::getInstance().process_stage_4(double_buffer[next]);
//       vulkan::Singleton::getInstance().sync();

// // #pragma omp parallel num_threads(4)
// //       {
// //         bind_thread_to_core<4, 5, 6, 7>();
// //         cifar_sparse::omp::process_stage_1(double_buffer[cur]);
// //         cifar_sparse::omp::process_stage_2(double_buffer[cur]);
// //         cifar_sparse::omp::process_stage_3(double_buffer[cur]);
// //         cifar_sparse::omp::process_stage_4(double_buffer[cur]);
// //       }
//     }
//   }

//   for (auto i = 0; i < 2; ++i) {
//     auto arg_max_index =
//         cifar_sparse::arg_max(double_buffer[i].u_linear_output.data());
//     cifar_sparse::print_prediction(arg_max_index);
//   }
// }

struct Task {
  int uid;

  // what should I use here?
  std::unique_ptr<cifar_sparse::AppData> app_data;

  bool is_done;
};

void run_2_stage_queue() {
  auto mr = vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> q_A;
  moodycamel::ConcurrentQueue<Task> q_AB;
  std::queue<Task> q_B;

  for (int i = 0; i < 10; ++i) {
    Task t;
    t.uid = i;
    t.app_data = std::make_unique<cifar_sparse::AppData>(mr);
    t.is_done = false;

    // Note: we use std::move since we have a unique_ptr in Task
    q_A.push(std::move(t));
  }

  // add a sentinel task
  q_A.push(Task{0, nullptr, true});

  // ---

  std::mutex mtx;

  std::thread t_A([&]() {
    while (true) {
      Task t = std::move(q_A.front());
      q_A.pop();
      if (t.is_done) {
        q_AB.enqueue(std::move(t));
        break;
      }

#pragma omp parallel num_threads(4)
      {
        bind_thread_to_core<1, 2, 3, 4>();
        cifar_sparse::omp::process_stage_1(*t.app_data);
        cifar_sparse::omp::process_stage_2(*t.app_data);
        cifar_sparse::omp::process_stage_3(*t.app_data);
        cifar_sparse::omp::process_stage_4(*t.app_data);
        cifar_sparse::omp::process_stage_5(*t.app_data);
      }

      q_AB.enqueue(std::move(t));

      {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[A] processed task " << t.uid << std::endl;
      }
    }
  });

  std::thread t_B([&]() {
    while (true) {
      Task t;
      if (q_AB.try_dequeue(t)) {
        if (t.is_done) {
          break;
        }

        vulkan::Singleton::getInstance().process_stage_6(*t.app_data);
        vulkan::Singleton::getInstance().process_stage_7(*t.app_data);
        vulkan::Singleton::getInstance().process_stage_8(*t.app_data);
        vulkan::Singleton::getInstance().process_stage_9(*t.app_data);
        vulkan::Singleton::getInstance().sync();

        q_B.push(std::move(t));

        {
          std::lock_guard<std::mutex> lock(mtx);
          std::cout << "[B] processed task " << t.uid << std::endl;
        }
      }
    }
  });

  t_A.join();
  t_B.join();

  std::cout << "[main] done" << std::endl;

  // -- check each result
  for (int i = 0; i < 10; ++i) {
    auto arg_max_index =
        cifar_sparse::arg_max(q_B.front().app_data->u_linear_output.data());
    cifar_sparse::print_prediction(arg_max_index);
    q_B.pop();
  }
}

void run_queue() {
  auto mr = vulkan::Singleton::getInstance().get_mr();

  std::queue<Task> q_A;
  std::queue<Task> q_B;

  for (int i = 0; i < 10; ++i) {
    Task t;
    t.uid = i;
    t.app_data = std::make_unique<cifar_sparse::AppData>(mr);
    t.is_done = false;

    // Note: we use std::move since we have a unique_ptr in Task
    q_A.push(std::move(t));
  }

  // add a sentinel task
  q_A.push(Task{0, nullptr, true});

  // ---

  while (true) {
    Task t = std::move(q_A.front());
    q_A.pop();
    if (t.is_done) {
      break;
    }

    // vulkan::Singleton::getInstance().process_stage_1(*t.app_data);
    // vulkan::Singleton::getInstance().process_stage_2(*t.app_data);
    // vulkan::Singleton::getInstance().process_stage_3(*t.app_data);
    // vulkan::Singleton::getInstance().process_stage_4(*t.app_data);
    // vulkan::Singleton::getInstance().process_stage_5(*t.app_data);

#pragma omp parallel num_threads(4)
    {
      cifar_sparse::omp::process_stage_1(*t.app_data);
      cifar_sparse::omp::process_stage_2(*t.app_data);
      cifar_sparse::omp::process_stage_3(*t.app_data);
      cifar_sparse::omp::process_stage_4(*t.app_data);
    }
    vulkan::Singleton::getInstance().process_stage_6(*t.app_data);
    vulkan::Singleton::getInstance().process_stage_7(*t.app_data);
    vulkan::Singleton::getInstance().process_stage_8(*t.app_data);
    vulkan::Singleton::getInstance().process_stage_9(*t.app_data);
    vulkan::Singleton::getInstance().sync();

    q_B.push(std::move(t));
  }

  // -- check each result
  for (int i = 0; i < 10; ++i) {
    auto arg_max_index =
        cifar_sparse::arg_max(q_B.front().app_data->u_linear_output.data());
    cifar_sparse::print_prediction(arg_max_index);
    q_B.pop();
  }
}

void run_normal() {
  auto mr = vulkan::Singleton::getInstance().get_mr();

  cifar_sparse::AppData app_data(mr);

  vulkan::Singleton::getInstance().process_stage_1(app_data);
  vulkan::Singleton::getInstance().process_stage_2(app_data);
  vulkan::Singleton::getInstance().process_stage_3(app_data);
  vulkan::Singleton::getInstance().process_stage_4(app_data);
  vulkan::Singleton::getInstance().process_stage_5(app_data);
  vulkan::Singleton::getInstance().process_stage_6(app_data);
  vulkan::Singleton::getInstance().process_stage_7(app_data);
  vulkan::Singleton::getInstance().process_stage_8(app_data);
  vulkan::Singleton::getInstance().process_stage_9(app_data);
  vulkan::Singleton::getInstance().sync();

  auto arg_max_index = cifar_sparse::arg_max(app_data.u_linear_output.data());
  cifar_sparse::print_prediction(arg_max_index);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  run_2_stage_queue();

  return 0;
}