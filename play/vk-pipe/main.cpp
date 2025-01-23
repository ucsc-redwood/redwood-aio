#include <concurrentqueue.h>
#include <spdlog/spdlog.h>

#include <mutex>

#include "../../builtin-apps/cifar-dense/arg_max.hpp"
#include "../../builtin-apps/cifar-dense/dense_appdata.hpp"
#include "../../builtin-apps/cifar-dense/omp/dense_kernel.hpp"
#include "../../builtin-apps/cifar-dense/vulkan/vk_dispatcher.hpp"
#include "../../builtin-apps/common/vulkan/engine.hpp"
#include "../../pipe/affinity.hpp"
#include "../../pipe/app.hpp"
#include "../../pipe/conf.hpp"

struct Task {
  uint32_t uid;

  cifar_dense::AppData* app_data;

  // std::unique_ptr<cifar_dense::AppData> app_data;
  bool is_done;
};

Task create_task(cifar_dense::AppData* app_data) {
  static uint32_t uid = 0;
  Task task;
  task.uid = uid++;
  task.app_data = app_data;
  task.is_done = false;
  return task;
}

Task create_sentinel_task() {
  Task task;
  task.uid = 0;
  task.app_data = nullptr;
  task.is_done = true;
  return task;
}

void process_stage_A(Task& task) {
  vulkan::Singleton::getInstance().process_stage_1(*task.app_data);
  vulkan::Singleton::getInstance().process_stage_2(*task.app_data);
  vulkan::Singleton::getInstance().process_stage_3(*task.app_data);
  vulkan::Singleton::getInstance().process_stage_4(*task.app_data);
  vulkan::Singleton::getInstance().process_stage_5(*task.app_data);
  // vulkan::Singleton::getInstance().process_stage_6(*task.app_data);
  // vulkan::Singleton::getInstance().process_stage_7(*task.app_data);
  // vulkan::Singleton::getInstance().process_stage_8(*task.app_data);
  vulkan::Singleton::getInstance().sync();
}

void process_stage_B(Task& task) {
#pragma omp parallel
  {
    // cifar_dense::omp::process_stage_1(*task.app_data);
    // cifar_dense::omp::process_stage_2(*task.app_data);
    // cifar_dense::omp::process_stage_3(*task.app_data);
    // cifar_dense::omp::process_stage_4(*task.app_data);
    // cifar_dense::omp::process_stage_5(*task.app_data);

    // cifar_dense::omp::process_stage_6(*task.app_data);
    // cifar_dense::omp::process_stage_7(*task.app_data);
    // cifar_dense::omp::process_stage_8(*task.app_data);
    // cifar_dense::omp::process_stage_9(*task.app_data);
  }
}

// int main(int argc, char** argv) {
//   parse_args(argc, argv);

//   spdlog::set_level(spdlog::level::trace);

//   auto mr = vulkan::Singleton::getInstance().get_mr();

//   std::array<cifar_dense::AppData, 10>
//   app_data_array{cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr),
//                                                       cifar_dense::AppData(mr)};

//   std::queue<Task> q_A;
//   for (uint32_t i = 0; i < 10; ++i) {
//     q_A.push(create_task(&app_data_array[i]));
//   }
//   q_A.push(create_sentinel_task());

//   while (true) {
//     Task t = q_A.front();
//     q_A.pop();
//     if (t.is_done) {
//       break;
//     }

//     std::cout << "Processing task " << t.uid << std::endl;
//     std::cout << "\tAdress: " << t.app_data << std::endl;

//     vulkan::Singleton::getInstance().process_stage_1(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_2(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_3(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_4(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_5(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_6(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_7(*t.app_data);
//     vulkan::Singleton::getInstance().sync();
//     vulkan::Singleton::getInstance().process_stage_8(*t.app_data);
//     vulkan::Singleton::getInstance().sync();

//     // Broken
//     // vulkan::Singleton::getInstance().process_stage_9(*t.app_data);
//     // vulkan::Singleton::getInstance().sync();

//     auto arg_max_index = arg_max(t.app_data->u_linear_out.data());
//     print_prediction(arg_max_index);
//   }

//   return 0;
// }

// int main(int argc, char** argv) {
//   parse_args(argc, argv);

//   spdlog::set_level(spdlog::level::trace);

//   auto mr = vulkan::Singleton::getInstance().get_mr();

//   cifar_dense::AppData app_data(mr);

//   vulkan::Singleton::getInstance().process_stage_1(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_2(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_3(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_4(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_5(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_6(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_7(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_8(app_data);
//   vulkan::Singleton::getInstance().sync();
//   vulkan::Singleton::getInstance().process_stage_9(app_data);
//   vulkan::Singleton::getInstance().sync();

//   auto arg_max_index = arg_max(app_data.u_linear_out.data());
//   print_prediction(arg_max_index);

//   return 0;
// }

int main(int argc, char** argv) {
  parse_args(argc, argv);

  //   run_2_stages();
  spdlog::set_level(spdlog::level::trace);

  auto mr = vulkan::Singleton::getInstance().get_mr();

  constexpr auto n_tasks = 10;
  std::vector<cifar_dense::AppData> app_data_vec;
  for (uint32_t i = 0; i < n_tasks; ++i) {
    app_data_vec.push_back(cifar_dense::AppData(mr));
  }

  std::queue<Task> q_A;
  moodycamel::ConcurrentQueue<Task> q_AB(n_tasks + 1);
  std::queue<Task> q_B;

  for (uint32_t i = 0; i < n_tasks; ++i) {
    q_A.push(create_task(&app_data_vec[i]));
  }
  q_A.push(create_sentinel_task());

  std::mutex mutex;

  std::thread t_A([&]() {
    while (true) {
      Task t = std::move(q_A.front());
      q_A.pop();
      if (t.is_done) {
        q_AB.enqueue(t);
        break;
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "[A] processing task " << t.uid << std::endl;
      }
      process_stage_A(t);

      q_AB.enqueue(t);
    }
  });

  std::thread t_B([&]() {
    while (true) {
      Task t;
      if (q_AB.try_dequeue(t)) {
        if (t.is_done) {
          break;
        }
        {
          std::lock_guard<std::mutex> lock(mutex);
          std::cout << "[B] processing task " << t.uid << std::endl;
        }
        process_stage_B(t);

        q_B.push(t);
      }
    }
  });

  t_A.join();
  t_B.join();

  while (q_B.size() > 0) {
    Task t = std::move(q_B.front());
    q_B.pop();
    auto arg_max_index = arg_max(t.app_data->u_linear_out.data());
    print_prediction(arg_max_index);
  }

  return 0;
}
