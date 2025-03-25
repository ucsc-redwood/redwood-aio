#pragma once
#include <thread>
#include <chrono>
#include <numeric>
#include "task.hpp"
#include "run_stages.hpp"
#include "../templates.hpp"
#include "../templates_vk.hpp"

// Generating code for GPU = cuda
#include <benchmark/benchmark.h>
#include "../cuda_manager.hpp"

namespace device_jetson {
namespace generated_schedules {
using bench_func_t = void(*)(void);
struct ScheduleRecord {
    const char* name;
    bench_func_t func;
};
} // namespace generated_schedules

static generated_schedules::ScheduleRecord schedule_table[] = {
static void BM_schedule_jetson_CifarDense_schedule_001() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 7>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_001", &BM_schedule_jetson_CifarDense_schedule_001},
static void BM_schedule_jetson_CifarDense_schedule_002() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 8>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_002", &BM_schedule_jetson_CifarDense_schedule_002},
static void BM_schedule_jetson_CifarDense_schedule_003() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<3, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_003", &BM_schedule_jetson_CifarDense_schedule_003},
static void BM_schedule_jetson_CifarDense_schedule_004() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<2, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_004", &BM_schedule_jetson_CifarDense_schedule_004},
static void BM_schedule_jetson_CifarDense_schedule_005() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);


    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, nullptr, cuda::run_multiple_stages<1, 9>, mgr); });

    t1.join();
}

    {"jetson_CifarDense_schedule_005", &BM_schedule_jetson_CifarDense_schedule_005},
static void BM_schedule_jetson_CifarDense_schedule_006() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 3, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<4, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_006", &BM_schedule_jetson_CifarDense_schedule_006},
static void BM_schedule_jetson_CifarDense_schedule_007() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 4, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<5, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_007", &BM_schedule_jetson_CifarDense_schedule_007},
static void BM_schedule_jetson_CifarDense_schedule_008() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<7, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_008", &BM_schedule_jetson_CifarDense_schedule_008},
static void BM_schedule_jetson_CifarDense_schedule_009() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 5, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<6, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_009", &BM_schedule_jetson_CifarDense_schedule_009},
static void BM_schedule_jetson_CifarDense_schedule_010() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 5>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<6, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_010", &BM_schedule_jetson_CifarDense_schedule_010},
static void BM_schedule_jetson_CifarDense_schedule_011() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 6, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<7, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_011", &BM_schedule_jetson_CifarDense_schedule_011},
static void BM_schedule_jetson_CifarDense_schedule_012() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 4>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<5, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_012", &BM_schedule_jetson_CifarDense_schedule_012},
static void BM_schedule_jetson_CifarDense_schedule_013() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 3>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<4, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_013", &BM_schedule_jetson_CifarDense_schedule_013},
static void BM_schedule_jetson_CifarDense_schedule_014() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 2>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<3, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_014", &BM_schedule_jetson_CifarDense_schedule_014},
static void BM_schedule_jetson_CifarDense_schedule_015() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, cuda::run_multiple_stages<1, 1>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, omp::run_multiple_stages<2, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_015", &BM_schedule_jetson_CifarDense_schedule_015},
static void BM_schedule_jetson_CifarDense_schedule_016() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 7, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<8, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_016", &BM_schedule_jetson_CifarDense_schedule_016},
static void BM_schedule_jetson_CifarDense_schedule_017() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);

    moodycamel::ConcurrentQueue<Task*> q_0_1;

    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, &q_0_1, omp::run_multiple_stages<1, 8, ProcessorType::kLittleCore, 6>, mgr); });
    std::thread t2([&]() { chunk<Task, cifar_dense::AppData>(q_0_1, nullptr, cuda::run_multiple_stages<9, 9>, mgr); });

    t1.join();
    t2.join();
}

    {"jetson_CifarDense_schedule_017", &BM_schedule_jetson_CifarDense_schedule_017},
static void BM_schedule_jetson_CifarDense_schedule_018() {
    cuda::CudaManager mgr;
    constexpr size_t num_tasks = 20;
    auto mr = &mgr.get_mr();

    // Preallocate data for all tasks
    auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

    // Initialize input tasks
    moodycamel::ConcurrentQueue<Task*> q_input = init_tasks(preallocated_data);


    std::thread t1([&]() { chunk<Task, cifar_dense::AppData>(q_input, nullptr, omp::run_multiple_stages<1, 9, ProcessorType::kLittleCore, 6>, mgr); });

    t1.join();
}

    {"jetson_CifarDense_schedule_018", &BM_schedule_jetson_CifarDense_schedule_018},
};
static const size_t schedule_count = 18;
} // namespace device_jetson

