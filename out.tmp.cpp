// =============================================================================
// AUTOMATICALLY GENERATED BENCHMARK CODE
// =============================================================================

// -----------------------------------------------------------------------------
// Schedule 001: 3A021JEHN02756_CifarDense_schedule_022
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_022(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kLittleCore, 4>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_022)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 002: 3A021JEHN02756_CifarDense_schedule_043
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_043(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_043)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 003: 3A021JEHN02756_CifarDense_schedule_013
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_013(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_013)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 004: 3A021JEHN02756_CifarDense_schedule_002
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_002(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_002)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 005: 3A021JEHN02756_CifarDense_schedule_031
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_031(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 8>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_031)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 006: 3A021JEHN02756_CifarDense_schedule_040
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_040(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kBigCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kLittleCore, 4>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_040)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 007: 3A021JEHN02756_CifarDense_schedule_001
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_001(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 1, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_0_1, &q_1_2, omp::run_multiple_stages<2, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t3(
        [&]() { chunk<Task, cifar_dense::AppData>(q_1_2, &q_2_3, vulkan::run_gpu_stages<3, 7>); });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<8, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_001)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 008: 3A021JEHN02756_CifarDense_schedule_007
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_007(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<8, 8, ProcessorType::kMediumCore, 2>);
    });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_007)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 009: 3A021JEHN02756_CifarDense_schedule_047
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 3
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_047(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kMediumCore, 2>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 8>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kBigCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_047)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

// -----------------------------------------------------------------------------
// Schedule 010: 3A021JEHN02756_CifarDense_schedule_008
// Device: 3A021JEHN02756
// Application: CifarDense
// Chunks: 4
// -----------------------------------------------------------------------------

static void BM_schedule_3A021JEHN02756_CifarDense_schedule_008(benchmark::State &state) {
  constexpr size_t num_tasks = 20;

  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

  // Preallocate data for all tasks
  auto preallocated_data = init_appdata<cifar_dense::AppData>(mr, num_tasks);

  // Track individual task times
  std::vector<double> task_times;
  task_times.reserve(num_tasks);

  for (auto _ : state) {
    state.PauseTiming();
    moodycamel::ConcurrentQueue<Task *> q_input = init_tasks(preallocated_data);

    auto start_time = std::chrono::high_resolution_clock::now();
    state.ResumeTiming();

    // ---------------------------------------------------------------------
    // Automatically generated from schedule JSON

    // Queue definitions:
    moodycamel::ConcurrentQueue<Task *> q_0_1;
    moodycamel::ConcurrentQueue<Task *> q_1_2;
    moodycamel::ConcurrentQueue<Task *> q_2_3;

    // Thread calls:
    std::thread t1([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_input, &q_0_1, omp::run_multiple_stages<1, 2, ProcessorType::kLittleCore, 4>);
    });
    std::thread t2(
        [&]() { chunk<Task, cifar_dense::AppData>(q_0_1, &q_1_2, vulkan::run_gpu_stages<3, 7>); });
    std::thread t3([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_1_2, &q_2_3, omp::run_multiple_stages<8, 8, ProcessorType::kBigCore, 2>);
    });
    std::thread t4([&]() {
      chunk<Task, cifar_dense::AppData>(
          q_2_3, nullptr, omp::run_multiple_stages<9, 9, ProcessorType::kMediumCore, 2>);
    });

    // Thread joins:
    t1.join();
    t2.join();
    t3.join();
    t4.join();

    // ---------------------------------------------------------------------

    state.PauseTiming();
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    task_times.push_back(elapsed / num_tasks);
    state.ResumeTiming();
  }

  // Calculate and report the actual average time per task
  double avg_task_time =
      std::accumulate(task_times.begin(), task_times.end(), 0.0) / task_times.size();
  state.counters["avg_time_per_task"] = avg_task_time;
}

BENCHMARK(BM_schedule_3A021JEHN02756_CifarDense_schedule_008)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);
