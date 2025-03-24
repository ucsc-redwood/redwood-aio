#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include "../templates.hpp"
#include "../templates_vk.hpp"
#include "benchmarks/argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"
#include "run_stages.hpp"
#include "task.hpp"

// =============================================================================
// AUTOMATICALLY GENERATED BENCHMARK CODE
// =============================================================================

namespace device_9b034f1b {

  
}  // namespace device_9b034f1b

// Extract benchmark registration logic outside of device namespaces
namespace benchmark_registration {

// Define a type for benchmark registration functions
using BenchmarkFunc = void (*)(benchmark::State &);

// Map of device IDs to their schedule IDs
const std::unordered_map<std::string, std::vector<int>> device_schedule_ids = {
    {"9b034f1b", {22, 43, 13, 2, 31, 40, 1, 7, 47, 8}},
    {"3A021JEHN02756", {22, 43, 13, 2, 31, 40, 1, 7, 47, 8}}};

// Create a hash table mapping device IDs and schedule IDs to their benchmark functions and names
const std::unordered_map<std::string,
                         std::unordered_map<int, std::pair<BenchmarkFunc, std::string>>>
    benchmark_map = {
        {"9b034f1b",
         {{22,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_022,
            "BM_schedule_9b034f1b_CifarSparse_schedule_022"}},
          {43,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_043,
            "BM_schedule_9b034f1b_CifarSparse_schedule_043"}},
          {13,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_013,
            "BM_schedule_9b034f1b_CifarSparse_schedule_013"}},
          {2,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_002,
            "BM_schedule_9b034f1b_CifarSparse_schedule_002"}},
          {31,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_031,
            "BM_schedule_9b034f1b_CifarSparse_schedule_031"}},
          {40,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_040,
            "BM_schedule_9b034f1b_CifarSparse_schedule_040"}},
          {1,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_001,
            "BM_schedule_9b034f1b_CifarSparse_schedule_001"}},
          {7,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_007,
            "BM_schedule_9b034f1b_CifarSparse_schedule_007"}},
          {47,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_047,
            "BM_schedule_9b034f1b_CifarSparse_schedule_047"}},
          {8,
           {&device_9b034f1b::BM_schedule_9b034f1b_CifarSparse_schedule_008,
            "BM_schedule_9b034f1b_CifarSparse_schedule_008"}}}},
        {"3A021JEHN02756",
         {{22,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_022,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_022"}},
          {43,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_043,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_043"}},
          {13,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_013,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_013"}},
          {2,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_002,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_002"}},
          {31,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_031,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_031"}},
          {40,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_040,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_040"}},
          {1,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_001,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_001"}},
          {7,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_007,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_007"}},
          {47,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_047,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_047"}},
          {8,
           {&device_3A021JEHN02756::BM_schedule_3A021JEHN02756_CifarSparse_schedule_008,
            "BM_schedule_3A021JEHN02756_CifarSparse_schedule_008"}}}}};

// Helper function to register a single benchmark
void register_single_benchmark(const std::string &device_id, int schedule_id) {
  auto device_it = benchmark_map.find(device_id);
  if (device_it == benchmark_map.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  auto schedule_it = device_it->second.find(schedule_id);
  if (schedule_it == device_it->second.end()) {
    throw std::runtime_error("Invalid schedule ID " + std::to_string(schedule_id) + " for device " +
                             device_id);
  }

  benchmark::RegisterBenchmark(schedule_it->second.second.c_str(), schedule_it->second.first)
      ->Unit(benchmark::kMillisecond)
      ->Iterations(10);
}

// Function to register benchmarks dynamically
void register_benchmarks(const std::string &device_id, int index) {
  auto device_schedule_it = device_schedule_ids.find(device_id);
  if (device_schedule_it == device_schedule_ids.end()) {
    throw std::runtime_error("Invalid device ID: " + device_id);
  }

  const auto &schedule_ids = device_schedule_it->second;

  if (index == -1) {
    // Register all benchmarks for this device
    for (int schedule_id : schedule_ids) {
      register_single_benchmark(device_id, schedule_id);
    }
    return;
  }

  if (index < 0 || index >= static_cast<int>(schedule_ids.size())) {
    throw std::runtime_error("Invalid index for device " + device_id + ". Must be between 0 and " +
                             std::to_string(schedule_ids.size() - 1) + ", or -1 for all");
  }

  register_single_benchmark(device_id, schedule_ids[index]);
}

}  // namespace benchmark_registration

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

int main(int argc, char **argv) {
  PARSE_ARGS_BEGIN;

  int schedule_index = 0;  // Default to first schedule
  app.add_option("-i,--index", schedule_index, "Schedule index (0-9, or -1 for all schedules)")
      ->required();

  PARSE_ARGS_END;

  spdlog::set_level(spdlog::level::off);

  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv);
  benchmark::Initialize(&new_argc, new_argv.data());

  // Register the benchmark based on the device ID and index
  try {
    benchmark_registration::register_benchmarks(g_device_id, schedule_index);
  } catch (const std::runtime_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
