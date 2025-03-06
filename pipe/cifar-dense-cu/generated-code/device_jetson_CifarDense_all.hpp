// Aggregated schedules for device: jetson, application: CifarDense
#pragma once

#include <queue>
#include <thread>
#include <concurrentqueue.h>
#include "../task.hpp"
#include "../../templates.hpp"  // chunk_first, chunk_middle, chunk_last, chunk_single
#include "../run_stages.hpp"

namespace device_jetson {

namespace schedule_jetson_CifarDense_schedule_001 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 7>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<8, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_001

namespace schedule_jetson_CifarDense_schedule_002 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 8>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<9, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_002

namespace schedule_jetson_CifarDense_schedule_003 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 2, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<3, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_003

namespace schedule_jetson_CifarDense_schedule_004 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 1, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<2, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_004

namespace schedule_jetson_CifarDense_schedule_005 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  std::thread t_only([&]() {
    chunk_single(tasks, out_tasks, run_gpu_stages<1, 9>);
  });
  t_only.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_005

namespace schedule_jetson_CifarDense_schedule_006 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 3, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<4, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_006

namespace schedule_jetson_CifarDense_schedule_007 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 4, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<5, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_007

namespace schedule_jetson_CifarDense_schedule_008 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<7, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_008

namespace schedule_jetson_CifarDense_schedule_009 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 5, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<6, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_009

namespace schedule_jetson_CifarDense_schedule_010 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 5>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<6, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_010

namespace schedule_jetson_CifarDense_schedule_011 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 6, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<7, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_011

namespace schedule_jetson_CifarDense_schedule_012 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 4>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<5, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_012

namespace schedule_jetson_CifarDense_schedule_013 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 3>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<4, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_013

namespace schedule_jetson_CifarDense_schedule_014 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 2>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<3, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_014

namespace schedule_jetson_CifarDense_schedule_015 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_gpu_stages<1, 1>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_cpu_stages<2, 9, ProcessorType::kLittleCore, 6>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_015

namespace schedule_jetson_CifarDense_schedule_016 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 7, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<8, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_016

namespace schedule_jetson_CifarDense_schedule_017 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  moodycamel::ConcurrentQueue<Task> q_0_1;

  std::thread t_chunk1([&]() {
    chunk_first(tasks, q_0_1, run_cpu_stages<1, 8, ProcessorType::kLittleCore, 6>);
  });
  std::thread t_chunk2([&]() {
    chunk_last(q_0_1, out_tasks, run_gpu_stages<9, 9>);
  });

  t_chunk1.join();
  t_chunk2.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_017

namespace schedule_jetson_CifarDense_schedule_018 {

inline void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks)
{
  std::thread t_only([&]() {
    chunk_single(tasks, out_tasks, run_cpu_stages<1, 9, ProcessorType::kLittleCore, 6>);
  });
  t_only.join();
}

}  // namespace schedule_jetson_CifarDense_schedule_018

// --------------------------------------------------------------------------
// Define function pointer type for run_pipeline
using RunPipelineFunc = void (*)(std::queue<Task>&, std::queue<Task>&);

// Array of function pointers to all run_pipeline implementations
static const RunPipelineFunc run_pipeline_table[] = {
    schedule_jetson_CifarDense_schedule_001::run_pipeline,
    schedule_jetson_CifarDense_schedule_002::run_pipeline,
    schedule_jetson_CifarDense_schedule_003::run_pipeline,
    schedule_jetson_CifarDense_schedule_004::run_pipeline,
    schedule_jetson_CifarDense_schedule_005::run_pipeline,
    schedule_jetson_CifarDense_schedule_006::run_pipeline,
    schedule_jetson_CifarDense_schedule_007::run_pipeline,
    schedule_jetson_CifarDense_schedule_008::run_pipeline,
    schedule_jetson_CifarDense_schedule_009::run_pipeline,
    schedule_jetson_CifarDense_schedule_010::run_pipeline,
    schedule_jetson_CifarDense_schedule_011::run_pipeline,
    schedule_jetson_CifarDense_schedule_012::run_pipeline,
    schedule_jetson_CifarDense_schedule_013::run_pipeline,
    schedule_jetson_CifarDense_schedule_014::run_pipeline,
    schedule_jetson_CifarDense_schedule_015::run_pipeline,
    schedule_jetson_CifarDense_schedule_016::run_pipeline,
    schedule_jetson_CifarDense_schedule_017::run_pipeline,
    schedule_jetson_CifarDense_schedule_018::run_pipeline,
};

[[nodiscard]] constexpr int get_num_schedules() {
    return sizeof(run_pipeline_table) / sizeof(run_pipeline_table[0]);
}

}  // namespace device_jetson
