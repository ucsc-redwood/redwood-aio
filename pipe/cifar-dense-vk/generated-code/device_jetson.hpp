// Auto-generated aggregated header for device: jetson
// Contains all 'CifarDense' schedules for device_jetson

#pragma once

#include <concurrentqueue.h>

#include <thread>
#include <vector>

#include "../task.hpp"

namespace device_jetson {

namespace CifarDense_schedule_001 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_001";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_001
namespace CifarDense_schedule_015 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_015";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_015
namespace CifarDense_schedule_018 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_018";

void chunk_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_018
namespace CifarDense_schedule_017 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_017";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_017
namespace CifarDense_schedule_010 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_010";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_010
namespace CifarDense_schedule_005 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_005";

void chunk_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_005
namespace CifarDense_schedule_003 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_003";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_003
namespace CifarDense_schedule_006 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_006";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_006
namespace CifarDense_schedule_013 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_013";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_013
namespace CifarDense_schedule_016 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_016";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_016
namespace CifarDense_schedule_004 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_004";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_004
namespace CifarDense_schedule_008 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_008";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_008
namespace CifarDense_schedule_007 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_007";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_007
namespace CifarDense_schedule_002 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_002";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_002
namespace CifarDense_schedule_011 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_011";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_011
namespace CifarDense_schedule_009 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_009";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_009
namespace CifarDense_schedule_014 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_014";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_014
namespace CifarDense_schedule_012 {

constexpr const char* kScheduleId = "jetson_CifarDense_schedule_012";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_012
}  // namespace device_jetson