// Auto-generated aggregated header for device: 9b034f1b
// Contains all 'CifarDense' schedules for device_9b034f1b

#pragma once

#include <concurrentqueue.h>

#include <queue>
#include <thread>
#include <vector>

#include "../task.hpp"

namespace device_9b034f1b {

namespace CifarDense_schedule_022 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_022";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_022
namespace CifarDense_schedule_048 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_048";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_048
namespace CifarDense_schedule_033 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_033";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_033
namespace CifarDense_schedule_042 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_042";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_042
namespace CifarDense_schedule_028 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_028";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_028
namespace CifarDense_schedule_023 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_023";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_023
namespace CifarDense_schedule_046 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_046";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_046
namespace CifarDense_schedule_041 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_041";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_041
namespace CifarDense_schedule_012 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_012";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_012
namespace CifarDense_schedule_006 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_006";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_006
namespace CifarDense_schedule_019 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_019";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_019
namespace CifarDense_schedule_029 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_029";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_029
namespace CifarDense_schedule_044 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_044";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_044
namespace CifarDense_schedule_024 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_024";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_024
namespace CifarDense_schedule_047 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_047";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_047
namespace CifarDense_schedule_043 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_043";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_043
namespace CifarDense_schedule_050 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_050";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_050
namespace CifarDense_schedule_039 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_039";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_039
namespace CifarDense_schedule_040 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_040";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_040
namespace CifarDense_schedule_003 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_003";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_003
namespace CifarDense_schedule_016 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_016";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_016
namespace CifarDense_schedule_026 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_026";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_026
namespace CifarDense_schedule_030 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_030";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_030
namespace CifarDense_schedule_038 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_038";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_038
namespace CifarDense_schedule_015 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_015";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_015
namespace CifarDense_schedule_027 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_027";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_027
namespace CifarDense_schedule_045 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_045";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_045
namespace CifarDense_schedule_009 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_009";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_009
namespace CifarDense_schedule_008 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_008";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_008
namespace CifarDense_schedule_005 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_005";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_005
namespace CifarDense_schedule_031 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_031";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_031
namespace CifarDense_schedule_013 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_013";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_013
namespace CifarDense_schedule_010 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_010";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_010
namespace CifarDense_schedule_001 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_001";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_001
namespace CifarDense_schedule_049 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_049";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_049
namespace CifarDense_schedule_018 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_018";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_018
namespace CifarDense_schedule_014 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_014";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_014
namespace CifarDense_schedule_021 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_021";

void chunk_chunk1(std::queue<Task>& in_tasks, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_021
namespace CifarDense_schedule_037 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_037";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_037
namespace CifarDense_schedule_036 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_036";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_036
namespace CifarDense_schedule_007 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_007";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_007
namespace CifarDense_schedule_004 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_004";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_004
namespace CifarDense_schedule_034 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_034";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_034
namespace CifarDense_schedule_002 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_002";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_002
namespace CifarDense_schedule_020 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_020";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_020
namespace CifarDense_schedule_025 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_025";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_025
namespace CifarDense_schedule_017 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_017";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_017
namespace CifarDense_schedule_035 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_035";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_035
namespace CifarDense_schedule_011 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_011";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_011
namespace CifarDense_schedule_032 {

constexpr const char* kScheduleId = "9b034f1b_CifarDense_schedule_032";

void chunk_chunk1(std::queue<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::queue<Task>& out_tasks);

void run_pipeline(std::queue<Task>& tasks, std::queue<Task>& out_tasks);

}  // namespace CifarDense_schedule_032
}  // namespace device_9b034f1b