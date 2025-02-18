// Auto-generated aggregated header for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756

#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "../task.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_001 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_001";

void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_001_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_001
namespace CifarSparse_schedule_003 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_003";

void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_003_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_003
namespace CifarSparse_schedule_002 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_002";

void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_002_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_002
}  // namespace device_3A021JEHN02756