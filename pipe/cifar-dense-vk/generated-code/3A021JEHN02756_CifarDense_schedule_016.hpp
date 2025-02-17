// Auto-generated code for schedule: 3A021JEHN02756_CifarDense_schedule_016
// Device ID: 3A021JEHN02756

#pragma once

#include <vector>
#include "../task.hpp"
#include <concurrentqueue.h>

namespace device_3A021JEHN02756 {
namespace CifarDense_schedule_016 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarDense_schedule_016";

void stage_group_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarDense_schedule_016
}  // namespace device_3A021JEHN02756
