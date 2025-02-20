// Auto-generated aggregated header for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756

#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "../task.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_013 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_013";

void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_013_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_013
namespace CifarSparse_schedule_004 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_004";

void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_004_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_004
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
namespace CifarSparse_schedule_015 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_015";

void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_015_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_015
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
namespace CifarSparse_schedule_008 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_008";

void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_008_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_008
namespace CifarSparse_schedule_014 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_014";

void stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_014_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_014
namespace CifarSparse_schedule_012 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_012";

void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_012_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_012
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
namespace CifarSparse_schedule_005 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_005";

void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_005_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_005
namespace CifarSparse_schedule_007 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_007";

void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_007_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_007
namespace CifarSparse_schedule_006 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_006";

void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_006_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_006
namespace CifarSparse_schedule_010 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_010";

void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_010_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_010
namespace CifarSparse_schedule_011 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_011";

void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_011_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_011
namespace CifarSparse_schedule_009 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_009";

void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk1(
    std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk2(
    moodycamel::ConcurrentQueue<Task>& in_q, moodycamel::ConcurrentQueue<Task>& out_q);
void stage_group_3A021JEHN02756_CifarSparse_schedule_009_chunk3(
    moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_009
}  // namespace device_3A021JEHN02756