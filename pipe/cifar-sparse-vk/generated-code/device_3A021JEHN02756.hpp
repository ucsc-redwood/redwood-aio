// Auto-generated aggregated header for device: 3A021JEHN02756
// Contains all 'CifarSparse' schedules for device_3A021JEHN02756

#pragma once

#include <concurrentqueue.h>

#include <thread>
#include <vector>

#include "../task.hpp"

namespace device_3A021JEHN02756 {

namespace CifarSparse_schedule_013 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_013";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_013
namespace CifarSparse_schedule_004 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_004";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_004
namespace CifarSparse_schedule_001 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_001";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_001
namespace CifarSparse_schedule_015 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_015";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_015
namespace CifarSparse_schedule_003 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_003";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_003
namespace CifarSparse_schedule_008 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_008";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_008
namespace CifarSparse_schedule_014 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_014";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_014
namespace CifarSparse_schedule_012 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_012";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_012
namespace CifarSparse_schedule_002 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_002";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_002
namespace CifarSparse_schedule_005 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_005";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_005
namespace CifarSparse_schedule_007 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_007";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_007
namespace CifarSparse_schedule_006 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_006";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_006
namespace CifarSparse_schedule_010 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_010";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_010
namespace CifarSparse_schedule_011 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_011";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_011
namespace CifarSparse_schedule_009 {

constexpr const char* kScheduleId = "3A021JEHN02756_CifarSparse_schedule_009";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_009
}  // namespace device_3A021JEHN02756