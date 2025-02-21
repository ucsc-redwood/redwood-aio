// Auto-generated aggregated header for device: ce0717178d7758b00b7e
// Contains all 'CifarSparse' schedules for device_ce0717178d7758b00b7e

#pragma once

#include <concurrentqueue.h>

#include <thread>
#include <vector>

#include "../task.hpp"

namespace device_ce0717178d7758b00b7e {

namespace CifarSparse_schedule_005 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_005";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_005
namespace CifarSparse_schedule_003 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_003";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_003
namespace CifarSparse_schedule_002 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_002";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_002
namespace CifarSparse_schedule_013 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_013";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_013
namespace CifarSparse_schedule_006 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_006";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_006
namespace CifarSparse_schedule_014 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_014";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_014
namespace CifarSparse_schedule_020 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_020";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_020
namespace CifarSparse_schedule_011 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_011";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_011
namespace CifarSparse_schedule_015 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_015";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_015
namespace CifarSparse_schedule_019 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_019";

void chunk_chunk1(std::vector<Task>& in_tasks, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_019
namespace CifarSparse_schedule_004 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_004";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_004
namespace CifarSparse_schedule_012 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_012";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_012
namespace CifarSparse_schedule_008 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_008";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_008
namespace CifarSparse_schedule_010 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_010";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_010
namespace CifarSparse_schedule_009 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_009";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_009
namespace CifarSparse_schedule_016 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_016";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_016
namespace CifarSparse_schedule_018 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_018";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_018
namespace CifarSparse_schedule_001 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_001";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q,
                  moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk3(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_001
namespace CifarSparse_schedule_007 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_007";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_007
namespace CifarSparse_schedule_017 {

constexpr const char* kScheduleId = "ce0717178d7758b00b7e_CifarSparse_schedule_017";

void chunk_chunk1(std::vector<Task>& in_tasks, moodycamel::ConcurrentQueue<Task>& out_q);
void chunk_chunk2(moodycamel::ConcurrentQueue<Task>& in_q, std::vector<Task>& out_tasks);

void run_pipeline(std::vector<Task>& tasks, std::vector<Task>& out_tasks);

}  // namespace CifarSparse_schedule_017
}  // namespace device_ce0717178d7758b00b7e