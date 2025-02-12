#pragma once

#include <bitset>
#include <stdexcept>
#include <string>
// #include <unordered_map>
#include <vector>

// The order in the cores vector is the order of core types
constexpr int kLittleCoreType = 0;
constexpr int kMediumCoreType = 1;
constexpr int kBigCoreType = 2;

// // define a mapping from std::string (device id) to table index
// // for example, "device_0" -> 0, "device_1" -> 1, etc.
// const static std::unordered_map<std::string, int> device_id_to_index = {
//     {"pc", 0},
//     {"jetson", 1},
//     {"3A021JEHN02756", 2},
//     {"9b034f1b", 3},
//     {"ce0717178d7758b00b7e", 4},
//     {"amd-minipc", 5},
// };

struct Device {
  size_t core_count;
  size_t core_type_count;
  std::bitset<32> pinable_mask;

  // cores[core_type] contains the list of cores of that type
  std::vector<std::vector<int>> cores;

  [[nodiscard]] int get_core_count(const int core_type) const {
    if (core_type >= (int)cores.size()) {
      return 0;
    }
    return cores[core_type].size();
  }

  [[nodiscard]] std::vector<int> get_pinable_cores(int core_type) const {
    if (core_type >= (int)cores.size()) {
      return {};
    }

    std::vector<int> pinable_cores;
    // Check each core in the requested core_type group
    for (int core : cores[core_type]) {
      if (pinable_mask[core]) {
        pinable_cores.push_back(core);
      }
    }
    return pinable_cores;
  }

  [[nodiscard]] std::vector<int> get_pinable_cores() const {
    std::vector<int> all_cores;
    for (size_t core_type = 0; core_type < cores.size(); ++core_type) {
      const auto pinable_cores = get_pinable_cores(core_type);
      all_cores.insert(all_cores.end(), pinable_cores.begin(), pinable_cores.end());
    }
    return all_cores;
  }
};

inline Device init_pc() {
  Device device;
  device.core_count = 8;
  device.core_type_count = 2;
  device.pinable_mask.set();

  device.cores.push_back({{8, 9, 10, 11}});  // little cores
  device.cores.push_back({{1, 2, 3, 6}});    // medium cores

  return device;
}

// jetson has 6 cores, all are pinable
// All cores are same type
inline Device init_jetson() {
  Device device;
  device.core_count = 6;
  device.core_type_count = 1;
  device.pinable_mask.set();

  device.cores.push_back({0, 1, 2, 3, 4, 5});  // all cores are same type
  return device;
}

// 3A021JEHN02756 has 8 cores, all are pinable
// CPU0–3 	0xd05 	Cortex‑A55 	Efficiency (“little”)
// CPU4–5 	0xd41 	Cortex‑A78 	Big (performance)
// CPU6–7 	0xd44 	Cortex‑X1 	Prime (top performance)
inline Device init_3A021JEHN02756() {
  Device device;
  device.core_count = 8;
  device.core_type_count = 3;
  device.pinable_mask.set();

  device.cores.push_back({0, 1, 2, 3});  // little cores
  device.cores.push_back({4, 5});        // big cores
  device.cores.push_back({6, 7});        // prime core
  return device;
}

// 9b034f1b has 8 cores, 0-4 are pinable
// CPU0–2 	0xd46 	Cortex‑A510 	Efficiency (“little”)
// CPU3–4 	0xd4d 	Cortex‑A7x (mid/high) 	Big (performance) cluster
// CPU5–6 	0xd47 	Cortex‑A7x (mid/high) 	Big (performance) cluster
// CPU7 	0xd4e 	Cortex‑X series 	Prime (top performance) core
//
// but because a lot of them are very much similar,
// let's just use
inline Device init_9b034f1b() {
  Device device;
  device.core_count = 8;
  device.core_type_count = 3;
  device.pinable_mask.set(0);
  device.pinable_mask.set(1);
  device.pinable_mask.set(2);
  device.pinable_mask.set(3);
  device.pinable_mask.set(4);

  device.cores.push_back({0, 1, 2});  // little cores
  device.cores.push_back({3, 4});     // big cores
  device.cores.push_back({5, 6, 7});  // adjusted big cores

  return device;
}

// Samsung Galaxy Note
inline Device init_ce0717178d7758b00b7e() {
  Device device;
  device.core_count = 8;
  device.core_type_count = 2;
  device.pinable_mask.set();

  device.cores.push_back({4, 5, 6, 7});  // little cores
  device.cores.push_back({0, 1, 2, 3});  // big cores

  return device;
}

// CPU: AMD Ryzen 9 7940HS w/ Radeon 780M Graphics (16) @ 5.26 GHz
// GPU: AMD Radeon 780M [Integrated]
// 16 cores, all are pinable
// warp size is 64
inline Device init_amd_minipc() {
  Device device;
  device.core_count = 16;
  device.core_type_count = 1;
  device.pinable_mask.set();

  device.cores.push_back({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});

  return device;
}

inline Device get_device(const std::string& device_id) {
  if (device_id == "pc") {
    return init_pc();
  } else if (device_id == "jetson") {
    return init_jetson();
  } else if (device_id == "3A021JEHN02756") {
    return init_3A021JEHN02756();
  } else if (device_id == "9b034f1b") {
    return init_9b034f1b();
  } else if (device_id == "ce0717178d7758b00b7e") {
    return init_ce0717178d7758b00b7e();
  } else if (device_id == "amd-minipc") {
    return init_amd_minipc();
  } else {
    throw std::runtime_error("Device not found");
  }
}
