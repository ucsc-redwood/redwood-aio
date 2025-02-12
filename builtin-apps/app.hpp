#pragma once

#include <vector>

#include "conf.hpp"
#include "third-party/CLI11.hpp"

inline std::string g_device_id;
inline std::vector<int> g_little_cores;
inline std::vector<int> g_medium_cores;
inline std::vector<int> g_big_cores;

[[nodiscard]] inline size_t get_vulkan_warp_size() {
  if (g_device_id == "3A021JEHN02756") {
    return 16;
  } else if (g_device_id == "9b034f1b") {
    return 64;
  } else if (g_device_id == "ce0717178d7758b00b7e") {
    return 32;
  } else if (g_device_id == "amd-minipc") {
    return 64;
  } else if (g_device_id == "pc" || g_device_id == "jetson") {
    return 32;
  }
  throw std::runtime_error("Invalid device ID. " + std::string(__FILE__) + ":" +
                           std::to_string(__LINE__));
}

inline int parse_args(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", g_device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (g_device_id.empty()) {
    throw std::runtime_error("Device ID is required");
  }

  auto device = get_device(g_device_id);
  g_little_cores = device.get_pinable_cores(kLittleCoreType);
  g_medium_cores = device.get_pinable_cores(kMediumCoreType);
  g_big_cores = device.get_pinable_cores(kBigCoreType);

  std::cout << "Little cores: ";
  for (auto core : g_little_cores) {
    std::cout << core << " ";
  }
  std::cout << std::endl;

  std::cout << "Mid cores: ";
  for (auto core : g_medium_cores) {
    std::cout << core << " ";
  }
  std::cout << std::endl;

  std::cout << "Big cores: ";
  for (auto core : g_big_cores) {
    std::cout << core << " ";
  }
  std::cout << std::endl;

  return 0;
}