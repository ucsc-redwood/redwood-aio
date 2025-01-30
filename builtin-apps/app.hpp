#pragma once

#include <CLI/CLI.hpp>
#include <vector>

#include "conf.hpp"

inline std::vector<int> g_little_cores;
inline std::vector<int> g_medium_cores;
inline std::vector<int> g_big_cores;

inline int parse_args(int argc, char **argv) {
  std::string device_id;

  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    throw std::runtime_error("Device ID is required");
  }

  auto device = get_device(device_id);
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