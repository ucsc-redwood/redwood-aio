#pragma once

#include <iostream>
#include <vector>

#include "conf.hpp"
#include "third-party/CLI11.hpp"

inline std::string g_device_id;
inline std::string g_spdlog_log_level;
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

inline int parse_args(int argc, char** argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", g_device_id, "Device ID")->required();
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info");
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (g_device_id.empty()) {
    throw std::runtime_error("Device ID is required");
  }

  auto& registry = GlobalDeviceRegistry();

  try {
    const Device& device = registry.getDevice(g_device_id);

    auto littleCores = device.getCores(ProcessorType::kLittleCore);
    auto mediumCores = device.getCores(ProcessorType::kMediumCore);
    auto bigCores = device.getCores(ProcessorType::kBigCore);

    std::cout << "Little cores: ";
    for (const auto& core : littleCores) {
      std::cout << core.id << " ";
      g_little_cores.push_back(core.id);
    }
    std::cout << std::endl;

    std::cout << "Medium cores: ";
    for (const auto& core : mediumCores) {
      std::cout << core.id << " ";
      g_medium_cores.push_back(core.id);
    }
    std::cout << std::endl;

    std::cout << "Big cores: ";
    for (const auto& core : bigCores) {
      std::cout << core.id << " ";
      g_big_cores.push_back(core.id);
    }
    std::cout << std::endl;

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}