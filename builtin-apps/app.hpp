#pragma once

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <vector>

#include "conf.hpp"

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

#define PARSE_ARGS_BEGIN CLI::App app{"default"};

// this way we can add app.add_option() before PARSE_ARGS_END to add additional options

#define PARSE_ARGS_END                                                                    \
  app.add_option("-d,--device", g_device_id, "Device ID")->required();                    \
  app.add_option("-l,--log-level", g_spdlog_log_level, "Log level")->default_val("info"); \
  app.allow_extras();                                                                     \
  CLI11_PARSE(app, argc, argv);                                                           \
  if (g_device_id.empty()) {                                                              \
    throw std::runtime_error("Device ID is required");                                    \
  }                                                                                       \
  auto& registry = GlobalDeviceRegistry();                                                \
  try {                                                                                   \
    const Device& device = registry.getDevice(g_device_id);                               \
    auto littleCores = device.getCores(ProcessorType::kLittleCore);                       \
    auto mediumCores = device.getCores(ProcessorType::kMediumCore);                       \
    auto bigCores = device.getCores(ProcessorType::kBigCore);                             \
    std::string little_cores_str;                                                         \
    for (const auto& core : littleCores) {                                                \
      little_cores_str += std::to_string(core.id) + " ";                                  \
      g_little_cores.push_back(core.id);                                                  \
    }                                                                                     \
    spdlog::info("Little cores: {}", little_cores_str);                                   \
    std::string medium_cores_str;                                                         \
    for (const auto& core : mediumCores) {                                                \
      medium_cores_str += std::to_string(core.id) + " ";                                  \
      g_medium_cores.push_back(core.id);                                                  \
    }                                                                                     \
    spdlog::info("Medium cores: {}", medium_cores_str);                                   \
    std::string big_cores_str;                                                            \
    for (const auto& core : bigCores) {                                                   \
      big_cores_str += std::to_string(core.id) + " ";                                     \
      g_big_cores.push_back(core.id);                                                     \
    }                                                                                     \
    spdlog::info("Big cores: {}", big_cores_str);                                         \
  } catch (const std::exception& e) {                                                     \
    std::cerr << e.what() << std::endl;                                                   \
    return 1;                                                                             \
  }

inline int parse_args(int argc, char** argv) {
  PARSE_ARGS_BEGIN
  PARSE_ARGS_END
  return 0;
}