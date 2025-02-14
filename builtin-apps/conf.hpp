#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// 1. Define an enum for core types.
enum class ProcessorType {
  kLittleCore,
  kMediumCore,
  kBigCore,
};

// 2. Define a struct for a Core.
struct Core {
  int id;              // The OS/core id.
  ProcessorType type;  // Type of the core (LITTLE, MEDIUM, BIG).
  bool pinnable;       // Whether this core is available for pinning.
};

// 3. Create a Device class that holds a list of cores.
class Device {
 public:
  // Construct a device with a name and a list of cores.
  Device(std::string name, std::vector<Core> cores)
      : name_(std::move(name)), cores_(std::move(cores)) {}

  // Get all cores.
  const std::vector<Core>& getCores() const { return cores_; }

  // Get all cores of a specific type.
  std::vector<Core> getCores(ProcessorType type) const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.type == type) {
        result.push_back(core);
      }
    }
    return result;
  }

  // Get all pinnable cores (optionally filtered by type).
  std::vector<Core> getPinnableCores(ProcessorType type = ProcessorType::kLittleCore) const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.pinnable && core.type == type) result.push_back(core);
    }
    return result;
  }

  // Get all pinnable cores regardless of type.
  std::vector<Core> getAllPinnableCores() const {
    std::vector<Core> result;
    for (const auto& core : cores_) {
      if (core.pinnable) result.push_back(core);
    }
    return result;
  }

 private:
  std::string name_;
  std::vector<Core> cores_;
};

// 4. Create a device registry for easy lookup.
class DeviceRegistry {
 public:
  DeviceRegistry() {
    // For "pc": 8 cores split into LITTLE and MEDIUM
    devices_.emplace("pc",
                     Device("pc",
                            std::vector<Core>{
                                {8, ProcessorType::kLittleCore, true},
                                {9, ProcessorType::kLittleCore, true},
                                {10, ProcessorType::kLittleCore, true},
                                {11, ProcessorType::kLittleCore, true},
                                {1, ProcessorType::kMediumCore, true},
                                {2, ProcessorType::kMediumCore, true},
                                {3, ProcessorType::kMediumCore, true},
                                {6, ProcessorType::kMediumCore, true},
                            }));

    // For "jetson": 6 cores all of one type.
    devices_.emplace("jetson",
                     Device("jetson",
                            std::vector<Core>{
                                {0, ProcessorType::kLittleCore, true},
                                {1, ProcessorType::kLittleCore, true},
                                {2, ProcessorType::kLittleCore, true},
                                {3, ProcessorType::kLittleCore, true},
                                {4, ProcessorType::kLittleCore, true},
                                {5, ProcessorType::kLittleCore, true},
                            }));

    // For "3A021JEHN02756": 8 cores in 3 groups.
    devices_.emplace("3A021JEHN02756",
                     Device("3A021JEHN02756",
                            std::vector<Core>{
                                {0, ProcessorType::kLittleCore, true},
                                {1, ProcessorType::kLittleCore, true},
                                {2, ProcessorType::kLittleCore, true},
                                {3, ProcessorType::kLittleCore, true},
                                {4, ProcessorType::kMediumCore, true},
                                {5, ProcessorType::kMediumCore, true},
                                {6, ProcessorType::kBigCore, true},
                                {7, ProcessorType::kBigCore, true},
                            }));

    // For "9b034f1b": 8 cores, only cores 0-4 are pinnable.
    devices_.emplace("9b034f1b",
                     Device("9b034f1b",
                            std::vector<Core>{
                                {0, ProcessorType::kLittleCore, true},
                                {1, ProcessorType::kLittleCore, true},
                                {2, ProcessorType::kLittleCore, true},
                                {3, ProcessorType::kMediumCore, true},
                                {4, ProcessorType::kMediumCore, true},
                                {5, ProcessorType::kBigCore, false},
                                {6, ProcessorType::kBigCore, false},
                                {7, ProcessorType::kBigCore, false},
                            }));

    // For "ce0717178d7758b00b7e": 8 cores split into LITTLE and BIG.
    devices_.emplace("ce0717178d7758b00b7e",
                     Device("ce0717178d7758b00b7e",
                            std::vector<Core>{
                                {4, ProcessorType::kLittleCore, true},
                                {5, ProcessorType::kLittleCore, true},
                                {6, ProcessorType::kLittleCore, true},
                                {7, ProcessorType::kLittleCore, true},
                                {0, ProcessorType::kBigCore, true},
                                {1, ProcessorType::kBigCore, true},
                                {2, ProcessorType::kBigCore, true},
                                {3, ProcessorType::kBigCore, true},
                            }));

    // For "amd-minipc": 16 cores all of the same type.
    std::vector<Core> amdCores;
    for (int i = 0; i < 16; ++i) {
      amdCores.push_back({i, ProcessorType::kLittleCore, true});
    }
    devices_.emplace("amd-minipc", Device("amd-minipc", amdCores));
  }

  // Retrieve a device configuration by its id.
  const Device& getDevice(const std::string& deviceId) const {
    auto it = devices_.find(deviceId);
    if (it != devices_.end()) return it->second;
    throw std::runtime_error("Device not found: " + deviceId);
  }

 private:
  std::unordered_map<std::string, Device> devices_;
};

inline DeviceRegistry& GlobalDeviceRegistry() {
  static DeviceRegistry instance;
  return instance;
}
