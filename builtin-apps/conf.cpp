#include "conf.hpp"

DeviceRegistry::DeviceRegistry() {
  // For "pc": 8 P cores and 12 E cores
  devices_.emplace(
      "pc",
      Device("pc",
             std::vector<Core>{
                 {0, ProcessorType::kBigCore, true},     {1, ProcessorType::kBigCore, true},
                 {2, ProcessorType::kBigCore, true},     {3, ProcessorType::kBigCore, true},
                 {4, ProcessorType::kBigCore, true},     {5, ProcessorType::kBigCore, true},
                 {6, ProcessorType::kBigCore, true},     {7, ProcessorType::kBigCore, true},
                 {8, ProcessorType::kLittleCore, true},  {9, ProcessorType::kLittleCore, true},
                 {10, ProcessorType::kLittleCore, true}, {11, ProcessorType::kLittleCore, true},
                 {12, ProcessorType::kLittleCore, true}, {13, ProcessorType::kLittleCore, true},
                 {14, ProcessorType::kLittleCore, true}, {15, ProcessorType::kLittleCore, true},
                 {16, ProcessorType::kLittleCore, true}, {17, ProcessorType::kLittleCore, true},
                 {18, ProcessorType::kLittleCore, true}, {19, ProcessorType::kLittleCore, true},
                 {20, ProcessorType::kLittleCore, true}, {21, ProcessorType::kLittleCore, true},
                 {22, ProcessorType::kLittleCore, true}, {23, ProcessorType::kLittleCore, true},
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

  // For "minipc": 16 cores all of the same type.
  devices_.emplace("minipc",
                   Device("minipc",
                          std::vector<Core>({
                              {0, ProcessorType::kBigCore, true},
                              {1, ProcessorType::kBigCore, true},
                              {2, ProcessorType::kBigCore, true},
                              {3, ProcessorType::kBigCore, true},
                              {4, ProcessorType::kBigCore, true},
                              {5, ProcessorType::kBigCore, true},
                              {6, ProcessorType::kBigCore, true},
                              {7, ProcessorType::kBigCore, true},
                              {8, ProcessorType::kBigCore, true},
                              {9, ProcessorType::kBigCore, true},
                              {10, ProcessorType::kBigCore, true},
                              {11, ProcessorType::kBigCore, true},
                              {12, ProcessorType::kBigCore, true},
                              {13, ProcessorType::kBigCore, true},
                              {14, ProcessorType::kBigCore, true},
                              {15, ProcessorType::kBigCore, true},
                          })));
}