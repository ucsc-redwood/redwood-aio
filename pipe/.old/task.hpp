#pragma once

#include <cstdint>

template <typename T>
struct Task {
  uint32_t uid;
  T* appdata;
};

template <typename T>
static Task<T> make_task(T* appdata) {
  static uint32_t uid_counter = 0;
  return Task<T>{uid_counter++, appdata};
}
