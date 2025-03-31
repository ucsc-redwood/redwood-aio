#pragma once

#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

template <size_t ND>
class NDArray {
 public:
  using Shape = std::array<size_t, ND>;

  // Construct the array given its shape.
  explicit NDArray(const Shape& shape) : shape_(shape) {
    compute_strides();
    total_size_ = std::accumulate(shape_.begin(), shape_.end(), size_t(1), std::multiplies<>());
    data_.resize(total_size_, 0.0f);
  }

  // Overloaded operator() for element access.
  template <typename... Indices>
    requires(sizeof...(Indices) == ND)
  float& operator()(Indices... indices) {
    size_t idx = compute_index({static_cast<size_t>(indices)...});
    return data_[idx];
  }

  template <typename... Indices>
    requires(sizeof...(Indices) == ND)
  const float& operator()(Indices... indices) const {
    size_t idx = compute_index({static_cast<size_t>(indices)...});
    return data_[idx];
  }

  // Utility to print the shape.
  void print_shape(const std::string& name) const {
    std::cout << name << ": (";
    for (size_t i = 0; i < ND; ++i) {
      std::cout << shape_[i];
      if (i < ND - 1) std::cout << " × ";
    }
    std::cout << ")\n";
  }

 private:
  const Shape shape_;
  Shape strides_;
  size_t total_size_;

  // TODO: use pmr::vector
  std::vector<float> data_;

  // Compute strides for row-major order.
  void compute_strides() {
    strides_[ND - 1] = 1;
    for (size_t i = ND - 1; i > 0; --i) {
      strides_[i - 1] = strides_[i] * shape_[i];
    }
  }

  // Compute the flat index from multi-dimensional indices.
  // TODO: make this work in both CPU and GPU
  size_t compute_index(const Shape& indices) const {
    size_t idx = 0;
    for (size_t i = 0; i < ND; ++i) {
      idx += indices[i] * strides_[i];
    }
    return idx;
  }
};
