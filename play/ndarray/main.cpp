#include <string>

#include "ndarray.hpp"

int main() {
  // Input: 3×32×32 CIFAR-10 image (channels, height, width)
  NDArray<3> input({3, 32, 32});
  input.print_shape("Input");

  // Conv1: 3 → 16 channels, spatial size remains 32×32
  NDArray<3> conv1_out({16, 32, 32});
  conv1_out.print_shape("Conv1 Output");

  // Pool1: 2×2 window with stride 2 → reduces spatial dimensions to 16×16
  NDArray<3> pool1_out({16, 16, 16});
  pool1_out.print_shape("Pool1 Output");

  // Conv2: 16 → 32 channels, spatial size remains 16×16
  NDArray<3> conv2_out({32, 16, 16});
  conv2_out.print_shape("Conv2 Output");

  // Pool2: 2×2 window with stride 2 → reduces spatial dimensions to 8×8
  NDArray<3> pool2_out({32, 8, 8});
  pool2_out.print_shape("Pool2 Output");

  // Conv3: 32 → 64 channels, spatial size remains 8×8
  NDArray<3> conv3_out({64, 8, 8});
  conv3_out.print_shape("Conv3 Output");

  // Conv4: 64 → 64 channels, spatial size remains 8×8
  NDArray<3> conv4_out({64, 8, 8});
  conv4_out.print_shape("Conv4 Output");

  // Conv5: 64 → 64 channels, spatial size remains 8×8
  NDArray<3> conv5_out({64, 8, 8});
  conv5_out.print_shape("Conv5 Output");

  // Pool3: 2×2 window with stride 2 → reduces spatial dimensions to 4×4
  NDArray<3> pool3_out({64, 4, 4});
  pool3_out.print_shape("Pool3 Output");

  // Linear layer: flatten the final tensor (64 channels × 4×4 = 1024 features)
  // and produce 10 outputs.
  NDArray<1> linear_out({10});
  linear_out.print_shape("Linear Output");

  return 0;
}
