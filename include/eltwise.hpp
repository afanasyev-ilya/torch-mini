#pragma once

#include <span>
#include <vector>
#include <stdexcept>
#include <cstddef>

namespace torch_mini {

// Simple elementwise add: returns C[i] = A[i] + B[i].
// Throws std::invalid_argument if sizes differ.
inline std::vector<float> eltwise_add(std::span<const float> a,
                                      std::span<const float> b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("eltwise_add: size mismatch");
  }
  std::vector<float> out;
  out.resize(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

} // namespace minitorch
