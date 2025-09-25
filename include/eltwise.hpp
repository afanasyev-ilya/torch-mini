#pragma once

#include <span>
#include <vector>
#include <stdexcept>
#include <cstddef>

namespace torchmini {


inline void eltwise_add(std::span<const float> a,
                        std::span<const float> b,
                         std::span<float> c) {
  if (a.size() != b.size() || c.size() != a.size()) {
    throw std::invalid_argument("eltwise_add: size mismatch");
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    c[i] = a[i] + b[i];
  }
}

[[nodiscard("Capture the result or use eltwise_add_inplace(a, b)")]]
inline std::vector<float> eltwise_add(std::span<const float> a,
                                      std::span<const float> b) {
  std::vector<float> out;
  out.resize(a.size());
  eltwise_add(a, b, out);
  return out;
}

} // namespace minitorch
