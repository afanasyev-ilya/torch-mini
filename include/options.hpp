#pragma once
#include <memory_resource>

namespace torchmini {

struct Device {
  enum class Kind { CPU /*, CUDA*/ } kind = Kind::CPU;
};

struct TensorOptions {
  Device device{};
  std::pmr::memory_resource* mr = std::pmr::get_default_resource();

  TensorOptions& set_device(Device d) noexcept { device = d; return *this; }
  TensorOptions& set_memory_resource(std::pmr::memory_resource* r) noexcept {
    mr = r; return *this;
  }
};

} // namespace torchmini
