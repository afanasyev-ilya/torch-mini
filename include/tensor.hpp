#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "options.hpp"
#include "storage.hpp"

namespace torchmini {

struct Shape4 {
  std::size_t N{0}, C{0}, H{0}, W{0};
};

struct Strides4 {
  std::size_t sN{0}, sC{0}, sH{0}, sW{0}; // in elements, row-major by default
};

constexpr Strides4 make_contiguous_strides(const Shape4& sh) noexcept {
  // Row-major (N, C, H, W): last axis stride = 1
  return Strides4{
    /*sN=*/sh.C * sh.H * sh.W,
    /*sC=*/sh.H * sh.W,
    /*sH=*/sh.W,
    /*sW=*/1
  };
}

class Tensor4D {
public:
  Tensor4D() = default;

  // Contiguous allocation (owns through shared Storage)
  [[nodiscard]] static Tensor4D contiguous(Shape4 shape,
                                           const TensorOptions& opt = {}) {
    const auto elems = safe_numel(shape);
    auto st = Storage::make(elems, opt.mr);
    return Tensor4D(std::move(st), shape, make_contiguous_strides(shape), /*offset=*/0);
  }

  // Construct a view into an existing storage (no allocation)
  Tensor4D(Storage storage, Shape4 shape, Strides4 strides, std::size_t storage_offset = 0)
  : storage_(std::move(storage)), shape_(shape), strides_(strides), offset_(storage_offset) {
    if (!storage_) throw std::invalid_argument("Tensor4D: null storage");
    if (numel() == 0) return; // allow empty
    // (M1) loose check: just ensure we don't obviously exceed buffer
    if (offset_ >= storage_.size())
      throw std::out_of_range("Tensor4D: offset beyond storage");
    // We can't easily validate all strided access without views; leave to ops.
  }

  // Observers
  Shape4   shape()   const noexcept { return shape_; }
  Strides4 strides() const noexcept { return strides_; }
  std::size_t numel() const noexcept { return safe_numel(shape_); }
  std::size_t bytes() const noexcept { return numel() * sizeof(float); }

  std::size_t N() const noexcept { return shape_.N; }
  std::size_t C() const noexcept { return shape_.C; }
  std::size_t H() const noexcept { return shape_.H; }
  std::size_t W() const noexcept { return shape_.W; }

  bool is_contiguous() const noexcept {
    return strides_.sN == shape_.C * shape_.H * shape_.W &&
           strides_.sC == shape_.H * shape_.W &&
           strides_.sH == shape_.W &&
           strides_.sW == 1;
  }

  float*       data()       noexcept { return storage_.data() + offset_; }
  const float* data() const noexcept { return storage_.data() + offset_; }

  const Storage& storage() const noexcept { return storage_; }
  Storage&       storage()       noexcept { return storage_; }

  std::size_t storage_offset() const noexcept { return offset_; }

  // Basic mutators utility (handy for tests/demo)
  void fill(float v) noexcept {
    // M1: only safe for contiguous tensors; non-contig fill will come with iterators
    if (!is_contiguous()) return;
    float* p = data();
    for (std::size_t i = 0, n = numel(); i < n; ++i) p[i] = v;
  }

private:
  static constexpr std::size_t safe_numel(const Shape4& sh) noexcept {
    // Very simple product; for overflow-robustness you can add checked mul later.
    return sh.N * sh.C * sh.H * sh.W;
  }

  Storage     storage_{};
  Shape4      shape_{};
  Strides4    strides_{};
  std::size_t offset_{0}; // in elements
};

} // namespace torchmini
