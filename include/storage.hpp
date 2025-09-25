#pragma once
#include <cstddef>
#include <memory>
#include <memory_resource>
#include <utility>

namespace torchmini {

class Storage {
  // Inner buffer that actually owns the memory
  struct Buffer {
    using alloc_t = std::pmr::polymorphic_allocator<float>;

    struct Deleter {
      alloc_t alloc{};
      std::size_t n{0}; // number of float elements
      void operator()(float* p) const noexcept {
        if (p) alloc.deallocate(p, n);
      }
    };

    std::size_t size_{0};
    alloc_t     alloc_{std::pmr::get_default_resource()};
    std::unique_ptr<float[], Deleter> ptr_{nullptr, Deleter{}};

    Buffer() = default;

    explicit Buffer(std::size_t n, std::pmr::memory_resource* mr)
    : size_(n), alloc_(mr),
      ptr_(alloc_.allocate(n), Deleter{alloc_, n}) {}

    Buffer(Buffer&&)            = default;
    Buffer& operator=(Buffer&&) = default;

    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    float*       data()       noexcept { return ptr_.get(); }
    const float* data() const noexcept { return ptr_.get(); }
    std::size_t  size() const noexcept { return size_; }
    std::pmr::memory_resource* resource() const noexcept { return alloc_.resource(); }
  };

public:
  Storage() = default;

  // Factory: allocate n floats
  [[nodiscard]] static Storage make(std::size_t n,
                                    std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
    Storage s;
    s.buf_ = std::make_shared<Buffer>(n, mr);
    return s;
  }

  // Observers
  explicit operator bool() const noexcept { return static_cast<bool>(buf_); }
  float*       data()       noexcept { return buf_ ? buf_->data() : nullptr; }
  const float* data() const noexcept { return buf_ ? buf_->data() : nullptr; }
  std::size_t  size() const noexcept { return buf_ ? buf_->size() : 0; }
  bool         unique() const noexcept { return buf_ && buf_.use_count() == 1; }
  std::pmr::memory_resource* resource() const noexcept {
    return buf_ ? buf_->resource() : std::pmr::get_default_resource();
  }

  // Share semantics (copy = share)
  Storage(const Storage&)            = default;
  Storage& operator=(const Storage&) = default;
  Storage(Storage&&)                 = default;
  Storage& operator=(Storage&&)      = default;

private:
  std::shared_ptr<Buffer> buf_{};
};

} // namespace torchmini
