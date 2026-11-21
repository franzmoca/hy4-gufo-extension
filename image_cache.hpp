#ifndef HY4FLASH_IMAGE_CACHE_HPP_
#define HY4FLASH_IMAGE_CACHE_HPP_

// Prepared images kept across requests: a conversation sends every earlier
// image again on each turn, and decoding, resizing and hashing them again
// costs 20 to 170 ms per image per turn.

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/image.hpp>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

namespace gufo::models::hy4_flash {

/// An image part's bytes decoded and resized as prompts do, with the SHA-256
/// of the resized pixels.
struct PreparedImage {
  core::Image pixels;  ///< resized RGB8, sides multiples of 32
  std::array<std::uint8_t, 32> digest{};
};

/// Decodes, resizes and hashes an image (PrepareImage), or returns the result
/// for the same bytes and detail from an earlier call. Entries are matched on
/// the exact bytes, so a hash collision only costs a comparison. The budget
/// counts the retained pixels and encoded bytes, least recently used first
/// out; evicted images stay alive in the prompts that hold them. Thread-safe;
/// decoding runs outside the lock.
class ImageCache {
public:
  static constexpr std::size_t kDefaultCapacityBytes = 256u << 20;
  explicit ImageCache(std::size_t capacity_bytes = kDefaultCapacityBytes)
      : capacity_bytes_(capacity_bytes) {}

  struct Stats {
    std::size_t entries = 0;
    std::size_t bytes = 0;
    std::size_t hits = 0;
    std::size_t misses = 0;
    std::size_t evictions = 0;
  };

  /// Throws std::invalid_argument for unreadable images.
  [[nodiscard]] std::shared_ptr<const PreparedImage> Get(
      const std::shared_ptr<const std::vector<std::uint8_t>>& encoded,
      bool low_detail);
  [[nodiscard]] Stats GetStats() const;

private:
  struct Entry {
    std::size_t key;
    bool low_detail;
    std::shared_ptr<const std::vector<std::uint8_t>> encoded;
    std::shared_ptr<const PreparedImage> image;
    std::size_t bytes;
  };
  std::list<Entry>::iterator Find(std::size_t key,
                                  const std::vector<std::uint8_t>& encoded,
                                  bool low_detail);

  const std::size_t capacity_bytes_;
  mutable std::mutex mutex_;
  std::list<Entry> entries_;  // most recently used first
  Stats stats_;
};

/// PrepareImage and the digest, without a cache.
[[nodiscard]] std::shared_ptr<const PreparedImage> PrepareAndHash(
    const std::vector<std::uint8_t>& encoded, bool low_detail);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_IMAGE_CACHE_HPP_
