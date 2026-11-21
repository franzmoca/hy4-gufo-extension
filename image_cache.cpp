#include "image_cache.hpp"

#include <algorithm>
#include <functional>
#include <gufo/ext/sha256.hpp>
#include <stdexcept>
#include <string_view>

#include "vision_prompt.hpp"

namespace gufo::models::hy4_flash {

std::shared_ptr<const PreparedImage> PrepareAndHash(
    const std::vector<std::uint8_t>& encoded, bool low_detail) {
  auto image = std::make_shared<PreparedImage>();
  image->pixels = PrepareImage(encoded, low_detail);
  crypto::Sha256Hasher hash;
  hash.Update(image->pixels.pixels);
  image->digest = hash.Finish();
  return image;
}

std::list<ImageCache::Entry>::iterator ImageCache::Find(
    std::size_t key, const std::vector<std::uint8_t>& encoded,
    bool low_detail) {
  return std::ranges::find_if(entries_, [&](const Entry& entry) {
    return entry.key == key && entry.low_detail == low_detail &&
           (entry.encoded.get() == &encoded ||
            std::ranges::equal(*entry.encoded, encoded));
  });
}

std::shared_ptr<const PreparedImage> ImageCache::Get(
    const std::shared_ptr<const std::vector<std::uint8_t>>& encoded,
    bool low_detail) {
  if (!encoded)
    throw std::invalid_argument("image part has no data");
  const std::size_t key = std::hash<std::string_view>{}(
      {reinterpret_cast<const char*>(encoded->data()), encoded->size()});
  {
    std::lock_guard lock(mutex_);
    const auto found = Find(key, *encoded, low_detail);
    if (found != entries_.end()) {
      ++stats_.hits;
      entries_.splice(entries_.begin(), entries_, found);
      return found->image;
    }
    ++stats_.misses;
  }
  // Outside the lock, so a new image does not hold up the conversation's
  // earlier ones; two concurrent misses may both decode, one is kept.
  auto image = PrepareAndHash(*encoded, low_detail);
  const std::size_t bytes =
      image->pixels.pixels.capacity() + encoded->capacity();
  if (bytes > capacity_bytes_)
    return image;
  std::lock_guard lock(mutex_);
  const auto found = Find(key, *encoded, low_detail);
  if (found != entries_.end()) {
    entries_.splice(entries_.begin(), entries_, found);
    return found->image;
  }
  while (stats_.bytes > capacity_bytes_ - bytes) {
    stats_.bytes -= entries_.back().bytes;
    entries_.pop_back();
    ++stats_.evictions;
  }
  entries_.push_front({key, low_detail, encoded, image, bytes});
  stats_.bytes += bytes;
  return image;
}

ImageCache::Stats ImageCache::GetStats() const {
  std::lock_guard lock(mutex_);
  Stats stats = stats_;
  stats.entries = entries_.size();
  return stats;
}

}  // namespace gufo::models::hy4_flash
