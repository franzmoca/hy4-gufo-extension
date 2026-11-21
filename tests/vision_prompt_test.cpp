// SmartResize against Transformers' smart_resize (HY V4 VL, factor 32):
// Python's half-to-even rounding, the floor branch above max_pixels and the
// ceil branch below min_pixels; the image span row layout; and the prepared
// image cache: matched on exact bytes and detail, bounded, least recently
// used out.
#include "vision_prompt.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "image_cache.hpp"
#include "runtime/hy4_runtime.hpp"

namespace {

namespace hy4 = gufo::models::hy4_flash;

int g_failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

bool Throws(std::uint32_t width, std::uint32_t height) {
  try {
    (void)hy4::SmartResize(width, height, 4096, 4194304);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

// A width x height RGB8 PNG with stored (uncompressed) deflate blocks, its
// pixels a function of `seed`.
std::shared_ptr<const std::vector<std::uint8_t>> Png(std::uint32_t width,
                                                     std::uint32_t height,
                                                     std::uint8_t seed) {
  std::vector<std::uint8_t> raw;
  for (std::uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);  // filter: none
    for (std::uint32_t x = 0; x < width * 3; ++x)
      raw.push_back(static_cast<std::uint8_t>(seed + x * 7 + y * 13));
  }
  std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  const auto u32 = [](std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(v >> shift));
  };
  const auto chunk = [&](const char* type,
                         const std::vector<std::uint8_t>& data) {
    u32(png, static_cast<std::uint32_t>(data.size()));
    std::vector<std::uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    std::uint32_t crc = 0xffffffffu;
    for (const std::uint8_t byte : body) {
      crc ^= byte;
      for (int k = 0; k < 8; ++k)
        crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
    }
    png.insert(png.end(), body.begin(), body.end());
    u32(png, ~crc);
  };
  std::vector<std::uint8_t> header;
  u32(header, width);
  u32(header, height);
  header.insert(header.end(), {8, 2, 0, 0, 0});  // 8-bit RGB
  chunk("IHDR", header);
  std::vector<std::uint8_t> z{0x78, 0x01};
  std::uint32_t a = 1, b = 0;
  for (const std::uint8_t byte : raw) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  for (std::size_t at = 0; at < raw.size(); at += 65535) {
    const auto n = static_cast<std::uint16_t>(
        std::min<std::size_t>(65535, raw.size() - at));
    z.push_back(at + n == raw.size() ? 1 : 0);
    z.insert(
        z.end(),
        {static_cast<std::uint8_t>(n), static_cast<std::uint8_t>(n >> 8),
         static_cast<std::uint8_t>(~n), static_cast<std::uint8_t>(~n >> 8)});
    z.insert(z.end(), raw.begin() + at, raw.begin() + at + n);
  }
  u32(z, (b << 16) | a);
  chunk("IDAT", z);
  chunk("IEND", {});
  return std::make_shared<const std::vector<std::uint8_t>>(std::move(png));
}

void TestImageCache() {
  const auto first = Png(96, 64, 1);
  const auto second = Png(96, 64, 2);
  const auto uncached = hy4::PrepareAndHash(*first, false);
  Expect(uncached->pixels.width == 96 && uncached->pixels.height == 64,
         "PNG decodes at its size");

  // One entry is 96*64*3 pixel bytes plus the PNG; room for two.
  const std::size_t entry = 96 * 64 * 3 + first->capacity();
  hy4::ImageCache cache(2 * entry + entry / 2);
  const auto a = cache.Get(first, false);
  Expect(a->pixels.pixels == uncached->pixels.pixels &&
             a->digest == uncached->digest,
         "cached image equals the uncached one");
  Expect(cache.Get(first, false) == a, "same bytes: hit");
  const auto copy = std::make_shared<const std::vector<std::uint8_t>>(*first);
  Expect(cache.Get(copy, false) == a, "equal bytes in a new buffer: hit");
  Expect(cache.Get(first, true) != a, "low detail is a separate entry");
  auto stats = cache.GetStats();
  Expect(stats.hits == 2 && stats.misses == 2 && stats.entries == 2,
         "two hits, two misses, two entries");

  // A third image evicts the least recently used: `first` at low detail.
  (void)cache.Get(first, false);
  const auto c = cache.Get(second, false);
  Expect(c->digest != a->digest, "different pixels, different digest");
  stats = cache.GetStats();
  Expect(stats.entries == 2 && stats.evictions == 1 && stats.bytes <= 2 * entry,
         "bounded: one eviction");
  Expect(cache.Get(first, false) == a, "recently used entry kept");
  const auto misses = cache.GetStats().misses;
  (void)cache.Get(first, true);
  Expect(cache.GetStats().misses == misses + 1, "evicted entry misses");

  // Larger than the whole budget: returned, not kept.
  hy4::ImageCache tiny(entry / 2);
  Expect(tiny.Get(first, false)->digest == a->digest &&
             tiny.GetStats().entries == 0,
         "oversized image not kept");

  bool threw = false;
  try {
    (void)cache.Get(
        std::make_shared<const std::vector<std::uint8_t>>(3, std::uint8_t{1}),
        false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, "unreadable bytes throw");
}

}  // namespace

int main() {
  // {width, height, max_pixels, want width, want height}, from
  // transformers' smart_resize with min_pixels 4096.
  constexpr std::uint32_t kCases[][5] = {
      {2048, 2048, 4194304, 2048, 2048}, {2875, 1500, 4194304, 2816, 1472},
      {128, 96, 4194304, 128, 96},       {48, 16, 4194304, 128, 64},
      {10, 10, 4194304, 64, 64},         {1000, 1000, 262144, 512, 512},
      {4000, 3000, 262144, 576, 416},    {352, 320, 4194304, 352, 320},
      {80, 80, 4194304, 64, 64},         {16, 3000, 4194304, 32, 896},
      {1920, 1080, 4194304, 1920, 1088},
  };
  for (const auto& c : kCases) {
    const hy4::ImageSize size = hy4::SmartResize(c[0], c[1], 4096, c[2]);
    Expect(size.width == c[3] && size.height == c[4],
           std::to_string(c[0]) + "x" + std::to_string(c[1]) + " gave " +
               std::to_string(size.width) + "x" + std::to_string(size.height));
  }
  Expect(Throws(0, 10), "empty image rejected");
  Expect(Throws(10, 2010), "aspect ratio over 200 rejected");

  // A 2048² image: 64 merged rows of 64 + 1 (the newline) after the start
  // token; its rows end where the end token sits.
  const hy4::ImageSpan span{10, 64, 64};
  Expect(span.Rows() == 4160 && span.End() == 4170, "image span rows");

  TestImageCache();

  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 vision prompt: all checks passed\n");
  return 0;
}
