// SmartResize against Transformers' smart_resize (HY V4 VL, factor 32):
// Python's half-to-even rounding, the floor branch above max_pixels and the
// ceil branch below min_pixels; and the image span row layout.
#include "vision_prompt.hpp"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

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

  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 vision prompt: all checks passed\n");
  return 0;
}
