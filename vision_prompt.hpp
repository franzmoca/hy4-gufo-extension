#ifndef HY4FLASH_VISION_PROMPT_HPP_
#define HY4FLASH_VISION_PROMPT_HPP_

// Chat prompts with images (docs/PLAN.md, decision 19): each image renders
// as `<｜hy_image▁start｜>`, one pad token per row and `<｜hy_image▁end｜>`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/image.hpp>
#include <gufo/ext/text.hpp>
#include <span>
#include <string_view>
#include <vector>

#include "runtime/hy4_runtime.hpp"
#include "runtime/tokenizer.hpp"

namespace gufo::models::hy4_flash {

/// HY V4 VL's smart resize: sides rounded to multiples of 32, scaled into
/// [min_pixels, max_pixels] keeping the aspect ratio.
struct ImageSize {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};
[[nodiscard]] ImageSize SmartResize(std::uint32_t width, std::uint32_t height,
                                    std::uint64_t min_pixels,
                                    std::uint64_t max_pixels);

struct ImagePrompt {
  struct Image {
    core::Image pixels;  ///< resized RGB8, sides multiples of 32
    ImageSpan span;
    /// Covers every image up to and including this one, with its placement,
    /// the preprocessing version and the encoder identity.
    std::array<std::uint8_t, 32> prefix_identity{};
  };
  std::vector<std::int32_t> tokens;
  std::vector<Image> images;
  /// Tokens a following user turn keeps (RenderChat's stable prefix).
  std::size_t stable_prefix_tokens = 0;
  /// Identity of all images.
  std::vector<std::uint8_t> cache_identity;
  /// Identity of the images starting before `token_count`.
  [[nodiscard]] std::span<const std::uint8_t> IdentityForPrefix(
      std::size_t token_count) const;
};

/// Renders `request`, decodes and resizes its images, and expands each into
/// its rows. Throws std::invalid_argument for unreadable images and
/// std::length_error past `max_context`.
[[nodiscard]] ImagePrompt PrepareImagePrompt(const Tokenizer& tokenizer,
                                             const ext::ChatRequest& request,
                                             std::string_view encoder_identity,
                                             std::uint32_t max_context);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_VISION_PROMPT_HPP_
