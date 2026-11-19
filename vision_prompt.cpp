#include "vision_prompt.hpp"

#include <algorithm>
#include <cmath>
#include <gufo/ext/sha256.hpp>
#include <stdexcept>
#include <string>

#include "chat_template.hpp"
#include "image_resize.hpp"
#include "runtime/hy4_vision.hpp"

namespace gufo::models::hy4_flash {
namespace {

// processor_config.json's pixel range; `detail: low` caps at 512².
constexpr std::uint64_t kMinPixels = 4096;
constexpr std::uint64_t kMaxPixels = 4194304;
constexpr std::uint64_t kLowDetailPixels = 512 * 512;

void HashU32(crypto::Sha256Hasher& hash, std::uint32_t value) {
  const std::array<std::uint8_t, 4> bytes{
      static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8),
      static_cast<std::uint8_t>(value >> 16),
      static_cast<std::uint8_t>(value >> 24)};
  hash.Update(bytes);
}

void HashText(crypto::Sha256Hasher& hash, std::string_view text) {
  hash.Update(
      {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
}

std::int32_t SingleToken(const Tokenizer& tokenizer,
                         std::string_view spelling) {
  const auto tokens = tokenizer.EncodeWithSpecials(spelling);
  if (tokens.size() != 1)
    throw std::logic_error("Hy4 vocabulary lacks " + std::string(spelling));
  return tokens.front();
}

}  // namespace

ImageSize SmartResize(std::uint32_t width, std::uint32_t height,
                      std::uint64_t min_pixels, std::uint64_t max_pixels) {
  constexpr double factor = VisionEncoder::kFactor;
  if (width == 0 || height == 0)
    throw std::invalid_argument("empty image");
  if (std::max(width, height) / std::min(width, height) > 200) {
    throw std::invalid_argument("image aspect ratio exceeds 200");
  }
  // Python's round: halves to even, as nearbyint in the default mode.
  double h = std::nearbyint(height / factor) * factor;
  double w = std::nearbyint(width / factor) * factor;
  if (h * w > static_cast<double>(max_pixels)) {
    const double beta = std::sqrt(static_cast<double>(height) * width /
                                  static_cast<double>(max_pixels));
    h = std::max(factor, std::floor(height / beta / factor) * factor);
    w = std::max(factor, std::floor(width / beta / factor) * factor);
  } else if (h * w < static_cast<double>(min_pixels)) {
    const double beta = std::sqrt(static_cast<double>(min_pixels) /
                                  (static_cast<double>(height) * width));
    h = std::ceil(height * beta / factor) * factor;
    w = std::ceil(width * beta / factor) * factor;
  }
  return {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h)};
}

std::span<const std::uint8_t> ImagePrompt::IdentityForPrefix(
    std::size_t token_count) const {
  for (auto image = images.rbegin(); image != images.rend(); ++image) {
    if (image->span.first < token_count)
      return image->prefix_identity;
  }
  return {};
}

ImagePrompt PrepareImagePrompt(const Tokenizer& tokenizer,
                               const ext::ChatRequest& request,
                               std::string_view encoder_identity,
                               std::uint32_t max_context) {
  std::vector<std::size_t> offsets;
  const std::string text = RenderChat(request, &offsets);
  const std::int32_t start = SingleToken(tokenizer, Hy4Tokens::kImageStart);
  const std::int32_t pad = SingleToken(tokenizer, Hy4Tokens::kImagePad);
  const std::int32_t end = SingleToken(tokenizer, Hy4Tokens::kImageEnd);

  ImagePrompt prompt;
  const auto append = [&](std::string_view piece) {
    const auto tokens = tokenizer.EncodeWithSpecials(piece);
    if (tokens.size() > max_context - std::min<std::size_t>(
                                          max_context, prompt.tokens.size())) {
      throw std::length_error(
          "prompt exceeds the " + std::to_string(max_context) +
          "-token context; increase --context or shorten the conversation");
    }
    prompt.tokens.insert(prompt.tokens.end(), tokens.begin(), tokens.end());
  };
  crypto::Sha256Hasher identity;
  HashText(identity, "hy4-image-bicubic-rgb8-f32-v1");
  HashText(identity, encoder_identity);

  // Image parts in render order: messages in order, each by offset.
  std::vector<const ext::ChatMessage::ImagePart*> parts;
  for (const auto& message : request.messages) {
    std::vector<const ext::ChatMessage::ImagePart*> own;
    for (const auto& part : message.images)
      own.push_back(&part);
    std::ranges::stable_sort(own, {},
                             [](const auto* part) { return part->offset; });
    parts.insert(parts.end(), own.begin(), own.end());
  }
  if (parts.size() != offsets.size())
    throw std::logic_error("Hy4 image rendering lost a part");

  std::size_t cursor = 0;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    append(std::string_view(text).substr(cursor, offsets[i] - cursor));
    cursor = offsets[i];
    if (!parts[i]->bytes)
      throw std::invalid_argument("image part has no data");
    const core::Image decoded = core::DecodeImage(*parts[i]->bytes);
    const std::uint64_t max_pixels =
        parts[i]->detail == ext::ChatMessage::ImageDetail::kLow
            ? kLowDetailPixels
            : kMaxPixels;
    const ImageSize size =
        SmartResize(decoded.width, decoded.height, kMinPixels, max_pixels);
    core::Image pixels = ResizeBicubic(decoded, size.width, size.height);
    const ImageSpan span{static_cast<std::uint32_t>(prompt.tokens.size() + 1),
                         size.height / VisionEncoder::kFactor,
                         size.width / VisionEncoder::kFactor};
    if (std::size_t{span.Rows()} + 2 >
        max_context -
            std::min<std::size_t>(max_context, prompt.tokens.size())) {
      throw std::length_error("image tokens exceed the model context");
    }
    prompt.tokens.push_back(start);
    prompt.tokens.insert(prompt.tokens.end(), span.Rows(), pad);
    prompt.tokens.push_back(end);
    HashU32(identity, span.first);
    HashU32(identity, pixels.width);
    HashU32(identity, pixels.height);
    identity.Update(pixels.pixels);
    prompt.images.push_back({std::move(pixels), span, identity.Digest()});
  }
  append(std::string_view(text).substr(cursor));
  const auto digest = identity.Finish();
  prompt.cache_identity.assign(digest.begin(), digest.end());
  return prompt;
}

}  // namespace gufo::models::hy4_flash
