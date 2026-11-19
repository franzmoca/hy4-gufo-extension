#ifndef HY4FLASH_CHAT_TEMPLATE_HPP_
#define HY4FLASH_CHAT_TEMPLATE_HPP_

// Hy4 Flash's chat encoding: the GGUF's `tokenizer.chat_template` (Tencent's
// template with the `:opensource` token suffix), rendered in C++.

#include <cstddef>
#include <gufo/ext/text.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::hy4_flash {

/// Special-token spellings the template writes.
struct Hy4Tokens {
  static constexpr std::string_view kStart = "<｜hy_start:opensource｜>";
  static constexpr std::string_view kMiddle = "<｜hy_middle:opensource｜>";
  static constexpr std::string_view kEnd = "<｜hy_end:opensource｜>";
  static constexpr std::string_view kThinkOpen = "<think:opensource>";
  static constexpr std::string_view kThinkClose = "</think:opensource>";
  static constexpr std::string_view kCallsOpen = "<tool_calls:opensource>";
  static constexpr std::string_view kCallsClose = "</tool_calls:opensource>";
  static constexpr std::string_view kCallOpen = "<tool_call:opensource>";
  static constexpr std::string_view kCallClose = "</tool_call:opensource>";
  static constexpr std::string_view kKeyOpen = "<arg_key:opensource>";
  static constexpr std::string_view kKeyClose = "</arg_key:opensource>";
  static constexpr std::string_view kValueOpen = "<arg_value:opensource>";
  static constexpr std::string_view kValueClose = "</arg_value:opensource>";
  static constexpr std::string_view kResponseOpen = "<tool_response:opensource>";
  static constexpr std::string_view kResponseClose = "</tool_response:opensource>";
  static constexpr std::string_view kReasoningMode = "<｜reasoning_mode:opensource｜>";
  static constexpr std::string_view kImageStart =
      "<｜hy_image▁start:opensource｜>";
  static constexpr std::string_view kImagePad = "<｜hy_image▁pad:opensource｜>";
  static constexpr std::string_view kImageEnd = "<｜hy_image▁end:opensource｜>";
};

/// The template's `reasoning_effort`: `no_think`, `high` or `max`.
[[nodiscard]] std::string_view ReasoningEffortName(const ReasoningOptions& reasoning);

/// Renders `request` as the template does with add_generation_prompt: the
/// prompt ends inside an open assistant turn (after `<think>` when thinking).
/// `developer` messages render as system messages. An image part renders as
/// its start, pad and end tokens; with `image_offsets` it renders as nothing
/// and its byte offset in the result is appended instead.
[[nodiscard]] std::string RenderChat(
    const ext::ChatRequest& request,
    std::vector<std::size_t>* image_offsets = nullptr);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_CHAT_TEMPLATE_HPP_
