#ifndef HY4FLASH_CHAT_TEMPLATE_HPP_
#define HY4FLASH_CHAT_TEMPLATE_HPP_

// Hy4 Flash's chat encoding: the GGUF's `tokenizer.chat_template` (Tencent's
// template), rendered in C++.

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::hy4_flash {

/// One checkpoint's special-token spellings. Every Hy4 special token carries a
/// suffix the checkpoint chose for all of them at once: `:opensource` on the
/// Xylolabs GGUF, `:6124c78e` on Tencent's trunk. The template builds each one
/// from a single `HYTK` variable, so the runner reads that suffix from the GGUF
/// and the renderer, the tool-call scanner and the tool dialect all take this
/// set. Spelling a turn with another checkpoint's suffix writes ordinary text
/// where the model expects a control token, and reads back a call block that
/// never matches.
struct Hy4Tokens {
  /// The suffix itself, leading colon included: `:opensource`.
  std::string suffix;
  std::string start;
  std::string middle;
  std::string end;
  std::string think_open;
  std::string think_close;
  std::string calls_open;
  std::string calls_close;
  std::string call_open;
  std::string call_close;
  std::string key_open;
  std::string key_close;
  std::string value_open;
  std::string value_close;
  std::string response_open;
  std::string response_close;
  std::string reasoning_mode;
  std::string image_start;
  std::string image_pad;
  std::string image_end;
  /// `calls_open` and `call_open` together: a complete call header, which ends
  /// a constrained reasoning phase and is what the dialect reports as its
  /// reasoning marker.
  std::string call_header;

  /// The template's `HYTK` for these spellings, from the `start` token.
  [[nodiscard]] static std::string SuffixOf(std::string_view start);
};

/// The spellings of one suffix, `:opensource` style or empty for none.
[[nodiscard]] Hy4Tokens MakeHy4Tokens(std::string_view suffix);

/// The Xylolabs spellings: the dialect the extension registers at startup, and
/// the goldens in tests/fixtures/chat_goldens.json.
[[nodiscard]] const Hy4Tokens& OpensourceHy4Tokens();

/// The suffix `reader`'s checkpoint spells its tokens with: the `HYTK` its
/// `tokenizer.chat_template` sets, or, when the template names none, the
/// suffix of its `hy_start` vocabulary token. Empty when the GGUF has neither,
/// which is a file this engine cannot prompt.
[[nodiscard]] std::string Hy4TokenSuffix(const core::GgufReader& reader);

/// The template's `reasoning_effort`: `no_think`, `high` or `max`.
[[nodiscard]] std::string_view ReasoningEffortName(const ReasoningOptions& reasoning);

/// A byte range of a rendered prompt that holds request text: message
/// content, reasoning, tool definitions, call names and arguments. Its
/// control-token spellings are text, not control tokens.
struct ContentSpan {
  std::size_t offset = 0;
  std::size_t size = 0;
};

/// Renders `request` as the template does with add_generation_prompt: the
/// prompt ends inside an open assistant turn (after `<think>` when thinking).
/// `developer` messages render as system messages. An image part renders as
/// its start, pad and end tokens; with `image_offsets` it renders as nothing
/// and its byte offset in the result is appended instead. `content_spans`
/// receives the request text's ranges in the result, in order.
/// `stable_prefix_bytes` receives how much of the result a following user
/// turn keeps; it falls after a special token, so its tokens are a prefix of
/// the result's.
[[nodiscard]] std::string RenderChat(
    const Hy4Tokens& tokens, const ext::ChatRequest& request,
    std::vector<std::size_t>* image_offsets = nullptr,
    std::vector<ContentSpan>* content_spans = nullptr,
    std::size_t* stable_prefix_bytes = nullptr);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_CHAT_TEMPLATE_HPP_
