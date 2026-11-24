#ifndef HY4FLASH_RUNTIME_TOKENIZER_HPP_
#define HY4FLASH_RUNTIME_TOKENIZER_HPP_

// Byte-level BPE for Hy4 Flash (`tokenizer.ggml.pre = hy_v4`). The
// pretokenizer is the DeepSeek3/JoyAI one, ported from DeepSeek V4.1 Flash's
// extension.

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "chat_template.hpp"

namespace gufo::models::hy4_flash {

class Tokenizer {
public:
  /// Loads the byte-level BPE tokenizer from `reader`. When
  /// `canonicalize_specials` is true, any checkpoint suffix on Hy4's suffixed
  /// special tokens (such as AngelSlim's `:6124c78e`) is rewritten to
  /// `:opensource` so the chat template, reasoning delimiters, tool dialect
  /// and vision prompt share one spelling, and every required chat/tool/think
  /// special token is verified to exist.
  [[nodiscard]] static std::unique_ptr<Tokenizer> Load(
      const core::GgufReader& reader, std::string* error,
      bool canonicalize_specials = true);

  /// Plain text: special-token spellings are tokenized as text.
  [[nodiscard]] std::vector<std::int32_t> Encode(std::string_view text) const;
  /// Rendered chat: every control or user-defined token spelling becomes its
  /// id (longest match); the text between is plain. A spelling that overlaps
  /// one of `literal` stays text; its offsets count from `base` (`text`
  /// begins at byte `base` of the rendered prompt) and are in order.
  [[nodiscard]] std::vector<std::int32_t> EncodeWithSpecials(
      std::string_view text, std::span<const ContentSpan> literal = {},
      std::size_t base = 0) const;
  /// How many of EncodeWithSpecials(text, literal, base)'s tokens encode
  /// text[0, cut) when `cut` falls before or after a special token outside
  /// `literal`; `total` is that encoding's size. Encodes the shorter side.
  [[nodiscard]] std::size_t TokensBefore(
      std::string_view text, std::size_t cut, std::size_t total,
      std::span<const ContentSpan> literal = {}, std::size_t base = 0) const;
  /// The bytes of one token: special tokens decode to their spelling.
  [[nodiscard]] std::string Decode(std::int32_t token) const;

  [[nodiscard]] std::int32_t Find(std::string_view spelling) const;
  [[nodiscard]] bool IsSpecial(std::int32_t token) const;
  [[nodiscard]] std::int32_t size() const { return static_cast<std::int32_t>(tokens_.size()); }
  [[nodiscard]] std::int32_t bos() const { return bos_; }
  [[nodiscard]] std::int32_t eos() const { return eos_; }

private:
  Tokenizer() = default;
  void EncodePlain(std::string_view text, std::vector<std::int32_t>* out) const;
  void EmitPiece(std::string_view piece, std::vector<std::int32_t>* out) const;

  std::vector<std::string> tokens_;
  std::vector<bool> special_;
  std::unordered_map<std::string_view, std::int32_t> token_to_id_;
  std::unordered_map<std::string, std::int32_t> merge_rank_;
  // Special spellings by first byte, longest first.
  std::vector<std::vector<std::int32_t>> specials_by_byte_;
  std::int32_t bos_ = -1;
  std::int32_t eos_ = -1;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_TOKENIZER_HPP_
