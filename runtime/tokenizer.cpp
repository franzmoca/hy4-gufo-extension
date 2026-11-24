// Byte-level BPE for Hy4 Flash. The pretokenizer split and the BPE merge loop
// are ported from DeepSeek V4.1 Flash's runtime/tokenizer.cpp; Hy4 declares
// the same pretokenizer regexes (llama.cpp groups hy_v4 with deepseek3-llm
// and joyai-llm).

#include "runtime/tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <optional>
#include <unordered_set>
#include <utility>
#include <variant>

namespace gufo::models::hy4_flash {

namespace {

// GGUF token types.
constexpr std::int64_t kTypeControl = 3;
constexpr std::int64_t kTypeUserDefined = 4;

constexpr std::string_view kCanonicalSuffix = ":opensource";

struct SuffixedToken {
  std::string_view prefix;
  std::string_view tail;
};

// All 24 Hy4 special tokens that carry the checkpoint's HYTK suffix. Their
// token IDs are shared across checkpoints; only the suffix varies.
constexpr SuffixedToken kSuffixedTokens[] = {
    {"<｜hy_start", "｜>"},
    {"<｜hy_middle", "｜>"},
    {"<｜hy_pad", "｜>"},
    {"<｜hy_end", "｜>"},
    {"<think", ">"},
    {"</think", ">"},
    {"<tool_call", ">"},
    {"</tool_call", ">"},
    {"<tool_calls", ">"},
    {"</tool_calls", ">"},
    {"<tool_response", ">"},
    {"</tool_response", ">"},
    {"<｜reasoning_mode", "｜>"},
    {"<arg_key", ">"},
    {"</arg_key", ">"},
    {"<arg_value", ">"},
    {"</arg_value", ">"},
    {"<｜hy_image▁start", "｜>"},
    {"<｜hy_image▁end", "｜>"},
    {"<｜hy_image▁pad", "｜>"},
    {"<｜hy_image▁newline", "｜>"},
    {"<｜hy_video▁start", "｜>"},
    {"<｜hy_video▁end", "｜>"},
    {"<｜hy_video▁pad", "｜>"},
};

std::optional<std::string> StartSuffix(std::string_view token) {
  constexpr std::string_view kPrefix = "<｜hy_start";
  constexpr std::string_view kTail = "｜>";
  if (!token.starts_with(kPrefix) || !token.ends_with(kTail) ||
      token.size() < kPrefix.size() + kTail.size()) {
    return std::nullopt;
  }
  return std::string(token.substr(
      kPrefix.size(), token.size() - kPrefix.size() - kTail.size()));
}

bool GptPrintable(std::uint32_t b) { return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255); }

std::uint32_t ByteToCodepoint(std::uint8_t b) {
  if (GptPrintable(b)) return b;
  std::uint32_t n = 0;
  for (std::uint32_t x = 0; x < 256; ++x) {
    if (GptPrintable(x)) continue;
    if (x == b) return 256 + n;
    ++n;
  }
  return b;
}

int CodepointToByte(std::uint32_t cp) {
  if (GptPrintable(cp)) return static_cast<int>(cp);
  std::uint32_t n = 0;
  for (std::uint32_t b = 0; b < 256; ++b) {
    if (GptPrintable(b)) continue;
    if (cp == 256 + n) return static_cast<int>(b);
    ++n;
  }
  return -1;
}

void Utf8Put(std::string* out, std::uint32_t cp) {
  if (cp <= 0x7f) {
    out->push_back(static_cast<char>(cp));
  } else if (cp <= 0x7ff) {
    out->push_back(static_cast<char>(0xc0 | (cp >> 6)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else if (cp <= 0xffff) {
    out->push_back(static_cast<char>(0xe0 | (cp >> 12)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  } else {
    out->push_back(static_cast<char>(0xf0 | (cp >> 18)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
    out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
    out->push_back(static_cast<char>(0x80 | (cp & 0x3f)));
  }
}

int Utf8Len(std::uint8_t c) {
  if (c < 0x80) return 1;
  if ((c & 0xe0) == 0xc0) return 2;
  if ((c & 0xf0) == 0xe0) return 3;
  if ((c & 0xf8) == 0xf0) return 4;
  return 1;
}

std::size_t NextChar(std::string_view s, std::size_t pos) {
  std::size_t n = static_cast<std::size_t>(Utf8Len(static_cast<std::uint8_t>(s[pos])));
  if (pos + n > s.size()) n = 1;
  return pos + n;
}

std::uint32_t Decode1(std::string_view s, std::size_t* pos) {
  const std::uint8_t c0 = static_cast<std::uint8_t>(s[*pos]);
  std::size_t n = static_cast<std::size_t>(Utf8Len(c0));
  if (*pos + n > s.size()) n = 1;
  std::uint32_t cp = c0;
  if (n == 2) cp = c0 & 0x1f;
  if (n == 3) cp = c0 & 0x0f;
  if (n == 4) cp = c0 & 0x07;
  for (std::size_t i = 1; i < n; ++i) cp = (cp << 6) | (static_cast<std::uint8_t>(s[*pos + i]) & 0x3f);
  *pos += n;
  return cp;
}

bool Alpha(std::uint8_t c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
bool Digit(std::uint8_t c) { return c >= '0' && c <= '9'; }
bool Space(std::uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool Newline(std::uint8_t c) { return c == '\n' || c == '\r'; }
bool Punct(std::uint8_t c) {
  return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

// Non-ASCII bytes count as letters, as in DS4.1's port; CJK and kana runs are
// split off before the letter rule.
bool LetterAt(std::string_view s, std::size_t pos) {
  const std::uint8_t c = static_cast<std::uint8_t>(s[pos]);
  return c < 128 ? Alpha(c) : true;
}

std::size_t ConsumeLetters(std::string_view s, std::size_t pos) {
  while (pos < s.size() && LetterAt(s, pos)) pos = NextChar(s, pos);
  return pos;
}

bool CjkAt(std::string_view s, std::size_t pos) {
  if (static_cast<std::uint8_t>(s[pos]) < 128) return false;
  const std::uint32_t cp = Decode1(s, &pos);
  return (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff);
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::Load(const core::GgufReader& reader,
                                           std::string* error,
                                           bool canonicalize_specials) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Tokenizer> {
    if (error != nullptr) {
      *error = std::move(message);
    }
    return nullptr;
  };
  std::unique_ptr<Tokenizer> t(new Tokenizer());
  const auto model = reader.GetMetadataString("tokenizer.ggml.model");
  if (model.value_or("") != "gpt2") {
    return fail("hy4: expected a gpt2 (byte-level BPE) tokenizer");
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (tokens.empty() || merges.empty() || tokens.size() > INT32_MAX) {
    return fail("hy4: GGUF tokenizer tables are missing");
  }
  std::vector<std::int64_t> types;
  if (const auto* v = reader.FindMetadata("tokenizer.ggml.token_type")) {
    if (const auto* s = std::get_if<std::vector<std::int64_t>>(&v->value)) {
      types = *s;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&v->value)) {
      for (const auto x : *u) {
        types.push_back(static_cast<std::int64_t>(x));
      }
    }
  }
  t->tokens_.assign(tokens.begin(), tokens.end());
  t->bos_ = static_cast<std::int32_t>(
      reader.GetMetadataUint32("tokenizer.ggml.bos_token_id").value_or(0));
  t->eos_ = static_cast<std::int32_t>(
      reader.GetMetadataUint32("tokenizer.ggml.eos_token_id").value_or(0));
  const auto is_special_id = [&](std::size_t i) {
    return i < types.size() &&
           (types[i] == kTypeControl || types[i] == kTypeUserDefined) &&
           !t->tokens_[i].empty();
  };
  if (canonicalize_specials) {
    std::optional<std::string> suffix;
    if (t->bos_ >= 0 && static_cast<std::size_t>(t->bos_) < t->tokens_.size() &&
        is_special_id(static_cast<std::size_t>(t->bos_))) {
      suffix = StartSuffix(t->tokens_[static_cast<std::size_t>(t->bos_)]);
    }
    for (std::size_t i = 0; !suffix && i < t->tokens_.size(); ++i) {
      if (is_special_id(i)) {
        suffix = StartSuffix(t->tokens_[i]);
      }
    }
    if (!suffix) {
      return fail("hy4: GGUF vocabulary has no <｜hy_start…｜> special token");
    }
    if (*suffix != kCanonicalSuffix) {
      std::unordered_map<std::string, std::string> rewrites;
      rewrites.reserve(std::size(kSuffixedTokens));
      for (const auto& entry : kSuffixedTokens) {
        std::string from;
        from.reserve(entry.prefix.size() + suffix->size() + entry.tail.size());
        from.append(entry.prefix).append(*suffix).append(entry.tail);
        std::string to;
        to.reserve(entry.prefix.size() + kCanonicalSuffix.size() +
                   entry.tail.size());
        to.append(entry.prefix).append(kCanonicalSuffix).append(entry.tail);
        rewrites.emplace(std::move(from), std::move(to));
      }
      std::vector<std::pair<std::size_t, std::string>> updates;
      std::unordered_set<std::string_view> targets;
      updates.reserve(rewrites.size());
      targets.reserve(rewrites.size());
      for (std::size_t i = 0; i < t->tokens_.size(); ++i) {
        if (!is_special_id(i)) {
          continue;
        }
        if (const auto it = rewrites.find(t->tokens_[i]);
            it != rewrites.end()) {
          updates.emplace_back(i, it->second);
          targets.insert(it->second);
        }
      }
      for (const std::string& token : t->tokens_) {
        if (targets.contains(token)) {
          return fail("hy4: GGUF already contains canonical token " + token);
        }
      }
      for (auto& [index, canonical] : updates) {
        t->tokens_[index] = std::move(canonical);
      }
    }
  }
  t->special_.assign(t->tokens_.size(), false);
  t->token_to_id_.reserve(t->tokens_.size());
  t->specials_by_byte_.resize(256);
  for (std::size_t i = 0; i < t->tokens_.size(); ++i) {
    const auto id = static_cast<std::int32_t>(i);
    t->token_to_id_.emplace(t->tokens_[i], id);
    if (is_special_id(i)) {
      t->special_[i] = true;
      t->specials_by_byte_[static_cast<std::uint8_t>(t->tokens_[i][0])]
          .push_back(id);
    }
  }
  for (auto& list : t->specials_by_byte_) {
    std::stable_sort(list.begin(), list.end(),
                     [&](std::int32_t a, std::int32_t b) {
                       return t->tokens_[a].size() > t->tokens_[b].size();
                     });
  }
  t->merge_rank_.reserve(merges.size());
  std::int32_t rank = 0;
  for (const auto m : merges) {
    t->merge_rank_.emplace(std::string(m), rank++);
  }
  if (canonicalize_specials) {
    for (const std::string_view required : {
             Hy4Tokens::kStart,
             Hy4Tokens::kMiddle,
             Hy4Tokens::kEnd,
             Hy4Tokens::kReasoningMode,
             Hy4Tokens::kThinkOpen,
             Hy4Tokens::kThinkClose,
             Hy4Tokens::kCallsOpen,
             Hy4Tokens::kCallsClose,
             Hy4Tokens::kCallOpen,
             Hy4Tokens::kCallClose,
             Hy4Tokens::kKeyOpen,
             Hy4Tokens::kKeyClose,
             Hy4Tokens::kValueOpen,
             Hy4Tokens::kValueClose,
             Hy4Tokens::kResponseOpen,
             Hy4Tokens::kResponseClose,
         }) {
      if (!t->IsSpecial(t->Find(required))) {
        return fail("hy4: GGUF is missing required special token " +
                    std::string(required));
      }
    }
  }
  return t;
}

std::int32_t Tokenizer::Find(std::string_view spelling) const {
  const auto it = token_to_id_.find(spelling);
  return it == token_to_id_.end() ? -1 : it->second;
}

bool Tokenizer::IsSpecial(std::int32_t token) const {
  return token >= 0 && token < size() && special_[static_cast<std::size_t>(token)];
}

std::string Tokenizer::Decode(std::int32_t token) const {
  if (token < 0 || token >= size()) return {};
  const std::string& s = tokens_[static_cast<std::size_t>(token)];
  if (special_[static_cast<std::size_t>(token)]) return s;
  std::string out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const int b = CodepointToByte(Decode1(s, &pos));
    if (b >= 0) out.push_back(static_cast<char>(b));
  }
  return out;
}

void Tokenizer::EmitPiece(std::string_view piece, std::vector<std::int32_t>* out) const {
  std::string encoded;
  for (const char c : piece) Utf8Put(&encoded, ByteToCodepoint(static_cast<std::uint8_t>(c)));
  std::vector<std::string> sym;
  for (std::size_t off = 0; off < encoded.size();) {
    const std::size_t next = NextChar(encoded, off);
    sym.emplace_back(encoded.substr(off, next - off));
    off = next;
  }
  std::string key;
  for (;;) {
    int best = -1;
    std::int32_t best_rank = INT32_MAX;
    for (std::size_t i = 0; i + 1 < sym.size(); ++i) {
      key.assign(sym[i]).append(" ").append(sym[i + 1]);
      const auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) break;
    sym[static_cast<std::size_t>(best)] += sym[static_cast<std::size_t>(best) + 1];
    sym.erase(sym.begin() + best + 1);
  }
  for (const auto& s : sym) {
    const auto it = token_to_id_.find(s);
    if (it != token_to_id_.end()) {
      out->push_back(it->second);
      continue;
    }
    for (std::size_t j = 0; j < s.size(); ++j) {
      const auto byte = token_to_id_.find(std::string_view(s).substr(j, 1));
      if (byte != token_to_id_.end()) out->push_back(byte->second);
    }
  }
}

void Tokenizer::EncodePlain(std::string_view text, std::vector<std::int32_t>* out) const {
  const std::size_t len = text.size();
  std::size_t pos = 0;
  std::unordered_map<std::string_view, std::vector<std::int32_t>> memo;
  while (pos < len) {
    const std::size_t start = pos;
    const std::uint8_t c = static_cast<std::uint8_t>(text[pos]);
    if (Digit(c)) {
      int digits = 0;
      while (pos < len && Digit(static_cast<std::uint8_t>(text[pos])) && digits < 3) {
        ++pos;
        ++digits;
      }
    } else if (CjkAt(text, pos)) {
      do {
        pos = NextChar(text, pos);
      } while (pos < len && CjkAt(text, pos));
    } else if (Punct(c) && pos + 1 < len && Alpha(static_cast<std::uint8_t>(text[pos + 1]))) {
      ++pos;
      while (pos < len && Alpha(static_cast<std::uint8_t>(text[pos]))) ++pos;
    } else if (LetterAt(text, pos)) {
      pos = ConsumeLetters(text, pos);
    } else if (!Newline(c) && !Punct(c) && pos + 1 < len && LetterAt(text, pos + 1)) {
      pos = ConsumeLetters(text, pos + 1);
    } else if (c == ' ' && pos + 1 < len && Punct(static_cast<std::uint8_t>(text[pos + 1]))) {
      ++pos;
      while (pos < len && Punct(static_cast<std::uint8_t>(text[pos]))) ++pos;
      while (pos < len && Newline(static_cast<std::uint8_t>(text[pos]))) ++pos;
    } else if (Punct(c)) {
      while (pos < len && Punct(static_cast<std::uint8_t>(text[pos]))) ++pos;
      while (pos < len && Newline(static_cast<std::uint8_t>(text[pos]))) ++pos;
    } else if (Space(c)) {
      std::size_t p = pos;
      std::size_t last_newline_end = 0;
      while (p < len && Space(static_cast<std::uint8_t>(text[p]))) {
        if (Newline(static_cast<std::uint8_t>(text[p++]))) last_newline_end = p;
      }
      if (last_newline_end) {
        pos = last_newline_end;
      } else if (p < len && p > pos + 1 && (LetterAt(text, p) || Punct(static_cast<std::uint8_t>(text[p])))) {
        pos = p - 1;  // one space joins the following word or punctuation
      } else {
        pos = p;
      }
    } else {
      pos = NextChar(text, pos);
    }
    if (pos == start) pos = NextChar(text, pos);
    const std::string_view piece = text.substr(start, pos - start);
    const bool memoize = piece.size() > 1 && piece.size() <= 64;
    if (memoize) {
      const auto found = memo.find(piece);
      if (found != memo.end()) {
        out->insert(out->end(), found->second.begin(), found->second.end());
        continue;
      }
    }
    const std::size_t first = out->size();
    EmitPiece(piece, out);
    if (memoize && memo.size() < 4096) memo.emplace(piece, std::vector<std::int32_t>(out->begin() + first, out->end()));
  }
}

std::vector<std::int32_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<std::int32_t> out;
  EncodePlain(text, &out);
  return out;
}

std::vector<std::int32_t> Tokenizer::EncodeWithSpecials(
    std::string_view text, std::span<const ContentSpan> literal,
    std::size_t base) const {
  std::vector<std::int32_t> out;
  std::size_t span = 0;
  std::size_t p = 0;
  std::size_t next = 0;  // first literal span that may reach past `p`
  while (p < text.size()) {
    std::int32_t hit = -1;
    for (const std::int32_t id : specials_by_byte_[static_cast<std::uint8_t>(text[p])]) {
      const std::string& s = tokens_[static_cast<std::size_t>(id)];
      if (text.compare(p, s.size(), s) == 0) {
        hit = id;
        break;
      }
    }
    if (hit >= 0) {
      // Request text spells control tokens as characters.
      const std::size_t at = base + p;
      while (next < literal.size() &&
             literal[next].offset + literal[next].size <= at)
        ++next;
      if (next < literal.size() &&
          literal[next].offset <
              at + tokens_[static_cast<std::size_t>(hit)].size())
        hit = -1;
    }
    if (hit < 0) {
      ++p;
      continue;
    }
    EncodePlain(text.substr(span, p - span), &out);
    out.push_back(hit);
    p += tokens_[static_cast<std::size_t>(hit)].size();
    span = p;
  }
  EncodePlain(text.substr(span), &out);
  return out;
}

std::size_t Tokenizer::TokensBefore(std::string_view text, std::size_t cut,
                                    std::size_t total,
                                    std::span<const ContentSpan> literal,
                                    std::size_t base) const {
  if (cut < text.size() - cut)
    return EncodeWithSpecials(text.substr(0, cut), literal, base).size();
  return total -
         std::min(
             total,
             EncodeWithSpecials(text.substr(cut), literal, base + cut).size());
}

}  // namespace gufo::models::hy4_flash
