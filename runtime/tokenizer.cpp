// Byte-level BPE for Hy4 Flash. The pretokenizer split and the BPE merge loop
// are ported from DeepSeek V4.1 Flash's runtime/tokenizer.cpp; Hy4 declares
// the same pretokenizer regexes (llama.cpp groups hy_v4 with deepseek3-llm
// and joyai-llm).

#include "runtime/tokenizer.hpp"

#include <algorithm>
#include <climits>
#include <cstring>
#include <variant>

namespace gufo::models::hy4_flash {

namespace {

// GGUF token types.
constexpr std::int64_t kTypeControl = 3;
constexpr std::int64_t kTypeUserDefined = 4;

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

std::unique_ptr<Tokenizer> Tokenizer::Load(const core::GgufReader& reader, std::string* error) {
  std::unique_ptr<Tokenizer> t(new Tokenizer());
  const auto model = reader.GetMetadataString("tokenizer.ggml.model");
  if (model.value_or("") != "gpt2") {
    *error = "hy4: expected a gpt2 (byte-level BPE) tokenizer";
    return nullptr;
  }
  const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto merges = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (tokens.empty() || merges.empty() || tokens.size() > INT32_MAX) {
    *error = "hy4: GGUF tokenizer tables are missing";
    return nullptr;
  }
  std::vector<std::int64_t> types;
  if (const auto* v = reader.FindMetadata("tokenizer.ggml.token_type")) {
    if (const auto* s = std::get_if<std::vector<std::int64_t>>(&v->value)) types = *s;
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&v->value)) {
      for (const auto x : *u) types.push_back(static_cast<std::int64_t>(x));
    }
  }
  t->tokens_.assign(tokens.begin(), tokens.end());
  t->special_.assign(t->tokens_.size(), false);
  t->token_to_id_.reserve(t->tokens_.size());
  t->specials_by_byte_.resize(256);
  for (std::size_t i = 0; i < t->tokens_.size(); ++i) {
    const auto id = static_cast<std::int32_t>(i);
    t->token_to_id_.emplace(t->tokens_[i], id);
    const bool special = i < types.size() && (types[i] == kTypeControl || types[i] == kTypeUserDefined);
    if (special && !t->tokens_[i].empty()) {
      t->special_[i] = true;
      t->specials_by_byte_[static_cast<std::uint8_t>(t->tokens_[i][0])].push_back(id);
    }
  }
  for (auto& list : t->specials_by_byte_) {
    std::stable_sort(list.begin(), list.end(),
                     [&](std::int32_t a, std::int32_t b) { return t->tokens_[a].size() > t->tokens_[b].size(); });
  }
  t->merge_rank_.reserve(merges.size());
  std::int32_t rank = 0;
  for (const auto m : merges) t->merge_rank_.emplace(std::string(m), rank++);
  t->bos_ = static_cast<std::int32_t>(reader.GetMetadataUint32("tokenizer.ggml.bos_token_id").value_or(0));
  t->eos_ = static_cast<std::int32_t>(reader.GetMetadataUint32("tokenizer.ggml.eos_token_id").value_or(0));
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

std::vector<std::int32_t> Tokenizer::EncodeWithSpecials(std::string_view text) const {
  std::vector<std::int32_t> out;
  std::size_t span = 0;
  std::size_t p = 0;
  while (p < text.size()) {
    std::int32_t hit = -1;
    for (const std::int32_t id : specials_by_byte_[static_cast<std::uint8_t>(text[p])]) {
      const std::string& s = tokens_[static_cast<std::size_t>(id)];
      if (text.compare(p, s.size(), s) == 0) {
        hit = id;
        break;
      }
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

}  // namespace gufo::models::hy4_flash
