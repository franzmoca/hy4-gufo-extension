#include "tool_calls.hpp"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <optional>
#include <string>

#include <gufo/ext/json.hpp>

#include "chat_template.hpp"

namespace gufo::models::hy4_flash {
namespace {

bool Space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

std::string_view Trim(std::string_view v) {
  while (!v.empty() && Space(v.front())) v.remove_prefix(1);
  while (!v.empty() && Space(v.back())) v.remove_suffix(1);
  return v;
}

std::size_t SkipSpace(std::string_view text, std::size_t pos) {
  while (pos < text.size() && Space(text[pos])) ++pos;
  return pos;
}

bool At(std::string_view text, std::size_t pos, std::string_view tag) { return text.substr(pos).starts_with(tag); }

// The text from `pos` could still grow into one of the tags.
bool PrefixOfAny(std::string_view rest, std::initializer_list<std::string_view> tags) {
  return std::ranges::any_of(tags, [&](std::string_view tag) { return tag.starts_with(rest); });
}

// Hold back a possible start of `delimiter` at the end of `value`.
std::string_view BeforePartialDelimiter(std::string_view value, std::string_view delimiter) {
  for (std::size_t size = std::min(value.size(), delimiter.size()); size > 0; --size) {
    if (delimiter.starts_with(value.substr(value.size() - size))) return value.substr(0, value.size() - size);
  }
  return value;
}

ext::ScannedToolArgument Argument(std::string name, std::string_view value) {
  const std::string_view trimmed = Trim(value);
  try {
    const json::Value parsed = json::parse(trimmed);
    if (!parsed.is_string()) return {std::move(name), std::string(trimmed), false};
  } catch (...) {
  }
  return {std::move(name), std::string(value), true};
}

// Whether an unfinished value is (so far) text rather than a JSON value.
bool StringSoFar(std::string_view value) {
  const std::string_view v = Trim(value);
  if (v.empty()) return false;
  if (std::string_view("{[-0123456789\"").find(v.front()) != std::string_view::npos) return false;
  return !PrefixOfAny(v, {"true", "false", "null"});
}

// The spellings of the dialect that opens a block with one of `markers`: the
// scanner is handed the markers and nothing else, and the checkpoint's suffix
// rides inside them, as the template builds them all from one suffix.
std::optional<Hy4Tokens> TokensFor(std::span<const std::string_view> markers) {
  constexpr std::string_view open = "<tool_calls";
  for (const std::string_view marker : markers) {
    if (!marker.starts_with(open) || !marker.ends_with(">")) continue;
    return MakeHy4Tokens(
        marker.substr(open.size(), marker.size() - open.size() - 1));
  }
  return std::nullopt;
}

}  // namespace

void ScanHy4ToolCalls(std::string_view text, std::span<const std::string_view> markers,
                      std::vector<ext::ScannedToolCall>* calls,
                      std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial) {
  const std::optional<Hy4Tokens> tokens = TokensFor(markers);
  if (!tokens) return;  // no Hy4 block opener among the markers
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t begin = std::string_view::npos;
    std::size_t marker_size = 0;
    for (const std::string_view marker : markers) {
      const std::size_t at = text.find(marker, cursor);
      if (at < begin) {
        begin = at;
        marker_size = marker.size();
      }
    }
    if (begin == std::string_view::npos) return;
    std::size_t pos = begin + marker_size;
    bool any_complete = false;
    while (true) {
      pos = SkipSpace(text, pos);
      if (At(text, pos, tokens->calls_close)) {
        pos += tokens->calls_close.size();
        if (spans) spans->emplace_back(begin, pos);
        break;
      }
      if (pos >= text.size() || !At(text, pos, tokens->call_open)) {
        const std::string_view rest = text.substr(pos);
        if (rest.empty() || PrefixOfAny(rest, {tokens->call_open, tokens->calls_close})) {
          // The block ends with the text: a final parse closes it there.
          if (!partial && any_complete && spans) spans->emplace_back(begin, text.size());
        }
        return;
      }
      pos += tokens->call_open.size();
      // The name ends at the first tag.
      const std::size_t name_end = text.find('<', pos);
      if (name_end == std::string_view::npos) return;  // the name may still grow
      ext::ScannedToolCall call;
      call.name = std::string(Trim(text.substr(pos, name_end - pos)));
      pos = name_end;
      while (true) {
        pos = SkipSpace(text, pos);
        if (At(text, pos, tokens->call_close)) {
          pos += tokens->call_close.size();
          call.valid = !call.name.empty();
          call.complete = true;
          break;
        }
        if (At(text, pos, tokens->key_open)) {
          const std::size_t key_begin = pos + tokens->key_open.size();
          const std::size_t key_end = text.find(tokens->key_close, key_begin);
          if (key_end == std::string_view::npos) {
            if (partial) calls->push_back(std::move(call));
            return;
          }
          std::string key(Trim(text.substr(key_begin, key_end - key_begin)));
          pos = SkipSpace(text, key_end + tokens->key_close.size());
          if (!At(text, pos, tokens->value_open)) {
            if (partial && PrefixOfAny(text.substr(pos), {tokens->value_open})) {
              calls->push_back(std::move(call));
              return;
            }
            call.complete = true;  // malformed
            calls->push_back(std::move(call));
            return;
          }
          const std::size_t value_begin = pos + tokens->value_open.size();
          const std::size_t value_end = text.find(tokens->value_close, value_begin);
          if (value_end == std::string_view::npos) {
            if (partial) {
              const std::string_view sofar =
                  BeforePartialDelimiter(text.substr(value_begin), tokens->value_close);
              if (StringSoFar(sofar)) call.unfinished = ext::ScannedToolArgument{key, std::string(sofar), true};
              calls->push_back(std::move(call));
            }
            return;
          }
          call.arguments.push_back(Argument(std::move(key), text.substr(value_begin, value_end - value_begin)));
          pos = value_end + tokens->value_close.size();
          continue;
        }
        if (partial && PrefixOfAny(text.substr(pos), {tokens->call_close, tokens->key_open})) {
          calls->push_back(std::move(call));
          return;
        }
        call.complete = true;  // malformed: valid stays false
        calls->push_back(std::move(call));
        return;
      }
      any_complete = any_complete || call.valid;
      calls->push_back(std::move(call));
    }
    cursor = pos;
  }
}

}  // namespace gufo::models::hy4_flash
