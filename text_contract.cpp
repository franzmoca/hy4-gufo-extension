#include "text_contract.hpp"

#include <cctype>
#include <deque>
#include <mutex>
#include <string>
#include <utility>

#include "chat_template.hpp"
#include "tool_calls.hpp"

namespace gufo::models::hy4_flash {
namespace {

// A registered dialect holds views into the spellings it was built from, and
// the registry owns the dialect for the process, so the spellings live here as
// long: a deque never moves the elements it already holds.
std::deque<Hy4Tokens>& DialectSpellings() {
  static std::deque<Hy4Tokens> spellings;
  return spellings;
}

// The registry has no lock of its own: `RegisterToolDialect` says registration
// is not thread-safe with lookups. One model load at a time is on us.
std::mutex& DialectMutex() {
  static std::mutex mutex;
  return mutex;
}

}  // namespace

std::string ToolDialectId(const Hy4Tokens& tokens) {
  if (tokens.suffix == OpensourceHy4Tokens().suffix) return std::string(kToolDialectId);
  std::string id(kToolDialectId);
  id += '-';
  for (const char c : tokens.suffix) {
    if (c == ':') continue;  // the suffix's own colon, not a separator
    id += std::isalnum(static_cast<unsigned char>(c)) != 0 ? c : '-';
  }
  return id;
}

sampling::JsonConstraint::ToolSyntax ToolGrammarSyntax(const Hy4Tokens& tokens) {
  // The tags carry no value type: raw and typed values share one opener, so
  // a parameter admitting strings is raw text.
  sampling::JsonConstraint::ToolSyntax syntax{
      .open = tokens.calls_open,
      .name_prefix = tokens.call_open,
      .name_suffix = "",
      .close = tokens.call_close,
      .block_close = tokens.calls_close,
      .name_delimiters = "<",
      .values = {},
      .tagged =
          sampling::JsonConstraint::TaggedArguments{
              .parameter_open = tokens.key_open,
              .raw_value_open = tokens.key_close + tokens.value_open,
              .typed_value_open = tokens.key_close + tokens.value_open,
              .value_close = tokens.value_close,
              .parameter_separator = "",
          },
      .reasoning_header = true,
  };
  return syntax;
}

ext::ToolDialect ToolDialect(const Hy4Tokens& tokens) {
  return {
      .markers = {std::string_view(tokens.calls_open)},
      .reasoning_markers = {std::string_view(tokens.call_header)},
      .closers = {},
      .scan_tool_calls = &ScanHy4ToolCalls,
      .id = ToolDialectId(tokens),
      .syntax = ToolGrammarSyntax(tokens),
  };
}

const ext::ToolDialect* EnsureToolDialectRegistered(const Hy4Tokens& tokens,
                                                   std::string* error) {
  const std::string id = ToolDialectId(tokens);
  const std::lock_guard lock(DialectMutex());
  if (const ext::ToolDialect* found = ext::FindToolDialect(id)) {
    if (found->markers.size() == 1 && found->markers[0] == tokens.calls_open) return found;
    if (error) *error = "tool dialect " + id + " already describes another spelling";
    return nullptr;
  }
  DialectSpellings().push_back(tokens);
  try {
    // The registry keeps views into the spellings, which is why they go in
    // before the dialect and stay whatever happens.
    ext::RegisterToolDialect(ToolDialect(DialectSpellings().back()));
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
  return ext::FindToolDialect(id);
}

ext::ToolDialect OpensourceToolDialect() { return ToolDialect(OpensourceHy4Tokens()); }

sampling::TextSamplingDefaults SamplingDefaults() {
  sampling::TextSamplingValues values;
  values.temperature = 0.9F;
  values.top_p = 1.0F;
  return {values, values, values};
}

std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(const core::GgufReader&) {
  return SamplingDefaults();
}

}  // namespace gufo::models::hy4_flash
