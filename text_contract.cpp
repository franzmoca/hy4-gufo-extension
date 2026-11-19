#include "text_contract.hpp"

#include <string>

#include "chat_template.hpp"
#include "tool_calls.hpp"

namespace gufo::models::hy4_flash {

const sampling::JsonConstraint::ToolSyntax& ToolGrammarSyntax() {
  using T = Hy4Tokens;
  // The tags carry no value type: raw and typed values share one opener, so
  // a parameter admitting strings is raw text.
  static const sampling::JsonConstraint::ToolSyntax syntax{
      .open = std::string(T::kCallsOpen),
      .name_prefix = std::string(T::kCallOpen),
      .name_suffix = "",
      .close = std::string(T::kCallClose),
      .block_close = std::string(T::kCallsClose),
      .name_delimiters = "<",
      .values = {},
      .tagged =
          sampling::JsonConstraint::TaggedArguments{
              .parameter_open = std::string(T::kKeyOpen),
              .raw_value_open = std::string(T::kKeyClose) + std::string(T::kValueOpen),
              .typed_value_open = std::string(T::kKeyClose) + std::string(T::kValueOpen),
              .value_close = std::string(T::kValueClose),
              .parameter_separator = "",
          },
      .reasoning_header = true,
  };
  return syntax;
}

ext::ToolDialect ToolDialect() {
  static const std::string native_open = std::string(Hy4Tokens::kCallsOpen) + std::string(Hy4Tokens::kCallOpen);
  return {
      .markers = {Hy4Tokens::kCallsOpen},
      .reasoning_markers = {native_open},
      .closers = {},
      .scan_tool_calls = &ScanHy4ToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = ToolGrammarSyntax(),
  };
}

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
