#ifndef HY4FLASH_TEXT_CONTRACT_HPP_
#define HY4FLASH_TEXT_CONTRACT_HPP_

#include <optional>
#include <string_view>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>

namespace gufo::models::hy4_flash {

/// Identity of the registered Hy4 tool dialect.
inline constexpr std::string_view kToolDialectId = "hy4flash";

/// The native call grammar the dialect constrains generation with.
[[nodiscard]] const sampling::JsonConstraint::ToolSyntax& ToolGrammarSyntax();

/// The dialect the extension registers.
[[nodiscard]] ext::ToolDialect ToolDialect();

/// The GGUF's generation defaults: temperature 0.9, top-p 1.
[[nodiscard]] sampling::TextSamplingDefaults SamplingDefaults();
/// `TextProvider::sampling_defaults`.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_TEXT_CONTRACT_HPP_
