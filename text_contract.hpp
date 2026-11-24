#ifndef HY4FLASH_TEXT_CONTRACT_HPP_
#define HY4FLASH_TEXT_CONTRACT_HPP_

#include <optional>
#include <string>
#include <string_view>

#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/text.hpp>

#include "chat_template.hpp"

namespace gufo::models::hy4_flash {

/// Identity of the tool dialect for the `:opensource` spellings, the one the
/// extension registers with itself.
inline constexpr std::string_view kToolDialectId = "hy4flash";

/// The dialect identity of one checkpoint's spellings: `kToolDialectId` for
/// `:opensource`, that and the suffix for every other checkpoint.
[[nodiscard]] std::string ToolDialectId(const Hy4Tokens& tokens);

/// The native call grammar `tokens` spells.
[[nodiscard]] sampling::JsonConstraint::ToolSyntax ToolGrammarSyntax(const Hy4Tokens& tokens);

/// The dialect for one checkpoint's spellings. Its markers view into `tokens`,
/// which therefore has to outlive the registration: use
/// `EnsureToolDialectRegistered` rather than registering one built here.
[[nodiscard]] ext::ToolDialect ToolDialect(const Hy4Tokens& tokens);

/// The dialect of `tokens`: the registered one when a dialect already holds
/// that identity, registered now otherwise. A `:opensource` model needs none
/// of this, since extension registration registers that dialect; any other
/// checkpoint carries its suffix inside its own GGUF, which no extension sees
/// while it registers, so its dialect arrives with the model. Returns null
/// with `error` set when the spellings cannot be registered.
[[nodiscard]] const ext::ToolDialect* EnsureToolDialectRegistered(const Hy4Tokens& tokens,
                                                                 std::string* error);

/// The dialect the extension registers, over the `:opensource` spellings.
[[nodiscard]] ext::ToolDialect OpensourceToolDialect();

/// The GGUF's generation defaults: temperature 0.9, top-p 1.
[[nodiscard]] sampling::TextSamplingDefaults SamplingDefaults();
/// `TextProvider::sampling_defaults`.
[[nodiscard]] std::optional<sampling::TextSamplingDefaults> ProbeSamplingDefaults(const core::GgufReader& reader);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_TEXT_CONTRACT_HPP_
