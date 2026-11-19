#ifndef HY4FLASH_TOOL_CALLS_HPP_
#define HY4FLASH_TOOL_CALLS_HPP_

#include <cstddef>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <gufo/ext/text.hpp>

namespace gufo::models::hy4_flash {

/// Scans Hy4 call blocks:
///   <tool_calls>(<tool_call>NAME(<arg_key>K</arg_key><arg_value>V</arg_value>)*</tool_call>)+</tool_calls>
/// (each tag with the `:opensource` suffix). The tags carry no value type, and
/// the scanner has no schema: a value that parses as JSON other than a string
/// (number, boolean, null, object, array) is typed, any other value a string.
/// `spans` receives each block, closed by its end tag or, in a final parse,
/// by the end of the text after a complete call.
void ScanHy4ToolCalls(std::string_view text, std::span<const std::string_view> markers,
                      std::vector<ext::ScannedToolCall>* calls,
                      std::vector<std::pair<std::size_t, std::size_t>>* spans, bool partial);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_TOOL_CALLS_HPP_
