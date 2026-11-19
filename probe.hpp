#ifndef HY4FLASH_PROBE_HPP_
#define HY4FLASH_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::hy4_flash {

/// An `hy_v4_flash` trunk GGUF. The Engram shard, the vision tower and the
/// DFly2 drafter are artifacts of their trunk, not models of their own, so
/// they do not match.
[[nodiscard]] bool ProbeHy4FlashGguf(const core::GgufReader& reader);
/// `model_root` is the trunk's first GGUF shard.
[[nodiscard]] bool ProbeHy4Flash(std::string_view model_root);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_PROBE_HPP_
