#include <gufo/ext/generated/hy4flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"

void hy4flash_register(gufo::ext::Registry&) {}

// Metadata and probe only: this revision has no execution factory yet.
void hy4flash_provider_register(gufo::ext::Registry& registry) {
  namespace hy4 = gufo::models::hy4_flash;
  registry.AddProvider(
      gufo::ext::generated::extension_hy4flash::kProvider_hy4flash);
  registry.AddTextProvider({
      .id = "hy4flash",
      .probe = &hy4::ProbeHy4Flash,
      .probe_gguf = &hy4::ProbeHy4FlashGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
  });
}
