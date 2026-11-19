#include <gufo/ext/generated/hy4flash_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "runner.hpp"
#include "text_contract.hpp"

void hy4flash_provider_register(gufo::ext::Registry& registry) {
  namespace hy4 = gufo::models::hy4_flash;
  registry.AddProvider(gufo::ext::generated::extension_hy4flash::kProvider_hy4flash);
  registry.AddTextProvider({
      .id = "hy4flash",
      .probe = &hy4::ProbeHy4Flash,
      .probe_gguf = &hy4::ProbeHy4FlashGguf,
      .create = &hy4::CreateHy4TextRunner,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &hy4::ProbeSamplingDefaults,
      // The first shard's reader maps the second, whose FP8 Engram table
      // (type 43) the runtime sizes itself.
      .accepts_unsized_tensors = true,
  });
}
