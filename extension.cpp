#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "text_contract.hpp"

// The tool dialect is host-side and registers in every build, before any
// runner is created. HIP builds register the provider from
// gufo_hy4flash_text (provider.cpp); without HIP the probes register with no
// execution factory.
void hy4flash_register(gufo::ext::Registry& registry) {
  gufo::ext::RegisterToolDialect(gufo::models::hy4_flash::OpensourceToolDialect());
#if !defined(ENGINE_ENABLE_HIP)
  namespace hy4 = gufo::models::hy4_flash;
  registry.AddTextProvider({
      .id = "hy4flash",
      .probe = &hy4::ProbeHy4Flash,
      .probe_gguf = &hy4::ProbeHy4FlashGguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
      .sampling_defaults = &hy4::ProbeSamplingDefaults,
      .accepts_unsized_tensors = true,
  });
#else
  (void)registry;
#endif
}
