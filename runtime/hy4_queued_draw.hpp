// The token a sampler returns before any draw from logits, in the order
// SamplerState::Sample takes it, for the device sampling path.
#pragma once

#include <cstdint>
#include <gufo/ext/sampling.hpp>
#include <optional>

namespace gufo::models::hy4_flash {

enum class QueuedDraw : std::uint8_t {
  kNone,   // nothing queued: draw from the logits
  kToken,  // *token is the next token; the sampler has consumed it
  kHost,   // only the host sampler may decide (it reports the violation)
};

// A forced token (an exhausted reasoning budget) wins and drops any deferred
// draw; a deferred draw comes next unless the grammar rejects it.
inline QueuedDraw TakeQueuedDraw(sampling::SamplerState& sampler,
                                 std::int32_t* token) {
  const auto saved = sampler.SaveDrawState();
  if (const auto forced = sampler.ForcedToken()) {
    if (saved.pending)
      sampler.RestoreDrawState({saved.rng, std::nullopt});
    *token = static_cast<std::int32_t>(*forced);
    return QueuedDraw::kToken;
  }
  if (!saved.pending)
    return QueuedDraw::kNone;
  if (sampler.HasConstraint() && !sampler.AllowsToken(*saved.pending))
    return QueuedDraw::kHost;
  sampler.RestoreDrawState({saved.rng, std::nullopt});
  *token = static_cast<std::int32_t>(*saved.pending);
  return QueuedDraw::kToken;
}

}  // namespace gufo::models::hy4_flash
