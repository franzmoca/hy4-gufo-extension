// TakeQueuedDraw against Gufo's host sampler: the device sampling path must
// return the token SamplerState::Sample would, and leave the same state, when
// a reasoning budget forces its end or a draw was deferred. No model, no GPU.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/hy4_queued_draw.hpp"

namespace {

namespace hy4 = gufo::models::hy4_flash;
using gufo::sampling::ConstraintVocabulary;
using gufo::sampling::JsonConstraint;
using gufo::sampling::ReasoningBudget;
using gufo::sampling::SamplerState;
using gufo::sampling::SamplingConfig;
using gufo::sampling::TokenConstraint;
using gufo::sampling::TokenId;

int g_failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

std::vector<float> Logits(TokenId winner, std::size_t vocab = 8) {
  std::vector<float> logits(vocab, 0.0F);
  logits[winner] = 5.0F;
  return logits;
}

// A spent budget forces its end even over a deferred draw, as the host does.
void TestForcedBeatsPending() {
  SamplingConfig config;
  auto budget = std::make_shared<ReasoningBudget>();
  budget->tokens = 1;
  budget->forced = {3};
  budget->ends = {3};
  config.reasoning_budget = budget;
  SamplerState sampler(config);
  sampler.Accept(sampler.Sample(Logits(1)));
  Expect(sampler.ForcedToken() == TokenId{3}, "the budget is spent");
  sampler.DeferSample(5);

  SamplerState host = sampler;
  const TokenId want = host.Sample(Logits(5));
  std::int32_t token = -1;
  Expect(hy4::TakeQueuedDraw(sampler, &token) == hy4::QueuedDraw::kToken,
         "forced: a queued token");
  Expect(token == static_cast<std::int32_t>(want) && want == 3,
         "forced: the host's token, the forced end (got " +
             std::to_string(token) + ", host " + std::to_string(want) + ")");
  Expect(!sampler.SaveDrawState().pending && !host.SaveDrawState().pending,
         "forced: the deferred draw is dropped, as on the host");
}

void TestPending() {
  SamplerState sampler(SamplingConfig{});
  sampler.DeferSample(2);
  std::int32_t token = -1;
  Expect(hy4::TakeQueuedDraw(sampler, &token) == hy4::QueuedDraw::kToken &&
             token == 2,
         "pending: the deferred draw");
  Expect(!sampler.SaveDrawState().pending, "pending: consumed");
  Expect(hy4::TakeQueuedDraw(sampler, &token) == hy4::QueuedDraw::kNone,
         "nothing queued after it");
}

// With a grammar, a deferred draw it rejects goes to the host sampler, which
// throws; one it admits is returned.
void TestPendingUnderGrammar() {
  const std::vector<std::string> pieces{"{\"x\":", "true}", "prose", ""};
  auto binding = std::make_shared<TokenConstraint>();
  binding->grammar = JsonConstraint::Object();
  binding->vocabulary = std::make_shared<ConstraintVocabulary>(
      pieces.size(), [&](std::uint32_t i) {
        return ConstraintVocabulary::Piece{pieces[i], i == 3};
      });
  SamplingConfig config;
  config.constraint = binding;

  SamplerState rejected(config);
  rejected.DeferSample(2);
  std::int32_t token = -1;
  Expect(hy4::TakeQueuedDraw(rejected, &token) == hy4::QueuedDraw::kHost,
         "grammar: a rejected deferred draw goes to the host");
  Expect(rejected.SaveDrawState().pending == TokenId{2},
         "grammar: the host still sees the deferred draw");
  bool threw = false;
  try {
    (void)rejected.Sample(Logits(2, pieces.size()));
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Expect(threw, "grammar: the host reports the violation");

  SamplerState admitted(config);
  admitted.DeferSample(0);
  Expect(hy4::TakeQueuedDraw(admitted, &token) == hy4::QueuedDraw::kToken &&
             token == 0,
         "grammar: an admitted deferred draw");
}

}  // namespace

int main() {
  TestForcedBeatsPending();
  TestPending();
  TestPendingUnderGrammar();
  if (g_failures != 0)
    std::fprintf(stderr, "%d failures\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
