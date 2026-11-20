#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/options.hpp>
#include <gufo/ext/text.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "continuation_adapter.hpp"
#include "prompt_lookup.hpp"
#include "runtime/hy4_dfly.hpp"
#include "runtime/hy4_runtime.hpp"
#include "runtime/hy4_vision.hpp"
#include "runtime/tokenizer.hpp"
#include "text_contract.hpp"
#include "vision_prompt.hpp"

namespace gufo::models::hy4_flash {
namespace {

using namespace gufo::ext;

constexpr std::string_view kStateAbi = "hy4-flash-gfx1151-state-v3";
// Without --context: the native context needs 7 GiB of latent rows per
// session at 256K, so serving defaults to this.
constexpr std::uint32_t kDefaultContext = 32768;
// Prompt-cache transfer streams besides each session's own.
constexpr std::size_t kBatchTransferStreams = 16;
// DFly2 drafts per step without --dfly-draft-tokens.
constexpr std::int64_t kDefaultDraftTokens = 7;
// See Hy4TextRunner::min_accept_.
constexpr float kDefaultMinAccept = 0.45f;

std::size_t CacheCeilingBytes() { return std::max(HostSnapshotCeilingBytes(), HostSnapshotBudgetBytes()); }

bool IsSha256Hex(std::string_view value) noexcept {
  return value.size() == 64 && std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

std::vector<std::uint8_t> CompatibilityIdentity(std::string_view artifact_fingerprint, std::uint32_t max_context) {
  if (!IsSha256Hex(artifact_fingerprint)) throw std::invalid_argument("Hy4 disk cache requires an artifact fingerprint");
  std::ostringstream identity;
  identity << "schema=gufo-text-continuation-v2\n"
           << "model_kind=hy4flash\n"
           << "artifact_id=" << ext::kGgufIdentityScheme << ':' << artifact_fingerprint << '\n'
           << "tokenizer=hy-v4-byte-bpe-v1\n"
           << "chat_template=hy4-flash-opensource-v1\n"
           << "state_abi=" << kStateAbi << '\n'
           << "payload_layout=hy4-rocm-v1-f32-latent\n"
           << "numerics=hy4-v1\n"
           << "context_tokens=" << max_context << '\n'
           << "position_policy=absolute-v1\n"
           << "prefill_policy=exact\n";
  const std::string canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

// Without file fingerprints the loaded model is the identity; it is valid only
// inside this process, so it can never match a durable cache.
std::vector<std::uint8_t> ProcessIdentity(const void* model, std::uint32_t max_context) {
  std::ostringstream identity;
  identity << "schema=gufo-process-text-continuation-v1\n"
           << "model_instance=" << model << '\n'
           << "context_tokens=" << max_context << '\n';
  const auto canonical = identity.str();
  return {canonical.begin(), canonical.end()};
}

// A request's images and their encodings, which a session reads as it
// evaluates their rows.
class ImageSource final : public ImageInputs {
public:
  ImageSource(std::shared_ptr<const ImagePrompt> prompt, VisionEncoder& encoder)
      : prompt_(std::move(prompt)),
        encoder_(&encoder),
        rows_(prompt_->images.size()) {
    for (const auto& image : prompt_->images)
      spans_.push_back(image.span);
  }
  [[nodiscard]] std::span<const ImageSpan> Spans() const override {
    return spans_;
  }
  [[nodiscard]] const float* Embeddings(std::size_t index,
                                        std::string* error) override {
    auto& rows = rows_.at(index);
    if (!rows) {
      const auto& image = prompt_->images[index];
      rows = encoder_->Encode(image.pixels.pixels.data(), image.pixels.width,
                              image.pixels.height, error);
      if (!rows)
        return nullptr;
      if (rows->rows() != image.span.Rows()) {
        rows.reset();
        if (error)
          *error = "hy4 vision: encoder rows differ from the prompt's";
        return nullptr;
      }
    }
    return rows->data();
  }
  [[nodiscard]] const ImagePrompt* prompt() const { return prompt_.get(); }

private:
  std::shared_ptr<const ImagePrompt> prompt_;
  VisionEncoder* encoder_;
  std::vector<ImageSpan> spans_;
  std::vector<std::shared_ptr<const DeviceRows>> rows_;
};

// An image's rows are one prefill span: the cache never keeps a frontier
// inside one, so a restore continues from text.
class ImageContext final : public TextPromptContext {
public:
  std::shared_ptr<const ImagePrompt> prompt;
  [[nodiscard]] std::size_t PrefillStop(std::size_t position,
                                        bool up) const override {
    for (const auto& image : prompt->images) {
      if (position > image.span.first && position < image.span.End())
        return up ? image.span.End() : image.span.first;
    }
    return position;
  }
};

class Hy4TextRunnerState final : public TextRunnerState {
public:
  Hy4TextRunnerState(Session& session, cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
    lookup_.Clear();
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }
  [[nodiscard]] const std::shared_ptr<ImageSource>& images() const {
    return images_;
  }
  /// Prompt lookup over the session's committed tokens.
  [[nodiscard]] lookup::PromptLookup<std::int32_t>& lookup() { return lookup_; }
  void set_images(std::shared_ptr<ImageSource> images) {
    images_ = std::move(images);
    session_->SetImages(images_);
  }

private:
  std::shared_ptr<ImageSource> images_;
  lookup::PromptLookup<std::int32_t> lookup_;
  Session* session_{};
  cache::Adapter* adapter_{};
  cache::Slot* slot_{};
  std::size_t position_{0};
};

Hy4TextRunnerState& RequireState(TextRunnerState& state) {
  auto* hy4 = dynamic_cast<Hy4TextRunnerState*>(&state);
  if (hy4 == nullptr) throw std::logic_error("text runner state is not Hy4");
  return *hy4;
}
const Hy4TextRunnerState& RequireState(const TextRunnerState& state) {
  return RequireState(const_cast<TextRunnerState&>(state));
}

std::vector<TextRunnerToken> RunnerTokens(std::span<const std::int32_t> tokens) {
  std::vector<TextRunnerToken> converted;
  converted.reserve(tokens.size());
  for (const std::int32_t token : tokens) {
    if (token < 0) throw std::invalid_argument("Hy4 token ID must not be negative");
    converted.push_back(static_cast<TextRunnerToken>(token));
  }
  return converted;
}

std::vector<std::int32_t> EngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const TextRunnerToken token : tokens) {
    if (token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Hy4 token ID exceeds engine range");
    }
    converted.push_back(static_cast<std::int32_t>(token));
  }
  return converted;
}

class Hy4TextRunner final : public HipTextModelRunner {
public:
  Hy4TextRunner(std::shared_ptr<Model> model,
                std::unique_ptr<Tokenizer> tokenizer, std::string model_name,
                std::uint32_t max_context, std::string artifact_fingerprint,
                std::size_t sessions, std::shared_ptr<Drafter> drafter,
                std::uint32_t draft_tokens,
                std::shared_ptr<VisionEncoder> vision)
      : model_(std::move(model)),
        drafter_(std::move(drafter)),
        vision_(std::move(vision)),
        draft_tokens_(draft_tokens),
        tokenizer_(std::move(tokenizer)),
        model_name_(std::move(model_name)),
        max_context_(max_context),
        sessions_(sessions) {
    component_identity_ = artifact_fingerprint.empty() ? ProcessIdentity(model_.get(), max_context_)
                                                       : CompatibilityIdentity(artifact_fingerprint, max_context_);
  }

  [[nodiscard]] std::string ToolDialectId() const override { return std::string(kToolDialectId); }

  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override {
    return {Hy4Tokens::kThinkOpen, Hy4Tokens::kThinkClose, {Hy4Tokens::kThinkClose}};
  }

  [[nodiscard]] std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t ram) const override {
    (void)ram;
    std::vector<std::uint32_t> taps;
    if (drafter_) taps.assign(drafter_->TargetLayers().begin(), drafter_->TargetLayers().end());
    auto adapter = std::make_unique<ContinuationAdapter>(model_, max_context_, component_identity_, std::move(taps));
    return hip::CreateHipComponentCacheResources(
        std::move(adapter), ledger,
        {[] { return CacheCeilingBytes(); },
         [](std::string_view message) { server::Logger::Warn("cache", message); },
         [](std::string_view message) { server::Logger::Info("cache", message); },
         kBatchTransferStreams + sessions_});
  }
  [[nodiscard]] std::unique_ptr<TextRunnerState> BindComponentState(cache::Adapter& adapter,
                                                                    cache::Slot& slot) const override {
    auto& hy4 = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<Hy4TextRunnerState>(hy4.GetSession(slot), adapter, slot);
  }
  void ReconcileComponentState(TextRunnerState& state, std::size_t position) const override {
    auto& hy4 = RequireState(state);
    if (hy4.session().Position() != position) throw std::logic_error("Hy4 component state position differs");
    hy4.set_position(position);
  }

  [[nodiscard]] TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_name_,
        .state_abi = std::string(kStateAbi),
        .max_context = max_context_,
        .capabilities =
            TextRunnerCapabilities{
                .supports_vision = vision_ != nullptr,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = true,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0u,
                .prefix_reuse = true,
            },
        .sampling_defaults = SamplingDefaults(),
        .scheduling = {.pass_tokens = model_->MaxChunk(),
                       .max_layer_groups = 1u,
                       .max_yield_decode_width = 1u},
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) capacity = free_bytes;
    return {
        .resident_weights_bytes = static_cast<std::size_t>(
            model_->ResidentBytes() +
            (drafter_ ? drafter_->ResidentBytes() : 0) +
            (vision_ ? vision_->ResidentBytes() : 0)),
        .state_capacity_bytes = capacity,
        .per_request_state_bytes = std::nullopt,
        .temporary_scratch_bytes = std::nullopt,
        .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = CacheCeilingBytes(),
        .requires_device_runtime_lock = true,
        .prompt_state_bytes_per_token = std::nullopt,
        .prompt_admission_budget_bytes = std::nullopt,
        .prompt_admission_available_bytes = {},
        .demand_mapped_state_bytes = std::nullopt,
    };
  }

  [[nodiscard]] std::vector<TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        static_cast<std::uint32_t>(model_->config().n_vocab), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{DecodeToken(static_cast<std::int32_t>(id)),
                                                       IsStopToken(static_cast<std::int32_t>(id))};
        });
  }
  [[nodiscard]] std::uint32_t VocabularySize() const override { return model_->config().n_vocab; }

  // Raw prompts map control-token spellings to their special tokens, as the
  // chat template's render does and as llama-server's /completion does.
  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    return RunnerTokens(tokenizer_->EncodeWithSpecials(text));
  }

  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    if (HasImages(request))
      return RunnerTokens(PrepareImages(request)->tokens);
    auto tokens = RunnerTokens(tokenizer_->EncodeWithSpecials(RenderChat(request)));
    if (tokens.empty()) return std::nullopt;
    return tokens;
  }

  [[nodiscard]] std::optional<std::string> RenderPrompt(const ChatRequest& request) const override {
    return RenderChat(request);
  }

  [[nodiscard]] ext::InitialOutputState InitialOutputState(const ChatRequest& request) const override {
    if (!request.messages.empty() && request.messages.back().role == ext::ChatRole::kAssistant) {
      return ext::InitialOutputState::kAuto;
    }
    return ReasoningEffortName(request.reasoning) == "no_think" ? ext::InitialOutputState::kContent
                                                                : ext::InitialOutputState::kReasoning;
  }

  [[nodiscard]] std::optional<TextPreparedPrompt> PreparePrompt(const ChatRequest& request) const override {
    if (HasImages(request)) {
      // Images extend the cache identity at their first row; a prefix ending
      // at or before an image holds only the earlier images.
      auto prompt = PrepareImages(request);
      auto context = std::make_shared<ImageContext>();
      context->cache_identity = prompt->cache_identity;
      for (const auto& image : prompt->images) {
        const auto identity = prompt->IdentityForPrefix(image.span.first);
        context->cache_prefixes.push_back(
            {image.span.first, {identity.begin(), identity.end()}});
      }
      TextPreparedPrompt prepared{RunnerTokens(prompt->tokens), nullptr};
      auto continuation = request;
      continuation.messages.emplace_back(ext::ChatRole::kUser, "");
      const auto continued = PrepareImages(continuation)->tokens;
      prepared.cache_prefix_tokens = static_cast<std::size_t>(
          std::mismatch(prompt->tokens.begin(), prompt->tokens.end(),
                        continued.begin(), continued.end())
              .first -
          prompt->tokens.begin());
      context->prompt = std::move(prompt);
      prepared.context = std::move(context);
      return prepared;
    }
    auto prepared = TextModelRunner::PreparePrompt(request);
    if (prepared) {
      // The stable frontier: what a following user turn keeps of this prompt.
      auto continuation = request;
      continuation.messages.emplace_back(ext::ChatRole::kUser, "");
      const auto continued = RenderAndTokenize(continuation);
      if (continued) {
        prepared->cache_prefix_tokens = static_cast<std::size_t>(
            std::mismatch(prepared->tokens.begin(), prepared->tokens.end(), continued->begin(), continued->end())
                .first -
            prepared->tokens.begin());
      }
    }
    return prepared;
  }

  [[nodiscard]] std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (HasImages(request))
      return PrepareImages(request)->tokens.size();
    return TextModelRunner::CountPromptTokens(request);
  }

  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& hy4 = RequireState(state);
    if (!context) {
      hy4.set_images(nullptr);
      return;
    }
    const auto* image = dynamic_cast<const ImageContext*>(context.get());
    if (image == nullptr || !vision_)
      throw std::invalid_argument("invalid Hy4 prompt context");
    // The same prompt keeps its encodings.
    if (hy4.images() && hy4.images()->prompt() == image->prompt.get())
      return;
    hy4.set_images(std::make_shared<ImageSource>(image->prompt, *vision_));
  }

  [[nodiscard]] std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const std::int32_t token : EngineTokens(tokens)) text += DecodeToken(token);
    return text;
  }

  void PreparePrefixReuse(TextRunnerState& state, std::span<const TextRunnerToken> prefix) const override {
    if (RequireState(state).position() != prefix.size()) {
      throw std::logic_error("Hy4 reused prefix does not match checkpoint");
    }
    RequireState(state).lookup().BeginRequest();
  }

  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState& state, std::span<const TextRunnerToken> prompt,
                                        std::size_t offset, std::size_t max_input_tokens) const override {
    auto& hy4 = RequireState(state);
    if (offset != hy4.position() || offset != hy4.session().Position()) {
      throw std::logic_error("Hy4 prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) throw std::logic_error("Hy4 prefill has no remaining input");
    std::size_t consumed =
        std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    // A step never ends inside an image: it stops before the image, or takes
    // the whole image when it starts there.
    if (const auto& images = hy4.images()) {
      for (const auto& span : images->Spans()) {
        const std::size_t end = offset + consumed;
        if (end > span.first && end < span.End())
          consumed = (span.first > offset ? span.first : span.End()) - offset;
      }
    }
    const auto tokens = EngineTokens(prompt.subspan(offset, consumed));
    hy4.lookup().BeginRequest();
    std::string error;
    if (!hy4.session().Prompt(tokens, &error))
      throw std::runtime_error("Hy4 prefill failed: " + error);
    hy4.set_position(offset + consumed);
    return {.consumed_tokens = consumed, .decode_ready = offset + consumed == prompt.size()};
  }

  [[nodiscard]] TextDecodeSelection SelectNext(TextRunnerState& state,
                                               sampling::SamplerState& sampler) const override {
    auto& hy4 = RequireState(state);
    if (hy4.position() >= max_context_) return Stop();
    return Select(hy4, sampler);
  }

  [[nodiscard]] std::optional<TextDecodeSelection> PreviewFirstToken(TextRunnerState& state,
                                                                     sampling::SamplerState& sampler) const override {
    auto& hy4 = RequireState(state);
    if (hy4.position() >= max_context_) return Stop();
    return Select(hy4, sampler);
  }

  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& hy4 = RequireState(state);
    const std::int32_t one[] = {EngineTokens(std::span(&token, 1)).front()};
    std::string error;
    if (!hy4.session().Step(one, &error)) throw std::runtime_error("Hy4 decode failed: " + error);
    hy4.set_position(hy4.position() + 1);
  }

  // Sample the anchor from the frontier, propose tokens after it, verify
  // anchor and proposals in one pass, and keep the proposals the target's own
  // draws agree with. A disagreeing draw is deferred to the next step, which
  // samples it from the same logits, so sampling stays exact. Proposals are
  // a prompt-lookup copy of the session's own tokens when the context
  // repeats them, else DFly2's drafts; without either the step is a plain
  // one.
  [[nodiscard]] TextDecodeStep DecodeStep(TextRunnerState& state, std::size_t max_tokens,
                                          sampling::SamplerState& sampler) const override {
    auto& hy4 = RequireState(state);
    Session& session = hy4.session();
    TextDecodeStep step;
    if (max_tokens == 0)
      return step;
    if (hy4.position() >= max_context_) {
      step.stop = true;
      return step;
    }
    sampling::SamplerState work = sampler;
    TextDecodeSelection anchor = Select(hy4, work);
    const std::size_t room = std::min<std::size_t>(
        max_tokens - 1, max_context_ - hy4.position() - 1);
    std::vector<std::int32_t> proposals;
    lookup::PromptLookup<std::int32_t>::Match copy{};
    if (!anchor.stop && room > 0) {
      copy = Copy(hy4, static_cast<std::int32_t>(anchor.token), work, room,
                  &proposals);
    }
    const bool draft = !anchor.stop && proposals.empty() && drafter_ &&
                       session.TapsReady() && room > 0;
    if (anchor.stop || (proposals.empty() && !draft)) {
      sampler.CopyDrawStateFrom(work);
      step.stop = anchor.stop;
      if (anchor.stop) {
        step.terminal_logprobs = std::move(anchor.logprobs);
      } else {
        const TextRunnerToken token = anchor.token;
        step.selections.push_back(std::move(anchor));
        Advance(state, token);
      }
      return step;
    }
    const auto anchor_token = static_cast<std::int32_t>(anchor.token);
    work.Accept(anchor.token);
    step.selections.push_back(std::move(anchor));

    std::string error;
    const auto t0 = std::chrono::steady_clock::now();
    if (draft) {
      const auto count = static_cast<std::uint32_t>(
          std::min<std::size_t>(draft_tokens_, room));
      if (!drafter_->Draft(session, anchor_token, count, min_accept_,
                           &proposals, &error)) {
        throw std::runtime_error("Hy4 draft failed: " + error);
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::vector<std::int32_t> tokens{anchor_token};
    tokens.insert(tokens.end(), proposals.begin(), proposals.end());
    if (tokens.size() == 1) {
      if (!session.Step(tokens, &error))
        throw std::runtime_error("Hy4 decode failed: " + error);
    } else if (!session.Verify(tokens, &error)) {
      throw std::runtime_error("Hy4 verify failed: " + error);
    }
    const auto t2 = std::chrono::steady_clock::now();
    std::uint32_t kept = 1;
    for (std::uint32_t i = 0; i < proposals.size(); ++i) {
      const auto logits = session.PassLogits(i);
      const auto token = static_cast<std::int32_t>(work.Sample(logits));
      if (token != proposals[i]) {
        work.DeferSample(static_cast<sampling::TokenId>(token));
        break;
      }
      auto logprobs = sampling::CollectLogprobs(logits, static_cast<TextRunnerToken>(token), sampler.config().logprobs);
      if (hy4.stop_at_eos() && IsStopToken(token)) {
        step.stop = true;
        step.terminal_logprobs = std::move(logprobs);
        break;
      }
      work.Accept(static_cast<sampling::TokenId>(token));
      ++kept;
      step.selections.push_back({.stop = false,
                                 .token = static_cast<TextRunnerToken>(token),
                                 .piece = DecodeToken(token),
                                 .logprobs = std::move(logprobs)});
    }
    if (tokens.size() > 1)
      session.Keep(kept);
    hy4.set_position(session.Position());
    if (copy.length != 0)
      hy4.lookup().Observe(copy.start, proposals.size(), kept - 1);
    if (profile_) {
      Profile& p = *profile_;
      p.rounds += 1;
      (copy.length != 0 ? p.copy_rounds : p.draft_rounds) += 1;
      (copy.length != 0 ? p.copied : p.drafted) += proposals.size();
      (copy.length != 0 ? p.copy_accepted : p.accepted) += kept - 1;
      p.draft_s += std::chrono::duration<double>(t1 - t0).count();
      p.verify_s += std::chrono::duration<double>(t2 - t1).count();
      if (p.rounds % 64 == 0) {
        const double draft_rounds = std::max<double>(1, p.draft_rounds);
        server::Logger::Info(
            "dfly",
            "event=hy4_dfly_profile rounds=" + std::to_string(p.rounds) +
                " draft_ms=" + std::to_string(1e3 * p.draft_s / draft_rounds) +
                " verify_ms=" + std::to_string(1e3 * p.verify_s / p.rounds) +
                " drafted=" + std::to_string(double(p.drafted) / draft_rounds) +
                " accepted=" +
                std::to_string(double(p.accepted) / draft_rounds) +
                " copy_rounds=" + std::to_string(p.copy_rounds) +
                " copied=" + std::to_string(p.copied) +
                " copy_accepted=" + std::to_string(p.copy_accepted));
      }
    }
    sampler.CopyDrawStateFrom(work);
    step.draft_rounds = 1;
    step.draft_tokens = proposals.size();
    step.draft_accepted_tokens = kept - 1;
    return step;
  }

  [[nodiscard]] std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return RequireState(state).position();
  }

private:
  [[nodiscard]] static bool HasImages(const ChatRequest& request) {
    return std::ranges::any_of(request.messages,
                               [](const auto& m) { return !m.images.empty(); });
  }
  [[nodiscard]] std::shared_ptr<const ImagePrompt> PrepareImages(
      const ChatRequest& request) const {
    if (!vision_)
      throw std::invalid_argument(
          "image input requires the Hy4 Flash --mmproj BF16 sidecar");
    return std::make_shared<const ImagePrompt>(PrepareImagePrompt(
        *tokenizer_, request, vision_->Identity(), max_context_));
  }

  [[nodiscard]] std::string DecodeToken(std::int32_t token) const { return tokenizer_->Decode(token); }
  [[nodiscard]] bool IsStopToken(std::int32_t token) const { return token == tokenizer_->eos(); }

  static TextDecodeSelection Stop() {
    TextDecodeSelection selection;
    selection.stop = true;
    return selection;
  }

  // Prompt lookup (PromptLookup): up to `cap` of the session's committed
  // tokens that followed an earlier occurrence of the context ending in
  // `anchor`, as far as every competing occurrence agrees. Greedy requests
  // need 12 matching tokens and copy up to 15, filling a 16-token verify
  // pass; sampled ones need 24, copy only from their prompt and at most 7,
  // since their draws leave a long copy more often (E34). A request's first
  // step copies nothing. Returns the source, or a zero-length match.
  lookup::PromptLookup<std::int32_t>::Match Copy(
      Hy4TextRunnerState& hy4, std::int32_t anchor,
      const sampling::SamplerState& sampler, std::size_t cap,
      std::vector<std::int32_t>* copies) const {
    using Lookup = lookup::PromptLookup<std::int32_t>;
    Lookup& lk = hy4.lookup();
    const std::vector<std::int32_t>& committed = hy4.session().History();
    if (!lk.Step(committed.size()))
      return {};
    const bool sampled = !sampler.config().can_use_unmodified_argmax();
    cap = std::min<std::size_t>(cap, sampled ? 7 : Session::kTapRows - 1);
    lk.Extend(committed, sampled);
    std::size_t agreed = 0;
    const auto match = lk.Find(
        committed, std::span<const std::int32_t>(&anchor, 1),
        sampled ? Lookup::kSampledMinMatch : Lookup::kMinMatch, false, &agreed);
    if (match.length == 0)
      return {};
    while (copies->size() < std::min(cap, agreed)) {
      const std::int32_t token = committed[match.start + copies->size()];
      copies->push_back(token);
      if (IsStopToken(token))
        break;
    }
    return copies->empty() ? Lookup::Match{} : match;
  }

  TextDecodeSelection Select(Hy4TextRunnerState& hy4, sampling::SamplerState& sampler) const {
    const auto& logits = hy4.session().LastLogits();
    if (logits.size() != model_->config().n_vocab) throw std::runtime_error("Hy4 token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    const bool stop = hy4.stop_at_eos() && IsStopToken(token);
    return {.stop = stop,
            .token = static_cast<TextRunnerToken>(token),
            .piece = stop ? std::string{} : DecodeToken(token),
            .logprobs = sampling::CollectLogprobs(logits, static_cast<TextRunnerToken>(token),
                                                  sampler.config().logprobs)};
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<Drafter> drafter_;
  std::shared_ptr<VisionEncoder> vision_;
  std::uint32_t draft_tokens_{0};
  // Draft while the confidence head's chance that every draft so far is
  // accepted stays at least this (HY4_DFLY_MIN_ACCEPT overrides).
  float min_accept_{std::getenv("HY4_DFLY_MIN_ACCEPT") ? std::strtof(std::getenv("HY4_DFLY_MIN_ACCEPT"), nullptr)
                                                       : kDefaultMinAccept};
  // HY4_DFLY_PROFILE=1: mean draft and verify times, logged every 64 steps.
  struct Profile {
    std::uint64_t rounds{0}, draft_rounds{0}, accepted{0}, drafted{0};
    std::uint64_t copy_rounds{0}, copied{0}, copy_accepted{0};
    double draft_s{0}, verify_s{0};
  };
  std::unique_ptr<Profile> profile_{std::getenv("HY4_DFLY_PROFILE") ? std::make_unique<Profile>() : nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  std::string model_name_;
  std::uint32_t max_context_;
  std::size_t sessions_;
  cache::Identity component_identity_;
};

}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateHy4TextRunner(const ext::TextRunnerOptions& options,
                                                          std::string* error) {
  const auto set_error = [&](std::string message) {
    if (error != nullptr) *error = std::move(message);
  };
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled) {
    set_error("Hy4 Flash drafts with DFly2: pass --dfly-model <checkpoint> instead of --speculative");
    return nullptr;
  }
  const std::string* dfly_option = ext::FindOption<std::string>(options.provider_options, "--dfly-model");
  const bool use_dfly = dfly_option != nullptr && !dfly_option->empty();
  std::string draft_path = use_dfly ? *dfly_option : std::string{};
  std::int64_t draft_tokens_option = kDefaultDraftTokens;
  if (const auto* value = ext::FindOption<std::int64_t>(options.provider_options, "--dfly-draft-tokens")) {
    draft_tokens_option = *value;
  }
  if (draft_tokens_option < 1 || draft_tokens_option >= Drafter::kBlock) {
    set_error("--dfly-draft-tokens must be 1 to 7");
    return nullptr;
  }
  if (use_dfly) {
    // A checkpoint directory holds the drafter in its one safetensors file.
    std::error_code ec;
    if (std::filesystem::is_directory(draft_path, ec)) {
      std::vector<std::filesystem::path> files;
      for (const auto& entry : std::filesystem::directory_iterator(draft_path, ec)) {
        if (entry.path().extension() == ".safetensors") files.push_back(entry.path());
      }
      if (files.size() != 1) {
        set_error("Hy4 DFly2: expected one .safetensors file in " + draft_path);
        return nullptr;
      }
      draft_path = files.front().string();
    }
  }
  if (options.session_count == 0) {
    set_error("HTTP session count must be at least one");
    return nullptr;
  }
  std::shared_ptr<const core::GgufReader> reader = options.reader;
  if (!reader) {
    std::string load_error;
    auto owner = core::GgufReader::OpenFile(options.model_path, &load_error, {.allow_unsized_tensors = true});
    if (!owner) {
      set_error("Failed to open GGUF: " + load_error);
      return nullptr;
    }
    reader = std::shared_ptr<const core::GgufReader>(std::move(owner));
  }
  std::string load_error;
  auto tokenizer = Tokenizer::Load(*reader, &load_error);
  if (!tokenizer) {
    set_error("Hy4 tokenizer: " + load_error);
    return nullptr;
  }
  std::uint32_t max_context = options.max_context;
  if (max_context == 0) {
    const std::uint64_t native = reader->GetMetadataUint64("hy_v4_flash.context_length").value_or(0);
    max_context = static_cast<std::uint32_t>(std::min<std::uint64_t>(native ? native : kDefaultContext,
                                                                     kDefaultContext));
  }
  if (max_context < 2) {
    set_error("Hy4 context must hold at least two tokens");
    return nullptr;
  }
  std::unique_ptr<Model> loaded =
      Model::Load(options.model_path, &load_error, max_context);
  if (!loaded) {
    set_error("Failed to load Hy4 Flash: " + load_error);
    return nullptr;
  }
  std::shared_ptr<Model> model(std::move(loaded));
  max_context = model->SessionContext(max_context);
  std::shared_ptr<Drafter> drafter;
  const auto draft_tokens = static_cast<std::uint32_t>(draft_tokens_option);
  if (use_dfly) {
    std::unique_ptr<Drafter> loaded_drafter = Drafter::Load(draft_path, *model, &load_error);
    if (!loaded_drafter) {
      set_error("Failed to load the DFly2 drafter: " + load_error);
      return nullptr;
    }
    drafter = std::move(loaded_drafter);
    server::Logger::Info("loader", "event=hy4_dfly_loaded resident_bytes=" + std::to_string(drafter->ResidentBytes()) +
                                       " draft_tokens=" + std::to_string(draft_tokens));
  }
  // The vision sidecar: --mmproj, else the one mmproj*.gguf beside the model.
  std::string vision_path = options.vision_model_path;
  if (vision_path.empty()) {
    std::error_code ec;
    std::vector<std::filesystem::path> found;
    const auto directory =
        std::filesystem::path(options.model_path).parent_path();
    for (const auto& entry : std::filesystem::directory_iterator(
             directory.empty() ? "." : directory, ec)) {
      const auto name = entry.path().filename().string();
      if (name.starts_with("mmproj") && entry.path().extension() == ".gguf")
        found.push_back(entry.path());
    }
    if (found.size() == 1)
      vision_path = found.front().string();
  }
  std::shared_ptr<VisionEncoder> vision;
  if (!vision_path.empty()) {
    std::unique_ptr<VisionEncoder> loaded_vision = VisionEncoder::Load(
        vision_path, model->config().n_embd, model->stream(), &load_error);
    if (!loaded_vision) {
      set_error("Failed to load the Hy4 Flash vision sidecar: " + load_error);
      return nullptr;
    }
    vision = std::move(loaded_vision);
    model->SetImageToken(tokenizer->Find(Hy4Tokens::kImagePad));
    server::Logger::Info("loader", "event=hy4_vision_loaded resident_bytes=" +
                                       std::to_string(vision->ResidentBytes()));
  }
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  try {
    if (!options.disk_cache.directory.empty() && model_fingerprint.empty()) {
      model_fingerprint = core::GgufIdentityHex(*reader);
    }
    const std::string name(reader->GetMetadataString("general.name").value_or("Hy4 Flash"));
    server::Logger::Info("loader", "event=hy4_loaded resident_bytes=" + std::to_string(model->ResidentBytes()) +
                                       " context=" + std::to_string(max_context));
    return std::make_shared<Hy4TextRunner>(
        std::move(model), std::move(tokenizer), name, max_context,
        std::move(model_fingerprint), options.session_count, std::move(drafter),
        draft_tokens, std::move(vision));
  } catch (const std::exception& e) {
    set_error(std::string("Hy4 load failed: ") + e.what());
    return nullptr;
  }
}

}  // namespace gufo::models::hy4_flash
