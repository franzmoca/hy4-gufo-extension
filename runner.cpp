#include "runner.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
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
#include "runtime/hy4_runtime.hpp"
#include "runtime/tokenizer.hpp"
#include "text_contract.hpp"

namespace gufo::models::hy4_flash {
namespace {

using namespace gufo::ext;

constexpr std::string_view kStateAbi = "hy4-flash-gfx1151-state-v1";
// Without --context: the native context needs 7 GiB of latent rows per
// session at 256K, so serving defaults to this.
constexpr std::uint32_t kDefaultContext = 32768;
// Prompt-cache transfer streams besides each session's own.
constexpr std::size_t kBatchTransferStreams = 16;

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

class Hy4TextRunnerState final : public TextRunnerState {
public:
  Hy4TextRunnerState(Session& session, cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}

  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    position_ = 0;
  }

  [[nodiscard]] Session& session() const { return *session_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void set_position(std::size_t position) noexcept { position_ = position; }

private:
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
  Hy4TextRunner(std::shared_ptr<Model> model, std::unique_ptr<Tokenizer> tokenizer, std::string model_name,
                std::uint32_t max_context, std::string artifact_fingerprint, std::size_t sessions)
      : model_(std::move(model)),
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
    auto adapter = std::make_unique<ContinuationAdapter>(model_, max_context_, component_identity_);
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
                .supports_vision = false,
                .incremental_prefill = true,
                .final_token_advance_required = false,
                .incremental_text_is_exact = true,
                .multi_token_decode = false,
                .batched_multi_token_decode = false,
                .batched_multi_token_decode_max_width = 0u,
                .prefix_reuse = true,
            },
        .sampling_defaults = SamplingDefaults(),
        .scheduling = {.pass_tokens = Model::kMaxChunk, .max_layer_groups = 1u, .max_yield_decode_width = 1u},
    };
  }

  [[nodiscard]] TextRunnerResourceClaim ResourceClaim() const override {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::optional<std::size_t> capacity;
    if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) capacity = free_bytes;
    return {
        .resident_weights_bytes = static_cast<std::size_t>(model_->ResidentBytes()),
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

  [[nodiscard]] std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    std::string text;
    for (const std::int32_t token : EngineTokens(tokens)) text += DecodeToken(token);
    return text;
  }

  void PreparePrefixReuse(TextRunnerState& state, std::span<const TextRunnerToken> prefix) const override {
    if (RequireState(state).position() != prefix.size()) {
      throw std::logic_error("Hy4 reused prefix does not match checkpoint");
    }
  }

  [[nodiscard]] TextPrefillStep Prefill(TextRunnerState& state, std::span<const TextRunnerToken> prompt,
                                        std::size_t offset, std::size_t max_input_tokens) const override {
    auto& hy4 = RequireState(state);
    if (offset != hy4.position() || offset != hy4.session().Position()) {
      throw std::logic_error("Hy4 prefill offset does not match retained state");
    }
    if (offset >= prompt.size()) throw std::logic_error("Hy4 prefill has no remaining input");
    const std::size_t consumed = std::min<std::size_t>(max_input_tokens, prompt.size() - offset);
    const auto tokens = EngineTokens(prompt.subspan(offset, consumed));
    std::string error;
    if (!hy4.session().Step(tokens, &error)) throw std::runtime_error("Hy4 prefill failed: " + error);
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

  [[nodiscard]] std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return RequireState(state).position();
  }

private:
  [[nodiscard]] std::string DecodeToken(std::int32_t token) const { return tokenizer_->Decode(token); }
  [[nodiscard]] bool IsStopToken(std::int32_t token) const { return token == tokenizer_->eos(); }

  static TextDecodeSelection Stop() {
    TextDecodeSelection selection;
    selection.stop = true;
    return selection;
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
    set_error("Hy4 Flash has no speculative decoding yet");
    return nullptr;
  }
  if (!options.vision_model_path.empty()) {
    set_error("Hy4 Flash has no vision encoder yet");
    return nullptr;
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
  std::unique_ptr<Model> loaded = Model::Load(options.model_path, &load_error);
  if (!loaded) {
    set_error("Failed to load Hy4 Flash: " + load_error);
    return nullptr;
  }
  std::shared_ptr<Model> model(std::move(loaded));
  max_context = model->SessionContext(max_context);
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  try {
    if (!options.disk_cache.directory.empty() && model_fingerprint.empty()) {
      model_fingerprint = core::GgufIdentityHex(*reader);
    }
    const std::string name(reader->GetMetadataString("general.name").value_or("Hy4 Flash"));
    server::Logger::Info("loader", "event=hy4_loaded resident_bytes=" + std::to_string(model->ResidentBytes()) +
                                       " context=" + std::to_string(max_context));
    return std::make_shared<Hy4TextRunner>(std::move(model), std::move(tokenizer), name, max_context,
                                           std::move(model_fingerprint), options.session_count);
  } catch (const std::exception& e) {
    set_error(std::string("Hy4 load failed: ") + e.what());
    return nullptr;
  }
}

}  // namespace gufo::models::hy4_flash
