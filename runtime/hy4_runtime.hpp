#ifndef HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_
#define HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_

// Hy4 Flash model loading and evaluation on HIP. The graph follows
// llama.cpp's hy_v4_flash (ggml-org/llama.cpp#32517); docs/PLAN.md describes
// the design.

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gufo/ext/experimental/state_memory.hpp>
#include <gufo/ext/state.hpp>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "runtime/hy4_config.hpp"

namespace gufo::models::hy4_flash {

class EngramReader;

/// A resident weight: device bytes, ggml type and GGUF dimensions.
struct Tensor {
  const void* data = nullptr;
  int type = 0;
  std::vector<std::uint64_t> dims;
  [[nodiscard]] explicit operator bool() const { return data != nullptr; }
  [[nodiscard]] const float* f32() const { return static_cast<const float*>(data); }
};

struct Layer {
  Tensor attn_norm, attn_sinks, wq_a, q_a_norm, wq_b, wkv_a, kv_a_norm, wk_b, wv_b, wo, attn_gate,
      attn_post_norm;
  Tensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  Tensor ffn_norm, ffn_post_norm;
  // Dense lead layer.
  Tensor ffn_gate, ffn_up, ffn_down;
  // MoE.
  Tensor gate_inp, exp_bias, routed_down, routed_up, gate_exps, up_exps, down_exps, gate_shexp,
      up_shexp, down_shexp;
  // wk_b split for prefill passes: values as binary16 bits, block scales.
  const std::uint16_t* wk_b_q = nullptr;
  const float* wk_b_d = nullptr;
  // Indexer.
  Tensor idx_k, idx_k_norm, idx_value, idx_gate, idx_ape, idx_comp_norm, idx_q_b, idx_proj;
  // Engram.
  Tensor engram_kv, engram_kv_b, engram_gate, engram_lookup_norm, engram_pre_gamma,
      engram_hc_scale;
};

struct Weights {
  Tensor tok_embd, output, output_norm, hc_head_fn, hc_head_base, hc_head_scale;
  std::vector<Layer> layers;
};

class Session;

/// An image's rows in a session's input: `height` merged rows of `width`
/// cells, each followed by a newline row, from position `first`.
struct ImageSpan {
  std::uint32_t first = 0;
  std::uint32_t height = 0;
  std::uint32_t width = 0;
  [[nodiscard]] std::uint32_t Rows() const { return height * (width + 1); }
  [[nodiscard]] std::uint32_t End() const { return first + Rows(); }
};

/// A prompt's images: where their rows are and their input embeddings.
class ImageInputs {
public:
  virtual ~ImageInputs() = default;
  /// In position order, not overlapping.
  [[nodiscard]] virtual std::span<const ImageSpan> Spans() const = 0;
  /// Device rows [Rows()][n_embd] of image `index`, encoded on first use on
  /// the model's stream; null on failure.
  [[nodiscard]] virtual const float* Embeddings(std::size_t index,
                                                std::string* error) = 0;
};

/// The loaded model. One evaluation runs at a time (scratch is shared).
class Model {
public:
  /// Most tokens one evaluation pass can take (longer spans run in passes);
  /// MaxChunk() is the width this model was loaded for.
  static constexpr std::uint32_t kMaxChunk = 8192;
  /// Contexts up to this many tokens take kMaxChunk-token passes; longer
  /// ones take 4,096, leaving the device memory to the sessions (E43).
  static constexpr std::uint32_t kWideContext = 262144;
  /// Passes up to this many tokens use matvecs; longer ones use GEMMs.
  static constexpr std::uint32_t kVecMax = 16;
  /// Rows of each indexer layer's value and gate ring.
  static constexpr std::uint32_t kIndexerRing = kMaxChunk + 8;
  /// Tokens whose indexer scores one launch computes ([kScoreRows][pools]
  /// per session: 2.1 GB at 1M tokens).
  static constexpr std::uint32_t kScoreRows = 2048;

  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// `trunk_path` is the first GGUF shard; the reader maps the others (the
  /// Engram table) beside it.
  /// `max_context` (0: unknown) sets the pass width, MaxChunk().
  [[nodiscard]] static std::unique_ptr<Model> Load(
      const std::string& trunk_path, std::string* error,
      std::uint32_t max_context = 0);
  /// Most tokens one evaluation pass takes: kMaxChunk for contexts up to
  /// kWideContext, else 4,096. Results do not depend on it (E18, E36).
  [[nodiscard]] std::uint32_t MaxChunk() const { return max_chunk_; }
  /// Prompt rows that stay backed. Where device memory is demand-mapped,
  /// the prompt scratch's rows beyond them take memory only while a wider
  /// pass needs them.
  static constexpr std::uint32_t kResidentRows = 2048;
  /// The widest pass a prompt of `tokens` tokens can take now: MaxChunk()
  /// once the wide rows are mapped (mapping them for a prompt wider than
  /// kResidentRows when the device and host can spare them), else
  /// kResidentRows. Results do not depend on it.
  [[nodiscard]] std::uint32_t PassRows(std::size_t tokens) const;
  /// Returns the wide rows' memory unless a wide pass is under way (paused
  /// between layer groups). Decode passes call it.
  void ReleaseWidePasses() const;

  [[nodiscard]] const Config& config() const { return config_; }
  [[nodiscard]] const Weights& weights() const { return weights_; }
  [[nodiscard]] hipStream_t stream() const { return stream_; }
  [[nodiscard]] std::uint64_t ResidentBytes() const { return resident_bytes_; }

  [[nodiscard]] std::unique_ptr<Session> CreateSession(std::uint32_t max_context,
                                                       std::string* error);
  /// Declares the buffers one session of `max_context` holds: full-context
  /// KV and pooled indexer keys as kContext regions, fixed rings and scratch
  /// as kFixed allocations.
  [[nodiscard]] mem::StateDeclaration DeclareSessionState(
      std::uint32_t max_context) const;
  /// Bytes one session of `max_context` claims (page-rounded when
  /// demand-mapped, plus any tap buffers if enabled).
  [[nodiscard]] std::size_t SessionStateBytes(std::uint32_t max_context,
                                              std::size_t tap_layers = 0) const;
  /// Bytes of demand-mapped context buffers for one session of `max_context`
  /// (0 when device memory is not demand-mapped).
  [[nodiscard]] std::size_t DemandMappedBytes(std::uint32_t max_context) const;
  /// The context a session created for `max_context` holds (a multiple of kpool).
  [[nodiscard]] std::uint32_t SessionContext(std::uint32_t max_context) const;
  /// Latent rows a layer keeps for a session of `max_context` (Session::KvRing).
  [[nodiscard]] std::uint32_t KvRing(std::uint32_t layer, std::uint32_t max_context) const;

  /// Hashes `tokens` (positions pos0..) against `history` (the tokens before
  /// them) for Engram module `module`: its table rows `rows` [token][col] and
  /// `valid` [token][col].
  void EngramRows(std::span<const std::int32_t> history,
                  std::span<const std::int32_t> tokens, std::uint32_t pos0,
                  std::uint32_t module, std::uint64_t* rows,
                  float* valid) const;
  /// Reads the FP8 Engram table's rows (EngramRows).
  [[nodiscard]] EngramReader& Engram() const { return *engram_reader_; }
  /// The image row token: Engram treats it as padding, and a session takes
  /// its rows' embeddings from ImageInputs.
  void SetImageToken(std::int32_t token) { image_token_ = token; }
  [[nodiscard]] std::int32_t ImageToken() const { return image_token_; }

private:
  friend class Session;
  friend class Drafter;
  Model() = default;
  bool AllocScratch(std::string* error);
  bool RouterToBf16(std::string* error);

  struct Shards;
  std::unique_ptr<Shards> shards_;
  Config config_;
  Weights weights_;
  std::vector<void*> allocations_;
  std::uint64_t resident_bytes_ = 0;
  std::uint32_t max_chunk_ = 4096;
  hipStream_t stream_ = nullptr;
  void* blas_ = nullptr;  // hipblasHandle_t
  void* gemv_scratch_ = nullptr;                 // hy4_gemv's, for stream_
  std::unique_ptr<EngramReader> engram_reader_;  // FP8 rows, direct reads
  std::int32_t image_token_ = -1;

  // Scratch for one evaluation of up to MaxChunk() tokens (device unless
  // noted).
  struct Scratch;
  // `resident` of the `rows` rows backed from `arena` (none: all resident);
  // `measure` adds up the arena bytes instead of allocating.
  bool AllocScratchRows(Scratch& s, std::uint32_t rows,
                        std::uint32_t resident = 0,
                        mem::ScratchArena* arena = nullptr,
                        std::size_t* measure = nullptr);
  std::unique_ptr<mem::ScratchArena> arena_;  // scratch_'s wide rows
  mutable bool wide_pass_running_ = false;
  std::unique_ptr<Scratch> scratch_;         // prompt passes, MaxChunk() rows
  std::unique_ptr<Scratch> decode_scratch_;  // other passes, kVecMax rows
  // Resident decode logits tracking and device sampling scratch.
  std::uint64_t logits_gen_ = 0;
  std::vector<Session*> logits_owners_;
  float* argmax_work_ = nullptr;
  std::int32_t* argmax_dev_ = nullptr;
  std::int32_t* argmax_host_ = nullptr;  // pinned
  void* sample_scratch_ = nullptr;
  std::uint32_t* sample_mask_ = nullptr;
  std::vector<std::uint32_t> sample_mask_words_;
  void* sample_out_host_ = nullptr;  // pinned hy4_sample_result
  // A prompt pass is paused between layer groups and holds scratch_.
  bool prompt_paused_ = false;
};

/// One sequence's state: KV rows, indexer pools and token history.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Evaluates `tokens` at the next positions; with `logits` set, writes the
  /// logits of every evaluated token ([n][n_vocab], host) when `all_logits`,
  /// else of the last one. `prompt` passes take the batched kernels at any
  /// size, so a prompt gives the same results however it is cut into passes;
  /// other passes of up to kVecMax tokens compute what one-token decoding
  /// would.
  [[nodiscard]] bool Evaluate(std::span<const std::int32_t> tokens,
                              std::vector<float>* logits, bool all_logits,
                              std::string* error, bool prompt = false);

  /// Evaluates `tokens` and keeps the last one's logits (LastLogits).
  [[nodiscard]] bool Step(std::span<const std::int32_t> tokens,
                          std::string* error);
  /// The logits after the last Step, [n_vocab] on the host (downloaded lazily
  /// if still resident on the device).
  [[nodiscard]] std::vector<float>& LastLogits();
  /// Cached finite argmax of the frontier row (or row `row` of the last
  /// Verify), if computed on the device.
  [[nodiscard]] std::optional<std::int32_t> LastArgmax() const {
    return last_argmax_;
  }
  [[nodiscard]] std::optional<std::int32_t> Argmax(std::uint32_t row) const {
    if (row < pass_argmax_.size() && pass_argmax_[row] >= 0)
      return pass_argmax_[row];
    return std::nullopt;
  }
  /// Device-side sampling on a resident Verify row (or the frontier row for
  /// DeviceSampleLast). Returns false if the row is not resident or the draw
  /// is degenerate, in which case the caller falls back to host logits.
  [[nodiscard]] bool DeviceSample(std::uint32_t row, const std::uint32_t* mask,
                                  std::uint32_t mask_words, float temperature,
                                  double u, std::int32_t* token) const;
  [[nodiscard]] bool DeviceSampleLast(const std::uint32_t* mask,
                                      std::uint32_t mask_words,
                                      float temperature, double u,
                                      std::int32_t* token) const;
  /// Step for prompt tokens (Evaluate's `prompt`).
  [[nodiscard]] bool Prompt(std::span<const std::int32_t> tokens,
                            std::string* error) {
    return Evaluate(tokens, &last_logits_, false, error, true);
  }
  /// Prompt for one pass (at most Model::MaxChunk() tokens) that also keeps,
  /// in row_logits[i], the logits after token rows[i] (ascending, before the
  /// last): those a pass ending after that token keeps, bit for bit, since a
  /// pass's rows do not depend on where it ends.
  [[nodiscard]] bool PromptThrough(std::span<const std::int32_t> tokens,
                                   std::span<const std::uint32_t> rows,
                                   std::vector<std::vector<float>>* row_logits,
                                   std::string* error);

  /// Evaluates `tokens` (at most kTapRows) and keeps every one's logits
  /// (PassLogits) until Keep.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens, std::string* error);
  /// Decode passes of several sessions as one pass: each session's tokens
  /// as its own Step (one token) or Verify (more), with the same results;
  /// at most Model::kVecMax tokens in all.
  [[nodiscard]] static bool DecodeBatch(
      std::span<Session* const> sessions,
      std::span<const std::span<const std::int32_t>> tokens,
      std::string* error);
  /// Logits after token `row` of the last Verify, [n_vocab] on the host.
  [[nodiscard]] std::span<const float> PassLogits(std::uint32_t row) const;
  /// Keeps the first `tokens` (>= 1) tokens of the last Verify and forgets
  /// the rest; LastLogits and the taps then follow the last kept token.
  void Keep(std::uint32_t tokens);

  /// Prompt passes from now on pause after about every n_layer / `groups`
  /// layers and call `yield` (no pause with groups < 2 or no yield). The
  /// yield may run other sessions' decode passes, never a prompt pass.
  void SetPrefillYield(std::size_t groups, std::function<void()> yield) {
    yield_groups_ = yield ? groups : 1;
    yield_ = std::move(yield);
  }

  /// Draft taps: each evaluation leaves its last kTapRows tokens' final
  /// hidden (after the output norm) and the indexer selections of the tapped
  /// layers on the device, for a drafter to read at the frontier.
  static constexpr std::uint32_t kTapRows = 16;
  [[nodiscard]] bool EnableTaps(std::span<const std::uint32_t> layers, std::string* error);
  /// Whether the taps hold the frontier token (a restored state has none).
  [[nodiscard]] bool TapsReady() const { return tap_ready_; }
  /// The frontier token's final hidden [n_embd] (device).
  [[nodiscard]] const float* TapHidden() const;
  /// The frontier token's selection on tapped layer `k` [count] (device,
  /// -1 unused, possibly past the frontier), or null when that layer
  /// attended to every earlier position.
  [[nodiscard]] const std::int32_t* TapSelection(std::uint32_t k, std::uint32_t* count) const;

  /// Forgets every evaluated token; the next evaluation starts at 0.
  void Reset();
  /// The images of the prompt being evaluated (null: none). Rows of an image
  /// span take its embeddings and rotate at 3D positions (docs/PLAN.md,
  /// decision 19).
  void SetImages(std::shared_ptr<ImageInputs> images) {
    images_ = std::move(images);
  }
  /// A text token at position p rotates at p + RopeDelta().
  [[nodiscard]] std::int32_t RopeDelta() const { return rope_delta_; }
  /// The temporal, height and width rope positions of evaluated position p.
  [[nodiscard]] std::array<std::int32_t, 3> RopeAt(std::uint32_t p) const {
    return rope_.at(p);
  }
  /// After BeginRestore: the restored state's rope offset, and the rope
  /// position of `group_first` (the first member of the open pool group).
  void RestoreRope(std::int32_t delta, std::uint32_t group_first,
                   std::array<std::int32_t, 3> group_rope);

  /// Called before an evaluation of positions [first, end) writes any row.
  using WriteHook = void (*)(void* user, std::uint32_t first, std::uint32_t end);
  void SetWriteHook(WriteHook hook, void* user) {
    write_hook_ = hook;
    write_user_ = user;
  }

  /// Begins restoring a state of `tokens` evaluated tokens: backs the
  /// session's demand-mapped rows up to `tokens` and returns the token
  /// history to fill (null on backing failure); the caller fills the rows,
  /// rings and logits.
  [[nodiscard]] std::int32_t* BeginRestore(std::uint32_t tokens);

  /// The session's demand-mapped state account (null without demand mapping).
  [[nodiscard]] std::shared_ptr<mem::StateAccount> StateAccount() const {
    return state_ ? state_->account() : nullptr;
  }

  // Device state, for prompt-cache capture and restore. Every layer keeps
  // latent rows [KvRing(layer)][kv_lora + n_rot] (position p in row
  // p % KvRing); indexer layers also keep a value | gate ring
  // [kIndexerRing][2 idx_dim] and pooled keys [max_context / kpool + 1][idx_dim].
  [[nodiscard]] std::uint16_t* KvRows(std::uint32_t layer) const {
    return cache_[layer];
  }
  [[nodiscard]] std::uint32_t KvRing(std::uint32_t layer) const { return ring_[layer]; }
  [[nodiscard]] float* IndexerRing(std::uint32_t layer) const { return idx_rows_[layer]; }
  [[nodiscard]] float* IndexerPools(std::uint32_t layer) const { return idx_pooled_[layer]; }

  [[nodiscard]] std::uint32_t Position() const {
    return static_cast<std::uint32_t>(history_.size());
  }
  [[nodiscard]] std::uint32_t MaxContext() const { return max_context_; }
  /// Host seconds spent gathering Engram rows since creation.
  /// Time evaluations waited for Engram rows (gathered beside the layers).
  [[nodiscard]] double EngramSeconds() const { return engram_seconds_; }
  [[nodiscard]] const std::vector<std::int32_t>& History() const { return history_; }

  /// Optional per-layer dump: called with the layer index (-1 for the head
  /// input) and the hc stream state [n][hc][n_embd] on the host.
  using LayerProbe = void (*)(void* user, int layer, const float* h, std::uint32_t n);
  void SetLayerProbe(LayerProbe probe, void* user) {
    probe_ = probe;
    probe_user_ = user;
  }

private:
  friend class Model;
  Session(Model* model, std::uint32_t max_context) : model_(model), max_context_(max_context) {}
  // One session's tokens in a pass, and where its logits go (as
  // Evaluate's).
  struct PassPart {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::vector<float>* logits;
    bool all_logits;
    bool lazy_logits = false;
  };
  // A part's rows of the pass: [row0, row0 + n) of the scratch.
  struct Rows {
    Session* session;
    std::uint32_t row0;
    std::uint32_t n;
  };
  bool EvaluateChunk(std::span<const std::int32_t> tokens, std::vector<float>* logits, bool all_logits,
                     std::string* error);
  // One pass over every part's tokens, each part's rows after the previous
  // one's. The projections, experts and head run once over all rows; each
  // session writes and attends over its own cache with its rows.
  static bool Pass(std::span<const PassPart> parts, std::string* error);
  // This session's inputs for the pass: tokens, positions, rope, images.
  bool PrepareRows(std::span<const std::int32_t> tokens, std::uint32_t row0,
                   std::string* error);
  static bool Attention(std::span<const Rows> parts, std::uint32_t il,
                        std::uint32_t n, std::string* error);
  // This session's rows of a layer's attention: cache rows, indexer
  // selection, taps and the attention itself.
  bool Attend(std::uint32_t il, std::uint32_t row0, std::uint32_t n,
              bool kv_stored, std::string* error);
  // The indexer's projections, over every part's rows.
  static bool IndexerProject(std::span<const Rows> parts, std::uint32_t il,
                             std::uint32_t n, std::string* error);
  bool Ffn(std::uint32_t il, std::uint32_t n, std::string* error);
  bool Indexer(std::uint32_t il, std::uint32_t row0, std::uint32_t n,
               std::uint32_t pos0, const std::int32_t** sel,
               std::uint32_t* n_sel, std::string* error);

  Model* model_;
  std::uint32_t max_context_;
  std::vector<std::int32_t> history_;
  // Per evaluated position: its temporal, height and width rope positions.
  std::vector<std::array<std::int32_t, 3>> rope_;
  std::int32_t rope_delta_ = 0;
  std::shared_ptr<ImageInputs> images_;
  // The rope positions of the pass being evaluated, from its first position.
  std::vector<std::array<std::int32_t, 3>> chunk_rope_;
  std::int32_t chunk_delta_ = 0;
  // Precomputed kpool groups completed by this session in the current pass.
  std::uint32_t chunk_group_offset_ = 0;
  std::uint32_t chunk_group_count_ = 0;
  bool chunk_group_3d_ = false;
  // The pass's image rows: scratch row, image, its first row, count, and
  // the image's embeddings.
  struct ImageCopy {
    std::uint32_t row, image, offset, count;
    const float* rows;
  };
  std::vector<ImageCopy> image_copies_;
  [[nodiscard]] std::array<std::int32_t, 3> PassRope(std::uint32_t p,
                                                     std::uint32_t pos0) const;
  // Core-owned session state (demand-mapped kContext regions + kFixed rings).
  std::unique_ptr<mem::SessionState> state_;
  std::vector<std::shared_ptr<mem::Region>> kv_regions_;
  std::vector<std::shared_ptr<mem::Region>> pool_regions_;
  bool EnsureRows(std::uint32_t rows, std::string* error);
  // Per layer: latent rows [ring][kv_lora + n_rot] in binary16.
  std::vector<std::uint16_t*> cache_;
  std::vector<std::uint32_t> ring_;
  // Per indexer layer: value | gate rows [kIdxRing][2 idx_dim] and pooled
  // keys [max_context / kpool][idx_dim].
  std::vector<float*> idx_rows_;
  std::vector<float*> idx_pooled_;
  // Indexer scores [kScoreRows][pool rows] and selections
  // [MaxChunk()][(top pools + 1) kpool].
  float* score_ = nullptr;
  std::int32_t* sel_ = nullptr;
  // Draft taps (EnableTaps): rows [kTapRows][n_embd] and, per tapped layer,
  // selections [kTapRows][tap_sel_count_]; tap_row_ is the frontier's row.
  // The scratch of the pass being evaluated (Model::scratch_ or
  // decode_scratch_), and SetPrefillYield's groups and yield.
  Model::Scratch* scratch_ = nullptr;
  std::size_t yield_groups_ = 1;
  std::function<void()> yield_;
  // PromptThrough's rows and their logits for the pass being evaluated.
  std::span<const std::uint32_t> through_rows_;
  std::vector<std::vector<float>>* through_logits_ = nullptr;
  std::vector<std::uint32_t> tap_layers_;
  std::vector<int> tap_index_;  // per layer: its tap, or -1
  float* tap_hidden_ = nullptr;
  std::vector<std::int32_t*> tap_sel_;
  std::vector<bool> tap_has_sel_;
  std::uint32_t tap_sel_count_ = 0;
  std::uint32_t tap_row_ = 0;
  std::uint32_t tap_rows_ = 0;  // rows the last pass left
  bool tap_ready_ = false;
  bool prompt_pass_ = false;  // the current Evaluate is a prompt pass
  // The last Verify: its first position and logits [n][n_vocab].
  std::uint32_t pass_pos0_ = 0;
  mutable std::vector<float> pass_logits_;
  std::vector<void*> allocations_;
  mutable std::vector<float> last_logits_;
  // Lazy device-resident logits in Model::decode_scratch_->logits.
  mutable std::uint64_t logits_gen_ = 0;
  mutable std::uint32_t logits_row0_ = 0;
  mutable std::uint32_t logits_count_ = 0;
  mutable std::uint32_t last_row_ = 0;
  mutable bool logits_on_host_ = true;
  std::optional<std::int32_t> last_argmax_;
  std::vector<std::int32_t> pass_argmax_;
  bool EnsureLogits(std::string* error) const;
  void ClearResidentLogits() noexcept;
  WriteHook write_hook_ = nullptr;
  void* write_user_ = nullptr;
  double engram_seconds_ = 0.0;
  LayerProbe probe_ = nullptr;
  void* probe_user_ = nullptr;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_
