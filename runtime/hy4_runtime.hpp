#ifndef HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_
#define HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_

// Hy4 Flash model loading and evaluation on HIP. The graph follows
// llama.cpp's hy_v4_flash (ggml-org/llama.cpp#32517); docs/PLAN.md describes
// the design.

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::hy4_flash {

struct Config {
  std::uint32_t n_layer = 0;
  std::uint32_t n_embd = 0;
  std::uint32_t n_vocab = 0;
  std::uint32_t n_ff_dense = 0;
  std::uint32_t n_dense_lead = 0;
  std::vector<std::uint32_t> n_head;  // per layer
  std::vector<bool> is_swa;           // per layer
  std::vector<bool> has_indexer;      // per layer
  std::uint32_t q_lora = 0;
  std::uint32_t kv_lora = 0;
  std::uint32_t head_k = 0;  // nope + rope (256)
  std::uint32_t head_v = 0;  // 256
  std::uint32_t n_rot = 0;   // 64
  float rope_base = 0.0f;
  float rope_base_swa = 0.0f;
  std::uint32_t swa_window = 0;
  float rms_eps = 0.0f;
  // MoE
  std::uint32_t n_expert = 0;
  std::uint32_t n_expert_used = 0;
  std::uint32_t n_ff_exp = 0;
  std::uint32_t n_latent = 0;
  float expert_scale = 1.0f;
  bool expert_norm = true;
  std::vector<float> swiglu_clamp;  // per layer
  // Hyper-connections
  std::uint32_t hc = 0;
  float hc_eps = 0.0f;
  float hc_magnitude = 0.0f;
  // Indexer
  std::uint32_t idx_heads = 0;
  std::uint32_t idx_dim = 0;
  std::uint32_t idx_top_k = 0;
  std::uint32_t kpool = 0;
  // Engram
  std::vector<std::uint32_t> engram_layers;
  std::uint32_t engram_ngram = 0;
  std::uint32_t engram_heads_per_ngram = 0;
  std::uint32_t engram_dim = 0;
  std::vector<std::uint64_t> engram_mult;
  std::vector<std::uint32_t> engram_offsets;
  std::vector<std::uint32_t> engram_primes;
  std::vector<std::int64_t> engram_token_map;
  std::uint32_t engram_pad = 0;

  [[nodiscard]] std::uint32_t EngramCols() const {
    return (engram_ngram - 1) * engram_heads_per_ngram;
  }
  [[nodiscard]] int EngramIndex(std::uint32_t layer) const;
};

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
  // wk_b dequantized to F32 [heads][kv_lora][nope], for batched passes.
  const float* wk_b_f32 = nullptr;
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

/// The loaded model. One evaluation runs at a time (scratch is shared).
class Model {
public:
  /// Most tokens one evaluation pass takes (longer spans run in passes).
  static constexpr std::uint32_t kMaxChunk = 256;
  /// Passes up to this many tokens use matvecs; longer ones use GEMMs.
  static constexpr std::uint32_t kVecMax = 8;
  /// Rows of each indexer layer's value and gate ring.
  static constexpr std::uint32_t kIndexerRing = kMaxChunk + 8;

  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// `trunk_path` is the first GGUF shard; the reader maps the others (the
  /// Engram table) beside it.
  [[nodiscard]] static std::unique_ptr<Model> Load(const std::string& trunk_path,
                                                   std::string* error);

  [[nodiscard]] const Config& config() const { return config_; }
  [[nodiscard]] const Weights& weights() const { return weights_; }
  [[nodiscard]] hipStream_t stream() const { return stream_; }
  [[nodiscard]] std::uint64_t ResidentBytes() const { return resident_bytes_; }

  [[nodiscard]] std::unique_ptr<Session> CreateSession(std::uint32_t max_context,
                                                       std::string* error);
  /// The context a session created for `max_context` holds (a multiple of kpool).
  [[nodiscard]] std::uint32_t SessionContext(std::uint32_t max_context) const;
  /// Latent rows a layer keeps for a session of `max_context` (Session::KvRing).
  [[nodiscard]] std::uint32_t KvRing(std::uint32_t layer, std::uint32_t max_context) const;

  /// Hashes `tokens` (positions pos0..) against `history` (the tokens before
  /// them) and copies their FP8 Engram rows into `rows` [module][token][col][d];
  /// `valid` [module][token][col].
  void EngramGather(std::span<const std::int32_t> history, std::span<const std::int32_t> tokens,
                    std::uint32_t pos0, std::uint8_t* rows, float* valid) const;

private:
  friend class Session;
  friend class Drafter;
  Model() = default;
  bool AllocScratch(std::string* error);

  struct Shards;
  std::unique_ptr<Shards> shards_;
  Config config_;
  Weights weights_;
  std::vector<void*> allocations_;
  std::uint64_t resident_bytes_ = 0;
  hipStream_t stream_ = nullptr;
  void* blas_ = nullptr;  // hipblasHandle_t
  const std::uint8_t* engram_table_ = nullptr;  // host mapping, FP8 rows
  std::uint64_t engram_bytes_ = 0;

  // Scratch for one evaluation of up to kMaxChunk tokens (device unless noted).
  struct Scratch;
  std::unique_ptr<Scratch> scratch_;
};

/// One sequence's state: KV rows, indexer pools and token history.
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Evaluates `tokens` at the next positions; with `logits` set, writes the
  /// logits of every evaluated token ([n][n_vocab], host) when `all_logits`,
  /// else of the last one.
  [[nodiscard]] bool Evaluate(std::span<const std::int32_t> tokens, std::vector<float>* logits,
                              bool all_logits, std::string* error);

  /// Evaluates `tokens` and keeps the last one's logits (LastLogits).
  [[nodiscard]] bool Step(std::span<const std::int32_t> tokens, std::string* error) {
    return Evaluate(tokens, &last_logits_, false, error);
  }
  /// The logits after the last Step, [n_vocab] on the host.
  [[nodiscard]] std::vector<float>& LastLogits() { return last_logits_; }

  /// Evaluates `tokens` (at most kTapRows) and keeps every one's logits
  /// (PassLogits) until Keep.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens, std::string* error);
  /// Logits after token `row` of the last Verify, [n_vocab] on the host.
  [[nodiscard]] std::span<const float> PassLogits(std::uint32_t row) const;
  /// Keeps the first `tokens` (>= 1) tokens of the last Verify and forgets
  /// the rest; LastLogits and the taps then follow the last kept token.
  void Keep(std::uint32_t tokens);

  /// Draft taps: each evaluation leaves its last kTapRows tokens' final
  /// hidden (after the output norm) and the indexer selections of the tapped
  /// layers on the device, for a drafter to read at the frontier.
  static constexpr std::uint32_t kTapRows = 8;
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
  void Reset() {
    history_.clear();
    last_logits_.clear();
    tap_ready_ = false;
  }

  /// Called before an evaluation of positions [first, end) writes any row.
  using WriteHook = void (*)(void* user, std::uint32_t first, std::uint32_t end);
  void SetWriteHook(WriteHook hook, void* user) {
    write_hook_ = hook;
    write_user_ = user;
  }

  /// Begins restoring a state of `tokens` evaluated tokens: returns the token
  /// history to fill; the caller fills the rows, rings and logits.
  [[nodiscard]] std::int32_t* BeginRestore(std::uint32_t tokens) {
    history_.assign(tokens, 0);
    last_logits_.assign(model_->config().n_vocab, 0.0f);
    tap_ready_ = false;
    return history_.data();
  }

  // Device state, for prompt-cache capture and restore. Every layer keeps
  // latent rows [KvRing(layer)][kv_lora + n_rot] (position p in row
  // p % KvRing); indexer layers also keep a value | gate ring
  // [kIndexerRing][2 idx_dim] and pooled keys [max_context / kpool + 1][idx_dim].
  [[nodiscard]] float* KvRows(std::uint32_t layer) const { return cache_[layer]; }
  [[nodiscard]] std::uint32_t KvRing(std::uint32_t layer) const { return ring_[layer]; }
  [[nodiscard]] float* IndexerRing(std::uint32_t layer) const { return idx_rows_[layer]; }
  [[nodiscard]] float* IndexerPools(std::uint32_t layer) const { return idx_pooled_[layer]; }

  [[nodiscard]] std::uint32_t Position() const {
    return static_cast<std::uint32_t>(history_.size());
  }
  [[nodiscard]] std::uint32_t MaxContext() const { return max_context_; }
  /// Host seconds spent gathering Engram rows since creation.
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
  bool EvaluateChunk(std::span<const std::int32_t> tokens, std::vector<float>* logits, bool all_logits,
                     std::string* error);
  bool Attention(std::uint32_t il, std::uint32_t n, std::string* error);
  bool Ffn(std::uint32_t il, std::uint32_t n, std::string* error);
  bool Indexer(std::uint32_t il, std::uint32_t n, std::uint32_t pos0, const std::int32_t** sel,
               std::uint32_t* n_sel, std::string* error);

  Model* model_;
  std::uint32_t max_context_;
  std::vector<std::int32_t> history_;
  // Per layer: latent rows [ring][kv_lora + n_rot].
  std::vector<float*> cache_;
  std::vector<std::uint32_t> ring_;
  // Per indexer layer: value | gate rows [kIdxRing][2 idx_dim] and pooled
  // keys [max_context / kpool][idx_dim].
  std::vector<float*> idx_rows_;
  std::vector<float*> idx_pooled_;
  // Indexer scores [kMaxChunk][pool rows] and selections
  // [kMaxChunk][(top pools + 1) kpool].
  float* score_ = nullptr;
  std::int32_t* sel_ = nullptr;
  // Draft taps (EnableTaps): rows [kTapRows][n_embd] and, per tapped layer,
  // selections [kTapRows][tap_sel_count_]; tap_row_ is the frontier's row.
  std::vector<std::uint32_t> tap_layers_;
  std::vector<int> tap_index_;  // per layer: its tap, or -1
  float* tap_hidden_ = nullptr;
  std::vector<std::int32_t*> tap_sel_;
  std::vector<bool> tap_has_sel_;
  std::uint32_t tap_sel_count_ = 0;
  std::uint32_t tap_row_ = 0;
  std::uint32_t tap_rows_ = 0;  // rows the last pass left
  bool tap_ready_ = false;
  // The last Verify: its first position and logits [n][n_vocab].
  std::uint32_t pass_pos0_ = 0;
  std::vector<float> pass_logits_;
  // Attention selection mask [kMaxChunk][max_context].
  float* mask_ = nullptr;
  std::vector<void*> allocations_;
  std::vector<float> last_logits_;
  WriteHook write_hook_ = nullptr;
  void* write_user_ = nullptr;
  double engram_seconds_ = 0.0;
  LayerProbe probe_ = nullptr;
  void* probe_user_ = nullptr;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_RUNTIME_HPP_
