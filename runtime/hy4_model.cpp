// Hy4 Flash GGUF loading: configuration, resident weights and the Engram
// table mapping.

#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/gguf_reader.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <variant>

#include <sys/mman.h>
#include <unistd.h>

#include "runtime/hy4_internal.hpp"
#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

namespace {

constexpr const char* kArch = "hy_v4_flash";
constexpr std::string_view kEngramTable = "per_layer_token_embd.weight";

std::string Key(const char* suffix) { return std::string(kArch) + "." + suffix; }

// Metadata arrays as doubles, whatever their element type.
std::vector<double> MetaArray(const core::GgufReader& r, const std::string& key) {
  std::vector<double> out;
  const auto* v = r.FindMetadata(key);
  if (!v) return out;
  std::visit(
      [&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::vector<std::uint64_t>> ||
                      std::is_same_v<T, std::vector<std::int64_t>> ||
                      std::is_same_v<T, std::vector<double>>) {
          for (const auto e : x) out.push_back(static_cast<double>(e));
        } else if constexpr (std::is_same_v<T, std::uint64_t> || std::is_same_v<T, std::int64_t> ||
                             std::is_same_v<T, double>) {
          out.push_back(static_cast<double>(x));
        } else if constexpr (std::is_same_v<T, bool>) {
          out.push_back(x ? 1.0 : 0.0);
        }
      },
      v->value);
  return out;
}

// Exact integer arrays (the Engram multipliers exceed double precision).
std::vector<std::uint64_t> MetaU64Array(const core::GgufReader& r, const std::string& key) {
  std::vector<std::uint64_t> out;
  const auto* v = r.FindMetadata(key);
  if (!v) return out;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&v->value)) return *u;
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&v->value)) {
    for (const auto e : *s) out.push_back(static_cast<std::uint64_t>(e));
  }
  return out;
}

double MetaScalar(const core::GgufReader& r, const std::string& key, double fallback) {
  const auto a = MetaArray(r, key);
  return a.empty() ? fallback : a[0];
}

bool ReadConfig(const core::GgufReader& r, Config* c, std::string* error) {
  if (r.GetMetadataString("general.architecture").value_or("") != kArch) {
    *error = "not an hy_v4_flash GGUF";
    return false;
  }
  c->n_layer = static_cast<std::uint32_t>(MetaScalar(r, Key("block_count"), 0));
  c->n_embd = static_cast<std::uint32_t>(MetaScalar(r, Key("embedding_length"), 0));
  c->n_vocab = static_cast<std::uint32_t>(MetaScalar(r, Key("vocab_size"), 0));
  c->n_ff_dense = static_cast<std::uint32_t>(MetaScalar(r, Key("feed_forward_length"), 0));
  c->n_dense_lead = static_cast<std::uint32_t>(MetaScalar(r, Key("leading_dense_block_count"), 0));
  c->q_lora = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.q_lora_rank"), 0));
  c->kv_lora = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.kv_lora_rank"), 0));
  c->head_k = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.key_length_mla"), 0));
  c->head_v = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.value_length_mla"), 0));
  c->n_rot = static_cast<std::uint32_t>(MetaScalar(r, Key("rope.dimension_count"), 0));
  c->rope_base = static_cast<float>(MetaScalar(r, Key("rope.freq_base"), 10000.0));
  c->rope_base_swa = static_cast<float>(MetaScalar(r, Key("rope.freq_base_swa"), 10000.0));
  c->swa_window = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.sliding_window"), 0));
  c->rms_eps = static_cast<float>(MetaScalar(r, Key("attention.layer_norm_rms_epsilon"), 1e-5));
  c->n_expert = static_cast<std::uint32_t>(MetaScalar(r, Key("expert_count"), 0));
  c->n_expert_used = static_cast<std::uint32_t>(MetaScalar(r, Key("expert_used_count"), 0));
  c->n_ff_exp = static_cast<std::uint32_t>(MetaScalar(r, Key("expert_feed_forward_length"), 0));
  c->n_latent = static_cast<std::uint32_t>(MetaScalar(r, Key("expert_latent_length"), 0));
  c->expert_scale = static_cast<float>(MetaScalar(r, Key("expert_weights_scale"), 1.0));
  c->expert_norm = MetaScalar(r, Key("expert_weights_norm"), 1.0) != 0.0;
  c->hc = static_cast<std::uint32_t>(MetaScalar(r, Key("hyper_connection.count"), 0));
  c->hc_eps = static_cast<float>(MetaScalar(r, Key("hyper_connection.epsilon"), 1e-6));
  c->hc_magnitude = static_cast<float>(MetaScalar(r, Key("hyper_connection.magnitude"), 1.0));
  c->idx_heads = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.indexer.head_count"), 0));
  c->idx_dim = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.indexer.key_length"), 0));
  c->idx_top_k = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.indexer.top_k"), 0));
  c->kpool = static_cast<std::uint32_t>(MetaScalar(r, Key("attention.indexer.kpool"), 0));

  if (MetaScalar(r, Key("expert_gating_func"), 0) != 2.0) {
    *error = "unsupported expert gating function (expected sigmoid)";
    return false;
  }
  if (!MetaScalar(r, Key("attention.indexer.kpool_select_tail"), 0)) {
    *error = "unsupported indexer: kpool_select_tail must be set";
    return false;
  }

  const auto heads = MetaArray(r, Key("attention.head_count"));
  const auto swa = MetaArray(r, Key("attention.sliding_window_pattern"));
  const auto idx = MetaArray(r, Key("attention.indexer.types"));
  const auto clamp = MetaArray(r, Key("swiglu_clamp_exp"));
  if (heads.size() != c->n_layer || swa.size() != c->n_layer || idx.size() != c->n_layer) {
    *error = "per-layer metadata does not match block_count";
    return false;
  }
  for (std::uint32_t i = 0; i < c->n_layer; ++i) {
    c->n_head.push_back(static_cast<std::uint32_t>(heads[i]));
    c->is_swa.push_back(swa[i] != 0.0);
    c->has_indexer.push_back(idx[i] != 0.0);
    c->swiglu_clamp.push_back(i < clamp.size() ? static_cast<float>(clamp[i]) : 0.0f);
  }

  for (const double l : MetaArray(r, Key("ple.layers"))) {
    c->engram_layers.push_back(static_cast<std::uint32_t>(l));
  }
  c->engram_ngram = static_cast<std::uint32_t>(MetaScalar(r, Key("ple.ngram_size"), 0));
  c->engram_heads_per_ngram = static_cast<std::uint32_t>(MetaScalar(r, Key("ple.heads_per_ngram"), 0));
  c->engram_dim = static_cast<std::uint32_t>(MetaScalar(r, Key("embedding_length_per_layer_input"), 0));
  c->engram_mult = MetaU64Array(r, Key("ple.layer_multipliers"));
  for (const auto v : MetaU64Array(r, Key("ple.head_offsets"))) {
    c->engram_offsets.push_back(static_cast<std::uint32_t>(v));
  }
  for (const auto v : MetaU64Array(r, Key("ple.head_vocab_sizes"))) {
    c->engram_primes.push_back(static_cast<std::uint32_t>(v));
  }
  for (const auto v : MetaU64Array(r, Key("engram.token_map"))) {
    c->engram_token_map.push_back(static_cast<std::int64_t>(v));
  }
  c->engram_pad = static_cast<std::uint32_t>(MetaScalar(r, Key("engram.pad_token_id"), 0));
  const std::size_t n_e = c->engram_layers.size();
  if (n_e > 0 && (c->engram_mult.size() != n_e * c->engram_ngram ||
                  c->engram_primes.size() != n_e * c->EngramCols() ||
                  c->engram_offsets.size() != n_e * c->EngramCols() ||
                  c->engram_pad >= c->engram_token_map.size())) {
    *error = "inconsistent Engram metadata";
    return false;
  }
  if (c->n_layer == 0 || c->n_embd == 0 || c->hc == 0 || c->kv_lora == 0 || c->n_rot == 0) {
    *error = "missing model dimensions";
    return false;
  }
  return true;
}

}  // namespace

int Config::EngramIndex(std::uint32_t layer) const {
  for (std::size_t e = 0; e < engram_layers.size(); ++e) {
    if (engram_layers[e] == layer) return static_cast<int>(e);
  }
  return -1;
}

struct Model::Shards {
  std::unique_ptr<core::GgufReader> trunk;  // every shard, mapped
};

Model::~Model() {
  scratch_.reset();
  for (void* p : allocations_) (void)hipFree(p);
  if (stream_) (void)hipStreamDestroy(stream_);
}

std::unique_ptr<Model> Model::Load(const std::string& trunk_path, std::string* error) {
  std::unique_ptr<Model> m(new Model());
  m->shards_ = std::make_unique<Shards>();
  // Opening the first shard maps the whole split model; the FP8 Engram table
  // is a type core cannot size, so it is accepted unsized and stays mapped.
  core::GgufOpenOptions opts;
  opts.allow_unsized_tensors = true;
  m->shards_->trunk = core::GgufReader::OpenFile(trunk_path, error, opts);
  if (!m->shards_->trunk) return nullptr;
  const core::GgufReader& r = *m->shards_->trunk;
  if (!ReadConfig(r, &m->config_, error)) return nullptr;
  const Config& c = m->config_;
  const auto* unsized = r.FirstUnsizedTensor();
  if (unsized && unsized->name != kEngramTable) {
    *error = r.UnsizedTensorError();
    return nullptr;
  }

  if (!c.engram_layers.empty()) {
    const auto* t = r.FindTensor(kEngramTable);
    if (!t || t->dimensions.size() != 2 || t->dimensions[0] != c.engram_dim || !t->data) {
      *error = "missing per_layer_token_embd.weight (the Engram shard)";
      return nullptr;
    }
    // Every hashed row must fall inside the table.
    std::uint64_t rows = 0;
    for (std::size_t i = 0; i < c.engram_primes.size(); ++i) {
      rows = std::max<std::uint64_t>(rows, std::uint64_t{c.engram_offsets[i]} + c.engram_primes[i]);
    }
    if (rows > t->dimensions[1] || t->size_bytes < t->dimensions[1] * c.engram_dim) {
      *error = "Engram table is smaller than its hash ranges";
      return nullptr;
    }
    m->engram_table_ = static_cast<const std::uint8_t*>(t->data);
    m->engram_bytes_ = t->dimensions[1] * c.engram_dim;
    // Rows are read at random: no readahead around each fault.
    const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
    const auto begin = reinterpret_cast<std::uintptr_t>(m->engram_table_) & ~(page - 1);
    const auto end = reinterpret_cast<std::uintptr_t>(m->engram_table_) + m->engram_bytes_;
    (void)madvise(reinterpret_cast<void*>(begin), end - begin, MADV_RANDOM);
  }

  if (hipStreamCreateWithFlags(&m->stream_, hipStreamNonBlocking) != hipSuccess) {
    *error = "hipStreamCreate failed";
    return nullptr;
  }

  // Resident upload of every trunk tensor.
  std::uint64_t total = 0;
  for (const auto& t : r.GetTensors()) {
    if (t.name != kEngramTable) total += t.size_bytes;
  }
  gufo::hip::WeightAllocator alloc(total);
  auto upload = gufo::hip::WeightUpload::Create(r.GetMappedRegions(), error);
  if (!upload) return nullptr;
  std::fprintf(stderr, "hy4: uploading %.2f GiB of weights (%s)\n", total / 1073741824.0,
               alloc.placement() == gufo::hip::Placement::kMalloc ? "device heap" : "managed");

  auto get = [&](const std::string& name, bool required, Tensor* out) -> bool {
    const auto* t = r.FindTensor(name);
    if (!t) {
      if (required) *error = "missing tensor " + name;
      return !required;
    }
    void* dev = nullptr;
    if (alloc.Allocate(&dev, t->size_bytes) != hipSuccess) {
      *error = "out of memory for " + name;
      return false;
    }
    m->allocations_.push_back(dev);
    if (!upload->CopyMapped(t->data, t->size_bytes, dev, error)) return false;
    m->resident_bytes_ += t->size_bytes;
    out->data = dev;
    out->type = static_cast<int>(t->type);
    out->dims = t->dimensions;
    return true;
  };
  auto blk = [](std::uint32_t i, const char* name) { return "blk." + std::to_string(i) + "." + name; };

  Weights& w = m->weights_;
  bool ok = get("token_embd.weight", true, &w.tok_embd) && get("output.weight", true, &w.output) &&
            get("output_norm.weight", true, &w.output_norm) &&
            get("output_hc_fn.weight", true, &w.hc_head_fn) &&
            get("output_hc_base.weight", true, &w.hc_head_base) &&
            get("output_hc_scale.weight", true, &w.hc_head_scale);
  w.layers.resize(c.n_layer);
  for (std::uint32_t i = 0; ok && i < c.n_layer; ++i) {
    Layer& l = w.layers[i];
    ok = get(blk(i, "attn_norm.weight"), true, &l.attn_norm) &&
         get(blk(i, "attn_sinks.weight"), true, &l.attn_sinks) &&
         get(blk(i, "attn_q_a.weight"), true, &l.wq_a) &&
         get(blk(i, "attn_q_a_norm.weight"), true, &l.q_a_norm) &&
         get(blk(i, "attn_q_b.weight"), true, &l.wq_b) &&
         get(blk(i, "attn_kv_a_mqa.weight"), true, &l.wkv_a) &&
         get(blk(i, "attn_kv_a_norm.weight"), true, &l.kv_a_norm) &&
         get(blk(i, "attn_k_b.weight"), true, &l.wk_b) && get(blk(i, "attn_v_b.weight"), true, &l.wv_b) &&
         get(blk(i, "attn_output.weight"), true, &l.wo) && get(blk(i, "attn_gate.weight"), true, &l.attn_gate) &&
         get(blk(i, "post_attention_norm.weight"), true, &l.attn_post_norm) &&
         get(blk(i, "hc_attn_fn.weight"), true, &l.hc_attn_fn) &&
         get(blk(i, "hc_attn_base.weight"), true, &l.hc_attn_base) &&
         get(blk(i, "hc_attn_scale.weight"), true, &l.hc_attn_scale) &&
         get(blk(i, "hc_ffn_fn.weight"), true, &l.hc_ffn_fn) &&
         get(blk(i, "hc_ffn_base.weight"), true, &l.hc_ffn_base) &&
         get(blk(i, "hc_ffn_scale.weight"), true, &l.hc_ffn_scale) &&
         get(blk(i, "ffn_norm.weight"), true, &l.ffn_norm) &&
         get(blk(i, "post_ffw_norm.weight"), true, &l.ffn_post_norm);
    if (!ok) break;
    if (i < c.n_dense_lead) {
      ok = get(blk(i, "ffn_gate.weight"), true, &l.ffn_gate) && get(blk(i, "ffn_up.weight"), true, &l.ffn_up) &&
           get(blk(i, "ffn_down.weight"), true, &l.ffn_down);
    } else {
      ok = get(blk(i, "ffn_gate_inp.weight"), true, &l.gate_inp) &&
           get(blk(i, "exp_probs_b.bias"), false, &l.exp_bias) &&
           get(blk(i, "ffn_routed_down.weight"), true, &l.routed_down) &&
           get(blk(i, "ffn_routed_up.weight"), true, &l.routed_up) &&
           get(blk(i, "ffn_gate_exps.weight"), true, &l.gate_exps) &&
           get(blk(i, "ffn_up_exps.weight"), true, &l.up_exps) &&
           get(blk(i, "ffn_down_exps.weight"), true, &l.down_exps) &&
           get(blk(i, "ffn_gate_shexp.weight"), true, &l.gate_shexp) &&
           get(blk(i, "ffn_up_shexp.weight"), true, &l.up_shexp) &&
           get(blk(i, "ffn_down_shexp.weight"), true, &l.down_shexp);
    }
    if (ok && c.has_indexer[i]) {
      ok = get(blk(i, "indexer.attn_k.weight"), true, &l.idx_k) &&
           get(blk(i, "indexer.k_norm.weight"), true, &l.idx_k_norm) &&
           get(blk(i, "indexer_compressor_value.weight"), true, &l.idx_value) &&
           get(blk(i, "indexer_compressor_gate.weight"), true, &l.idx_gate) &&
           get(blk(i, "indexer_compressor_ape.weight"), true, &l.idx_ape) &&
           get(blk(i, "indexer_compressor_norm.weight"), true, &l.idx_comp_norm) &&
           get(blk(i, "indexer.attn_q_b.weight"), true, &l.idx_q_b) &&
           get(blk(i, "indexer.proj.weight"), true, &l.idx_proj);
    }
    if (ok && c.EngramIndex(i) >= 0) {
      ok = get(blk(i, "engram_kv.weight"), true, &l.engram_kv) &&
           get(blk(i, "engram_kv_b.weight"), true, &l.engram_kv_b) &&
           get(blk(i, "engram_gate.weight"), true, &l.engram_gate) &&
           get(blk(i, "engram_lookup_norm.weight"), true, &l.engram_lookup_norm) &&
           get(blk(i, "engram_pre_gamma.weight"), true, &l.engram_pre_gamma) &&
           get(blk(i, "engram_hc_scale.weight"), true, &l.engram_hc_scale);
    }
  }
  std::string finish_error;
  const bool finished = upload->Finish(&finish_error);
  if (!ok) return nullptr;
  if (!finished) {
    *error = finish_error;
    return nullptr;
  }
  for (const auto& l : w.layers) {
    if (l.engram_gate && l.engram_gate.type != 0) {
      *error = "quantized engram_gate is not supported";
      return nullptr;
    }
  }
  if (!m->AllocScratch(error)) return nullptr;
  std::fprintf(stderr, "hy4: %.2f GiB resident\n", m->resident_bytes_ / 1073741824.0);
  return m;
}

void Model::EngramGather(std::span<const std::int32_t> history, std::span<const std::int32_t> tokens,
                         std::uint32_t pos0, std::uint8_t* rows, float* valid) const {
  const Config& c = config_;
  const std::uint32_t n_gram = c.engram_ngram;
  const std::uint32_t n_head = c.engram_heads_per_ngram;
  const std::uint32_t cols = c.EngramCols();
  const std::uint32_t d = c.engram_dim;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  constexpr std::int64_t kDead = -1;
  auto cid = [&](std::int64_t t) -> std::int64_t {
    return t < 0 || t >= static_cast<std::int64_t>(c.engram_token_map.size()) ? kDead : c.engram_token_map[t];
  };
  const std::int64_t pad = c.engram_token_map[c.engram_pad];
  // The token at absolute position p, from the history or this chunk.
  auto token_at = [&](std::int64_t p) -> std::int64_t {
    if (p < pos0) return history[static_cast<std::size_t>(p)];
    return tokens[static_cast<std::size_t>(p - pos0)];
  };
  // Row offsets first, so every page read is issued before the first copy
  // waits on one.
  const std::size_t total = c.engram_layers.size() * n * cols;
  std::vector<std::uint64_t> offsets(total);
  std::vector<std::uint64_t> toks(n_gram);
  for (std::size_t e = 0; e < c.engram_layers.size(); ++e) {
    const std::uint64_t* mult = c.engram_mult.data() + e * n_gram;
    const std::uint32_t* primes = c.engram_primes.data() + e * cols;
    const std::uint32_t* offs = c.engram_offsets.data() + e * cols;
    for (std::uint32_t i = 0; i < n; ++i) {
      const std::int64_t p = static_cast<std::int64_t>(pos0) + i;
      std::int64_t id = cid(tokens[i]);
      const bool dead = id == kDead;
      toks[0] = static_cast<std::uint64_t>(dead ? pad : id);
      for (std::uint32_t s = 1; s < n_gram; ++s) {
        id = p - s < 0 ? kDead : cid(token_at(p - s));
        toks[s] = static_cast<std::uint64_t>(id == kDead ? pad : id);
      }
      std::uint64_t rolling = toks[0] * mult[0];
      for (std::uint32_t o = 1; o < n_gram; ++o) {
        rolling ^= toks[o] * mult[o];
        for (std::uint32_t h = 0; h < n_head; ++h) {
          const std::uint32_t col = (o - 1) * n_head + h;
          const std::size_t at = (e * n + i) * cols + col;
          offsets[at] = (rolling % primes[col] + offs[col]) * d;
          valid[at] = dead ? 0.0f : 1.0f;
        }
      }
    }
  }
  const auto page = static_cast<std::uintptr_t>(sysconf(_SC_PAGESIZE));
  for (const std::uint64_t off : offsets) {
    const auto first = reinterpret_cast<std::uintptr_t>(engram_table_ + off) & ~(page - 1);
    const auto last = reinterpret_cast<std::uintptr_t>(engram_table_ + off + d - 1) & ~(page - 1);
    (void)madvise(reinterpret_cast<void*>(first), last - first + page, MADV_WILLNEED);
  }
  for (std::size_t at = 0; at < total; ++at) std::memcpy(rows + at * d, engram_table_ + offsets[at], d);
}

}  // namespace gufo::models::hy4_flash
