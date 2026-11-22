// The Hy4 Flash trunk's GGUF metadata: what the model is, and which of its
// layers this engine runs.

#include "runtime/hy4_config.hpp"

#include <charconv>
#include <string>
#include <string_view>
#include <variant>

namespace gufo::models::hy4_flash {

namespace {

constexpr const char* kArch = "hy_v4_flash";

std::string Key(const char* suffix) {
  return std::string(kArch) + "." + suffix;
}

// Metadata arrays as doubles, whatever their element type.
std::vector<double> MetaArray(const core::GgufReader& r,
                              const std::string& key) {
  std::vector<double> out;
  const auto* v = r.FindMetadata(key);
  if (!v)
    return out;
  std::visit(
      [&](const auto& x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::vector<std::uint64_t>> ||
                      std::is_same_v<T, std::vector<std::int64_t>> ||
                      std::is_same_v<T, std::vector<double>>) {
          for (const auto e : x)
            out.push_back(static_cast<double>(e));
        } else if constexpr (std::is_same_v<T, std::uint64_t> ||
                             std::is_same_v<T, std::int64_t> ||
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
std::vector<std::uint64_t> MetaU64Array(const core::GgufReader& r,
                                        const std::string& key) {
  std::vector<std::uint64_t> out;
  const auto* v = r.FindMetadata(key);
  if (!v)
    return out;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&v->value))
    return *u;
  if (const auto* s = std::get_if<std::vector<std::int64_t>>(&v->value)) {
    for (const auto e : *s)
      out.push_back(static_cast<std::uint64_t>(e));
  }
  return out;
}

double MetaScalar(const core::GgufReader& r, const std::string& key,
                  double fallback) {
  const auto a = MetaArray(r, key);
  return a.empty() ? fallback : a[0];
}

}  // namespace

int Config::EngramIndex(std::uint32_t layer) const {
  for (std::size_t e = 0; e < engram_layers.size(); ++e) {
    if (engram_layers[e] == layer)
      return static_cast<int>(e);
  }
  return -1;
}

bool ReadConfig(const core::GgufReader& r, Config* c, std::string* error) {
  if (r.GetMetadataString("general.architecture").value_or("") != kArch) {
    if (error)
      *error = "not an hy_v4_flash GGUF";
    return false;
  }
  const std::uint32_t blocks =
      static_cast<std::uint32_t>(MetaScalar(r, Key("block_count"), 0));
  c->n_nextn =
      static_cast<std::uint32_t>(MetaScalar(r, Key("nextn_predict_layers"), 0));
  if (c->n_nextn >= blocks) {
    if (error)
      *error = "block_count holds no backbone layers";
    return false;
  }
  c->n_layer = blocks - c->n_nextn;
  c->n_embd =
      static_cast<std::uint32_t>(MetaScalar(r, Key("embedding_length"), 0));
  c->n_vocab = static_cast<std::uint32_t>(MetaScalar(r, Key("vocab_size"), 0));
  c->n_ff_dense =
      static_cast<std::uint32_t>(MetaScalar(r, Key("feed_forward_length"), 0));
  c->n_dense_lead = static_cast<std::uint32_t>(
      MetaScalar(r, Key("leading_dense_block_count"), 0));
  c->q_lora = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.q_lora_rank"), 0));
  c->kv_lora = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.kv_lora_rank"), 0));
  c->head_k = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.key_length_mla"), 0));
  c->head_v = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.value_length_mla"), 0));
  c->n_rot =
      static_cast<std::uint32_t>(MetaScalar(r, Key("rope.dimension_count"), 0));
  c->rope_base =
      static_cast<float>(MetaScalar(r, Key("rope.freq_base"), 10000.0));
  c->rope_base_swa =
      static_cast<float>(MetaScalar(r, Key("rope.freq_base_swa"), 10000.0));
  c->swa_window = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.sliding_window"), 0));
  c->rms_eps = static_cast<float>(
      MetaScalar(r, Key("attention.layer_norm_rms_epsilon"), 1e-5));
  c->n_expert =
      static_cast<std::uint32_t>(MetaScalar(r, Key("expert_count"), 0));
  c->n_expert_used =
      static_cast<std::uint32_t>(MetaScalar(r, Key("expert_used_count"), 0));
  c->n_ff_exp = static_cast<std::uint32_t>(
      MetaScalar(r, Key("expert_feed_forward_length"), 0));
  c->n_latent =
      static_cast<std::uint32_t>(MetaScalar(r, Key("expert_latent_length"), 0));
  c->expert_scale =
      static_cast<float>(MetaScalar(r, Key("expert_weights_scale"), 1.0));
  c->expert_norm = MetaScalar(r, Key("expert_weights_norm"), 1.0) != 0.0;
  c->hc = static_cast<std::uint32_t>(
      MetaScalar(r, Key("hyper_connection.count"), 0));
  c->hc_eps =
      static_cast<float>(MetaScalar(r, Key("hyper_connection.epsilon"), 1e-6));
  c->hc_magnitude =
      static_cast<float>(MetaScalar(r, Key("hyper_connection.magnitude"), 1.0));
  c->idx_heads = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.indexer.head_count"), 0));
  c->idx_dim = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.indexer.key_length"), 0));
  c->idx_top_k = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.indexer.top_k"), 0));
  c->kpool = static_cast<std::uint32_t>(
      MetaScalar(r, Key("attention.indexer.kpool"), 0));

  if (MetaScalar(r, Key("expert_gating_func"), 0) != 2.0) {
    if (error)
      *error = "unsupported expert gating function (expected sigmoid)";
    return false;
  }
  if (!MetaScalar(r, Key("attention.indexer.kpool_select_tail"), 0)) {
    if (error)
      *error = "unsupported indexer: kpool_select_tail must be set";
    return false;
  }

  // A trunk with draft layers records them in block_count. Its per-layer
  // arrays either describe them too, so the backbone is the leading entries,
  // or already hold one entry per backbone layer.
  std::vector<double> heads = MetaArray(r, Key("attention.head_count"));
  std::vector<double> swa =
      MetaArray(r, Key("attention.sliding_window_pattern"));
  std::vector<double> idx = MetaArray(r, Key("attention.indexer.types"));
  const std::vector<double> clamp = MetaArray(r, Key("swiglu_clamp_exp"));
  const auto backbone = [&](const char* key, const std::vector<double>& v) {
    if (v.size() == c->n_layer || v.size() == blocks)
      return true;
    if (error)
      *error = "per-layer metadata " + Key(key) + " has " +
               std::to_string(v.size()) +
               " entries, not one per backbone layer";
    return false;
  };
  if (!backbone("attention.head_count", heads) ||
      !backbone("attention.sliding_window_pattern", swa) ||
      !backbone("attention.indexer.types", idx)) {
    return false;
  }
  heads.resize(c->n_layer);
  swa.resize(c->n_layer);
  idx.resize(c->n_layer);
  for (std::uint32_t i = 0; i < c->n_layer; ++i) {
    c->n_head.push_back(static_cast<std::uint32_t>(heads[i]));
    c->is_swa.push_back(swa[i] != 0.0);
    c->has_indexer.push_back(idx[i] != 0.0);
    c->swiglu_clamp.push_back(i < clamp.size() ? static_cast<float>(clamp[i])
                                               : 0.0f);
  }

  for (const double l : MetaArray(r, Key("ple.layers"))) {
    c->engram_layers.push_back(static_cast<std::uint32_t>(l));
  }
  c->engram_ngram =
      static_cast<std::uint32_t>(MetaScalar(r, Key("ple.ngram_size"), 0));
  c->engram_heads_per_ngram =
      static_cast<std::uint32_t>(MetaScalar(r, Key("ple.heads_per_ngram"), 0));
  c->engram_dim = static_cast<std::uint32_t>(
      MetaScalar(r, Key("embedding_length_per_layer_input"), 0));
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
  c->engram_pad =
      static_cast<std::uint32_t>(MetaScalar(r, Key("engram.pad_token_id"), 0));
  const std::size_t n_e = c->engram_layers.size();
  if (n_e > 0 && (c->engram_mult.size() != n_e * c->engram_ngram ||
                  c->engram_primes.size() != n_e * c->EngramCols() ||
                  c->engram_offsets.size() != n_e * c->EngramCols() ||
                  c->engram_pad >= c->engram_token_map.size())) {
    if (error)
      *error = "inconsistent Engram metadata";
    return false;
  }
  for (const std::uint32_t l : c->engram_layers) {
    if (l >= c->n_layer) {
      if (error)
        *error = "Engram layer outside the backbone";
      return false;
    }
  }
  if (c->n_layer == 0 || c->n_embd == 0 || c->hc == 0 || c->kv_lora == 0 ||
      c->n_rot == 0) {
    if (error)
      *error = "missing model dimensions";
    return false;
  }
  return true;
}

bool BackboneTensor(std::string_view name, const Config& c) {
  constexpr std::string_view prefix = "blk.";
  if (!name.starts_with(prefix))
    return true;  // an embedding, a head, a norm
  const char* const first = name.data() + prefix.size();
  const char* const last = name.data() + name.size();
  std::uint64_t index = 0;
  const auto parsed = std::from_chars(first, last, index);
  if (parsed.ec != std::errc{} || parsed.ptr == first || parsed.ptr >= last ||
      *parsed.ptr != '.')
    return true;  // not a layer tensor
  return index < c.n_layer;
}

}  // namespace gufo::models::hy4_flash
