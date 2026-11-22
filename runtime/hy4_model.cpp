// Hy4 Flash GGUF loading: configuration, resident weights and the Engram
// table mapping.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <string>

#include "runtime/hy4_config.hpp"
#include "runtime/hy4_engram.hpp"
#include "runtime/hy4_internal.hpp"
#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

namespace {

constexpr std::string_view kEngramTable = "per_layer_token_embd.weight";

}  // namespace

struct Model::Shards {
  std::unique_ptr<core::GgufReader> trunk;  // every shard, mapped
};

Model::~Model() {
  if (blas_) (void)hipblasDestroy(static_cast<hipblasHandle_t>(blas_));
  scratch_.reset();
  for (void* p : allocations_) (void)hipFree(p);
  if (stream_) (void)hipStreamDestroy(stream_);
}

std::unique_ptr<Model> Model::Load(const std::string& trunk_path,
                                   std::string* error,
                                   std::uint32_t max_context) {
  std::unique_ptr<Model> m(new Model());
  m->max_chunk_ =
      max_context != 0 && max_context <= kWideContext ? kMaxChunk : 4096;
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
    // The table's rows are read from its shard file, not through the mapping.
    const auto* data = static_cast<const std::uint8_t*>(t->data);
    const core::GgufMappedRegion* shard = nullptr;
    for (const auto& region : r.GetMappedRegions()) {
      const auto* first = static_cast<const std::uint8_t*>(region.data);
      if (first && data >= first &&
          data + t->dimensions[1] * c.engram_dim <= first + region.size)
        shard = &region;
    }
    if (!shard || !platform::FileValid(shard->file_descriptor)) {
      *error = "hy4: the Engram table's shard has no file to read";
      return nullptr;
    }
    m->engram_reader_ = EngramReader::Open(
        shard->file_descriptor,
        static_cast<std::uint64_t>(
            data - static_cast<const std::uint8_t*>(shard->data)),
        t->dimensions[1], c.engram_dim, error);
    if (!m->engram_reader_)
      return nullptr;
  }

  if (hipStreamCreateWithFlags(&m->stream_, hipStreamNonBlocking) != hipSuccess) {
    *error = "hipStreamCreate failed";
    return nullptr;
  }

  // Resident upload of every trunk tensor. The Engram table is read from its
  // shard, and a quant's draft layers stay on disk: DFly2 has its own
  // checkpoint, so uploading them would only take memory.
  std::uint64_t total = 0, draft_bytes = 0;
  for (const auto& t : r.GetTensors()) {
    if (t.name == kEngramTable)
      continue;
    if (!BackboneTensor(t.name, c)) {
      draft_bytes += t.size_bytes;
      continue;
    }
    total += t.size_bytes;
  }
  if (draft_bytes != 0) {
    std::fprintf(stderr,
                 "hy4: the %u nextn draft layer%s stay%s on disk (%.2f GiB)\n",
                 c.n_nextn, c.n_nextn == 1 ? "" : "s",
                 c.n_nextn == 1 ? "s" : "", draft_bytes / 1073741824.0);
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

void Model::EngramRows(std::span<const std::int32_t> history,
                       std::span<const std::int32_t> tokens, std::uint32_t pos0,
                       std::uint32_t module, std::uint64_t* rows,
                       float* valid) const {
  const Config& c = config_;
  const std::uint32_t n_gram = c.engram_ngram;
  const std::uint32_t n_head = c.engram_heads_per_ngram;
  const std::uint32_t cols = c.EngramCols();
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  constexpr std::int64_t kDead = -1;
  auto cid = [&](std::int64_t t) -> std::int64_t {
    // An image row hashes as padding and gets no Engram delta (decision 19).
    return t < 0 || t >= static_cast<std::int64_t>(c.engram_token_map.size()) ||
                   t == image_token_
               ? kDead
               : c.engram_token_map[t];
  };
  const std::int64_t pad = c.engram_token_map[c.engram_pad];
  // The token at absolute position p, from the history or this chunk.
  auto token_at = [&](std::int64_t p) -> std::int64_t {
    if (p < pos0)
      return history[static_cast<std::size_t>(p)];
    return tokens[static_cast<std::size_t>(p - pos0)];
  };
  std::vector<std::uint64_t> toks(n_gram);
  const std::uint64_t* mult =
      c.engram_mult.data() + std::size_t{module} * n_gram;
  const std::uint32_t* primes =
      c.engram_primes.data() + std::size_t{module} * cols;
  const std::uint32_t* offs =
      c.engram_offsets.data() + std::size_t{module} * cols;
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
        const std::size_t at = std::size_t{i} * cols + col;
        rows[at] = rolling % primes[col] + offs[col];
        valid[at] = dead ? 0.0f : 1.0f;
      }
    }
  }
}

}  // namespace gufo::models::hy4_flash
