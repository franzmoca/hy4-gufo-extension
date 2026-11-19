// Hy4 Flash evaluation: one chunk of up to Model::kMaxChunk tokens through
// the 42-layer iHC stack, following llama.cpp's hy_v4_flash graph.

#include <gufo/ext/experimental/managed_alloc.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <hipblas/hipblas.h>

#include "kernels/rocm/hy4_kernels.h"
#include "kernels/rocm/mmq/hy4_mmq.h"
#include "runtime/hy4_internal.hpp"
#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

namespace {

// The reference Engram norms leave eps unset: the FP32 machine epsilon.
constexpr float kEngramEps = 1.1920929e-07f;
// Attention scores per batched pass ([query rows][keys] floats).
constexpr std::size_t kScoreFloats = std::size_t{64} << 20;

bool Fail(std::string* error, const std::string& what) {
  if (error) *error = what;
  return false;
}

#define HY4_CHECK(expr)                                                     \
  do {                                                                      \
    if ((expr) != 0) return Fail(error, std::string("hy4: failed: ") + #expr); \
  } while (0)

template <typename T>
bool DeviceAlloc(T** p, std::size_t count, std::vector<void*>* owned) {
  void* v = nullptr;
  if (gufo::hip::AllocateDevice(&v, std::max<std::size_t>(count, 1) * sizeof(T)) != hipSuccess) return false;
  // hipMemset runs on the null stream, which the engine's non-blocking stream
  // does not wait for: finish it before any kernel can touch the buffer.
  if (hipMemset(v, 0, std::max<std::size_t>(count, 1) * sizeof(T)) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess) {
    (void)hipFree(v);
    return false;
  }
  owned->push_back(v);
  *p = static_cast<T*>(v);
  return true;
}

template <typename T>
bool HostAlloc(T** p, std::size_t count, std::vector<void*>* owned) {
  void* v = nullptr;
  if (hipHostMalloc(&v, std::max<std::size_t>(count, 1) * sizeof(T), hipHostMallocDefault) != hipSuccess) {
    return false;
  }
  owned->push_back(v);
  *p = static_cast<T*>(v);
  return true;
}

}  // namespace

bool MatVec(const Tensor& w, const float* x, float* y, std::uint32_t m, std::uint32_t k, std::uint32_t n,
            const Exec& ex, std::string* error) {
  if (w.dims.size() < 2 || w.dims[0] != k || w.dims[1] != m) {
    return Fail(error, "hy4: matvec shape mismatch (" + std::to_string(m) + "x" + std::to_string(k) + ")");
  }
  hipStream_t stream = ex.stream;
  const bool batched = n > Model::kVecMax;
  switch (w.type) {
    case HY4_TYPE_F32:
      if (batched) {
        // Row-major y[n][m] = x[n][k] W^T, as column-major y^T = W^T x^T.
        const float one = 1.0f;
        const float zero = 0.0f;
        if (hipblasSgemm(static_cast<hipblasHandle_t>(ex.blas_handle), HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(m), static_cast<int>(n),
                         static_cast<int>(k), &one, w.f32(), static_cast<int>(k), x, static_cast<int>(k), &zero, y,
                         static_cast<int>(m)) != HIPBLAS_STATUS_SUCCESS) {
          return Fail(error, "hy4: sgemm failed");
        }
        return true;
      }
      [[fallthrough]];
    case HY4_TYPE_F16:
      HY4_CHECK(hy4_gemv_float(w.type, w.data, x, y, m, k, n, stream));
      return true;
    case HY4_TYPE_Q8_0:
    case HY4_TYPE_Q4_K:
    case HY4_TYPE_Q6_K:
      if (k % 256 == 0) {
        if (batched) {
          HY4_CHECK(hy4_mmq_dense_typed(w.type, w.data, x, y, static_cast<int>(m), static_cast<int>(n),
                                        static_cast<int>(k), stream));
          return true;
        }
        HY4_CHECK(hy4_mmq_dense_vec_typed(w.type, w.data, x, y, static_cast<int>(m), static_cast<int>(n),
                                          static_cast<int>(k), stream));
        return true;
      }
      if (w.type == HY4_TYPE_Q8_0) {
        HY4_CHECK(hy4_gemv_q8_0_grouped(w.data, x, y, m, k, 1, n, k, 0, m, 0, stream));
        return true;
      }
      break;
    default:
      break;
  }
  return Fail(error, "hy4: unsupported matvec type " + std::to_string(w.type));
}

bool Model::AllocScratch(std::string* error) {
  if (hy4_mmq_init(0) != 0) return Fail(error, "hy4: mmq init failed");
  const Config& c = config_;
  const std::uint32_t n = kMaxChunk;
  const std::uint32_t hmax = *std::max_element(c.n_head.begin(), c.n_head.end());
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t modules = static_cast<std::uint32_t>(c.engram_layers.size());
  const std::uint32_t cols = c.engram_layers.empty() ? 0 : c.EngramCols();
  const std::size_t ffn = std::max<std::size_t>(std::size_t{c.n_ff_dense} * n,
                                                std::size_t{c.n_expert_used} * n *
                                                    std::max(c.n_ff_exp, c.n_latent));
  scratch_ = std::make_unique<Scratch>();
  Scratch& s = *scratch_;
  auto* owned = &allocations_;
  const bool ok =
      DeviceAlloc(&s.tokens, n, owned) && DeviceAlloc(&s.pos, n, owned) &&
      DeviceAlloc(&s.groups, n, owned) && DeviceAlloc(&s.group_pos, n, owned) &&
      DeviceAlloc(&s.group_hw, 2 * n, owned) &&
      DeviceAlloc(&s.n_pool, n, owned) && DeviceAlloc(&s.rope, n, owned) &&
      DeviceAlloc(&s.rope_hw, 2 * n, owned) &&
      HostAlloc(&s.engram_host, std::size_t{modules} * n * cols * c.engram_dim,
                owned) &&
      HostAlloc(&s.engram_valid_host, std::size_t{modules} * n * cols, owned) &&
      HostAlloc(&s.engram_any_host, std::size_t{modules} * n, owned) &&
      DeviceAlloc(&s.engram_rows,
                  std::size_t{modules} * n * cols * c.engram_dim, owned) &&
      DeviceAlloc(&s.engram_valid, std::size_t{modules} * n * cols, owned) &&
      DeviceAlloc(&s.engram_any, std::size_t{modules} * n, owned) &&
      DeviceAlloc(&s.engram_normed, std::size_t{n} * cols * c.engram_dim,
                  owned) &&
      DeviceAlloc(&s.engram_kv, std::size_t{n} * (c.hc + 1) * c.n_embd,
                  owned) &&
      DeviceAlloc(&s.h, std::size_t{n} * c.hc * c.n_embd, owned) &&
      DeviceAlloc(&s.post, std::size_t{n} * c.hc, owned) &&
      DeviceAlloc(&s.x, std::size_t{n} * c.n_embd, owned) &&
      DeviceAlloc(&s.cur, std::size_t{n} * c.n_embd, owned) &&
      DeviceAlloc(&s.y, std::size_t{n} * c.n_embd, owned) &&
      DeviceAlloc(&s.qr, std::size_t{n} * c.q_lora, owned) &&
      DeviceAlloc(&s.q, std::size_t{n} * hmax * c.head_k, owned) &&
      DeviceAlloc(&s.kva, std::size_t{n} * dk, owned) &&
      DeviceAlloc(&s.qfull, std::size_t{n} * hmax * dk, owned) &&
      DeviceAlloc(&s.att, std::size_t{n} * hmax * c.kv_lora, owned) &&
      DeviceAlloc(&s.attv, std::size_t{n} * hmax * c.head_v, owned) &&
      DeviceAlloc(&s.gate, std::size_t{n} * hmax * c.head_v, owned) &&
      DeviceAlloc(&s.ik, std::size_t{n} * c.idx_dim, owned) &&
      DeviceAlloc(&s.ivg, std::size_t{n} * 2 * c.idx_dim, owned) &&
      DeviceAlloc(&s.iq, std::size_t{n} * c.idx_heads * c.idx_dim, owned) &&
      DeviceAlloc(&s.iw, std::size_t{n} * c.idx_heads, owned) &&
      DeviceAlloc(&s.ffn_a, ffn, owned) && DeviceAlloc(&s.ffn_b, ffn, owned) &&
      DeviceAlloc(&s.ffn_c, ffn, owned) &&
      DeviceAlloc(&s.latent, std::size_t{n} * c.n_latent, owned) &&
      DeviceAlloc(&s.logits_r, std::size_t{n} * c.n_expert, owned) &&
      DeviceAlloc(&s.ids, std::size_t{n} * c.n_expert_used, owned) &&
      HostAlloc(&s.ids_host, std::size_t{n} * c.n_expert_used, owned) &&
      DeviceAlloc(&s.route_w, std::size_t{n} * c.n_expert_used, owned) &&
      DeviceAlloc(&s.moe, std::size_t{n} * c.n_latent, owned) &&
      DeviceAlloc(&s.shared, std::size_t{n} * c.n_embd, owned) &&
      DeviceAlloc(&s.logits, std::size_t{n} * c.n_vocab, owned) &&
      HostAlloc(&s.host_h, std::size_t{n} * c.hc * c.n_embd, owned);
  const bool ok2 = ok && DeviceAlloc(&s.keys, std::size_t{c.swa_window + n} * dk, owned) &&
                   DeviceAlloc(&s.scores, kScoreFloats, owned);
  if (!ok2) return Fail(error, "hy4: scratch allocation failed");
  hipblasHandle_t blas = nullptr;
  if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS || hipblasSetStream(blas, stream_) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error, "hy4: hipblas init failed");
  }
  blas_ = blas;
  // wk_b (K = 192, outside the MMQ tiles) as F32 for strided-batched GEMMs.
  for (Layer& l : weights_.layers) {
    const std::uint64_t count = l.wk_b.dims[0] * l.wk_b.dims[1] * l.wk_b.dims[2];
    float* f = nullptr;
    if (!DeviceAlloc(&f, count, owned) || hy4_dequant_q8_0(l.wk_b.data, count / 32, f, stream_) != 0) {
      return Fail(error, "hy4: wk_b dequantization failed");
    }
    l.wk_b_f32 = f;
  }
  if (hipStreamSynchronize(stream_) != hipSuccess) return Fail(error, "hy4: wk_b dequantization failed");
  return true;
}

std::uint32_t Model::SessionContext(std::uint32_t max_context) const {
  const Config& c = config_;
  max_context = std::max<std::uint32_t>(max_context, c.kpool);
  return (max_context + c.kpool - 1) / c.kpool * c.kpool;
}

std::uint32_t Model::KvRing(std::uint32_t layer, std::uint32_t max_context) const {
  max_context = SessionContext(max_context);
  return config_.is_swa[layer] ? std::min(max_context, config_.swa_window + kMaxChunk) : max_context;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context, std::string* error) {
  const Config& c = config_;
  max_context = SessionContext(max_context);
  std::unique_ptr<Session> s(new Session(this, max_context));
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t pools = max_context / c.kpool + 1;
  s->cache_.assign(c.n_layer, nullptr);
  s->ring_.assign(c.n_layer, 0);
  s->idx_rows_.assign(c.n_layer, nullptr);
  s->idx_pooled_.assign(c.n_layer, nullptr);
  bool ok = true;
  for (std::uint32_t i = 0; ok && i < c.n_layer; ++i) {
    s->ring_[i] = KvRing(i, max_context);
    ok = DeviceAlloc(&s->cache_[i], std::size_t{s->ring_[i]} * dk, &s->allocations_);
    if (ok && c.has_indexer[i]) {
      ok = DeviceAlloc(&s->idx_rows_[i], std::size_t{Model::kIndexerRing} * 2 * c.idx_dim, &s->allocations_) &&
           DeviceAlloc(&s->idx_pooled_[i], std::size_t{pools} * c.idx_dim, &s->allocations_);
    }
  }
  const std::uint32_t top_pools = c.idx_top_k / c.kpool - 1;
  ok = ok && DeviceAlloc(&s->mask_, std::size_t{kMaxChunk} * max_context, &s->allocations_) &&
       DeviceAlloc(&s->score_, std::size_t{kMaxChunk} * pools, &s->allocations_) &&
       DeviceAlloc(&s->sel_, std::size_t{kMaxChunk} * (top_pools + 1) * c.kpool, &s->allocations_);
  if (!ok) {
    Fail(error, "hy4: session allocation failed");
    return nullptr;
  }
  return s;
}

Session::~Session() {
  for (void* p : allocations_) (void)hipFree(p);
}

bool Session::Evaluate(std::span<const std::int32_t> tokens, std::vector<float>* logits, bool all_logits,
                       std::string* error) {
  if (logits) logits->clear();
  std::vector<float> part;
  for (std::size_t i = 0; i < tokens.size(); i += Model::kMaxChunk) {
    const std::size_t n = std::min<std::size_t>(Model::kMaxChunk, tokens.size() - i);
    const bool last = i + n == tokens.size();
    std::vector<float>* want = logits && (all_logits || last) ? &part : nullptr;
    if (!EvaluateChunk(tokens.subspan(i, n), want, all_logits, error)) return false;
    if (want) logits->insert(logits->end(), part.begin(), part.end());
  }
  return true;
}

bool Session::Verify(std::span<const std::int32_t> tokens, std::string* error) {
  if (tokens.empty() || tokens.size() > kTapRows) return Fail(error, "hy4: verify takes 1 to 8 tokens");
  pass_pos0_ = Position();
  tap_ready_ = false;
  if (!Evaluate(tokens, &pass_logits_, true, error)) return false;
  last_logits_.assign(pass_logits_.end() - model_->config().n_vocab, pass_logits_.end());
  return true;
}

std::span<const float> Session::PassLogits(std::uint32_t row) const {
  const std::size_t v = model_->config().n_vocab;
  return std::span<const float>(pass_logits_).subspan(row * v, v);
}

void Session::Keep(std::uint32_t tokens) {
  const std::uint32_t evaluated = Position() - pass_pos0_;
  tokens = std::clamp<std::uint32_t>(tokens, 1, evaluated);
  history_.resize(pass_pos0_ + tokens);
  rope_.resize(history_.size());
  const auto row = PassLogits(tokens - 1);
  last_logits_.assign(row.begin(), row.end());
  // The pass left its last tap_rows_ (= evaluated) rows.
  if (tap_ready_) tap_row_ = tap_rows_ - (evaluated - tokens) - 1;
}

std::array<std::int32_t, 3> Session::PassRope(std::uint32_t p,
                                              std::uint32_t pos0) const {
  return p < pos0 ? rope_.at(p) : chunk_rope_.at(p - pos0);
}

void Session::RestoreRope(std::int32_t delta, std::uint32_t group_first,
                          std::array<std::int32_t, 3> group_rope) {
  rope_delta_ = delta;
  rope_.resize(history_.size());
  for (std::size_t p = 0; p < rope_.size(); ++p) {
    const std::int32_t t = static_cast<std::int32_t>(p) + delta;
    rope_[p] = {t, t, t};
  }
  if (group_first < rope_.size())
    rope_[group_first] = group_rope;
}

bool Session::EnableTaps(std::span<const std::uint32_t> layers, std::string* error) {
  const Config& c = model_->config();
  if (tap_hidden_) return Fail(error, "hy4: taps already enabled");
  tap_index_.assign(c.n_layer, -1);
  tap_sel_count_ = c.idx_top_k / c.kpool * c.kpool;
  for (const std::uint32_t layer : layers) {
    if (layer >= c.n_layer || !c.has_indexer[layer] || c.is_swa[layer]) {
      return Fail(error, "hy4: a tapped layer must be a sparse indexer layer");
    }
    tap_index_[layer] = static_cast<int>(tap_layers_.size());
    tap_layers_.push_back(layer);
    tap_sel_.push_back(nullptr);
    if (!DeviceAlloc(&tap_sel_.back(), std::size_t{kTapRows} * tap_sel_count_, &allocations_)) {
      return Fail(error, "hy4: tap allocation failed");
    }
  }
  tap_has_sel_.assign(layers.size(), false);
  if (!DeviceAlloc(&tap_hidden_, std::size_t{kTapRows} * c.n_embd, &allocations_)) {
    return Fail(error, "hy4: tap allocation failed");
  }
  return true;
}

const float* Session::TapHidden() const {
  return tap_hidden_ + std::size_t{tap_row_} * model_->config().n_embd;
}

const std::int32_t* Session::TapSelection(std::uint32_t k, std::uint32_t* count) const {
  *count = tap_sel_count_;
  return tap_has_sel_[k] ? tap_sel_[k] + std::size_t{tap_row_} * tap_sel_count_ : nullptr;
}

bool Session::EvaluateChunk(std::span<const std::int32_t> tokens, std::vector<float>* logits, bool all_logits,
                            std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Weights& w = m.weights_;
  Model::Scratch& s = *m.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_};
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t pos0 = Position();
  if (n == 0) return true;
  if (pos0 + n > max_context_) return Fail(error, "hy4: context full");
  for (const std::int32_t t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.n_vocab) return Fail(error, "hy4: token out of range");
  }

  // Rope positions (decision 19): an image row (r, c) rotates at (s, s + r,
  // s + c), with s the position its image starts at; text continues after
  // the image's square of positions. Committed with the history.
  struct ImageCopy {
    std::uint32_t row, image, offset, count;
  };
  std::vector<ImageCopy> copies;
  std::vector<std::array<std::int32_t, 3>> rope(n);
  std::int32_t delta = rope_delta_;
  s.rope_3d = false;
  const std::span<const ImageSpan> spans =
      images_ ? images_->Spans() : std::span<const ImageSpan>{};
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t p = pos0 + i;
    const auto it = std::ranges::find_if(spans, [&](const ImageSpan& span) {
      return p >= span.first && p < span.End();
    });
    if (it == spans.end()) {
      const std::int32_t t = static_cast<std::int32_t>(p) + delta;
      rope[i] = {t, t, t};
      continue;
    }
    const std::uint32_t k = static_cast<std::uint32_t>(it - spans.begin());
    const std::uint32_t offset = p - it->first;
    const std::int32_t start = static_cast<std::int32_t>(it->first) + delta;
    rope[i] = {start,
               start + static_cast<std::int32_t>(offset / (it->width + 1)),
               start + static_cast<std::int32_t>(offset % (it->width + 1))};
    s.rope_3d = true;
    if (!copies.empty() && copies.back().image == k &&
        copies.back().row + copies.back().count == i) {
      ++copies.back().count;
    } else {
      copies.push_back({i, k, offset, 1});
    }
    if (p + 1 == it->End()) {
      delta = start +
              static_cast<std::int32_t>(std::max(it->height, it->width + 1)) -
              static_cast<std::int32_t>(it->End());
    }
  }
  // Encode before this pass writes any row.
  std::vector<const float*> image_rows(copies.size());
  for (std::size_t k = 0; k < copies.size(); ++k) {
    image_rows[k] = images_->Embeddings(copies[k].image, error);
    if (image_rows[k] == nullptr)
      return false;
  }

  if (write_hook_) write_hook_(write_user_, pos0, pos0 + n);
  std::vector<std::int32_t> pos(n), rope_t(n), rope_hw(2 * n);
  for (std::uint32_t i = 0; i < n; ++i) {
    pos[i] = static_cast<std::int32_t>(pos0 + i);
    rope_t[i] = rope[i][0];
    rope_hw[2 * i] = rope[i][1];
    rope_hw[2 * i + 1] = rope[i][2];
  }
  HY4_CHECK(hipMemcpyAsync(s.tokens, tokens.data(), n * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
  HY4_CHECK(hipMemcpyAsync(s.pos, pos.data(), n * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
  HY4_CHECK(hipMemcpyAsync(s.rope, rope_t.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, st));
  if (s.rope_3d) {
    HY4_CHECK(hipMemcpyAsync(s.rope_hw, rope_hw.data(),
                             2 * n * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, st));
  }
  chunk_rope_ = rope;

  // Engram rows for every module, gathered on the host from the mapping.
  const std::uint32_t modules = static_cast<std::uint32_t>(c.engram_layers.size());
  const std::uint32_t cols = modules ? c.EngramCols() : 0;
  if (modules) {
    const auto g0 = std::chrono::steady_clock::now();
    m.EngramGather(history_, tokens, pos0, s.engram_host, s.engram_valid_host);
    engram_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - g0).count();
    for (std::uint32_t e = 0; e < modules; ++e) {
      for (std::uint32_t i = 0; i < n; ++i) {
        float any = 0.0f;
        for (std::uint32_t col = 0; col < cols; ++col) any += s.engram_valid_host[(e * n + i) * cols + col];
        s.engram_any_host[e * n + i] = any > 0.0f ? 1.0f : 0.0f;
      }
    }
    HY4_CHECK(hipMemcpyAsync(s.engram_rows, s.engram_host, std::size_t{modules} * n * cols * c.engram_dim,
                             hipMemcpyHostToDevice, st));
    HY4_CHECK(hipMemcpyAsync(s.engram_valid, s.engram_valid_host, std::size_t{modules} * n * cols * sizeof(float),
                             hipMemcpyHostToDevice, st));
    HY4_CHECK(hipMemcpyAsync(s.engram_any, s.engram_any_host, std::size_t{modules} * n * sizeof(float),
                             hipMemcpyHostToDevice, st));
  }

  HY4_CHECK(hy4_embed_q8_0(w.tok_embd.data, s.tokens, n, c.n_embd, s.x, st));
  for (std::size_t k = 0; k < copies.size(); ++k) {
    const ImageCopy& copy = copies[k];
    HY4_CHECK(
        hipMemcpyAsync(s.x + std::size_t{copy.row} * c.n_embd,
                       image_rows[k] + std::size_t{copy.offset} * c.n_embd,
                       std::size_t{copy.count} * c.n_embd * sizeof(float),
                       hipMemcpyDeviceToDevice, st));
  }
  HY4_CHECK(hy4_hc_init(s.x, s.h, n, c.hc, c.n_embd, st));

  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const Layer& l = w.layers[il];
    const int e = c.EngramIndex(il);
    if (e >= 0) {
      const std::uint32_t width = cols * c.engram_dim;
      HY4_CHECK(hy4_engram_rows(s.engram_rows + std::size_t(e) * n * width, s.engram_valid + std::size_t(e) * n * cols,
                                l.engram_lookup_norm.f32(), s.engram_normed, n, cols, c.engram_dim, kEngramEps, st));
      if (!MatVec(l.engram_kv, s.engram_normed, s.engram_kv, (c.hc + 1) * c.n_embd, width, n, ex, error)) {
        return false;
      }
      HY4_CHECK(hy4_add_bias_rows(s.engram_kv, l.engram_kv_b.f32(), n, c.hc + 1, c.n_embd, st));
      HY4_CHECK(hy4_engram_apply(s.h, s.engram_kv, l.engram_gate.f32(), l.engram_hc_scale.f32(),
                                 l.engram_pre_gamma.f32(), s.engram_any + std::size_t(e) * n, n, c.hc, c.n_embd,
                                 kEngramEps, st));
    }

    HY4_CHECK(hy4_hc_pre(s.h, l.hc_attn_fn.f32(), l.hc_attn_scale.f32(), l.hc_attn_base.f32(), s.x, s.post, n, c.hc,
                         c.n_embd, c.rms_eps, c.hc_eps, c.hc_magnitude, st));
    HY4_CHECK(hy4_rms_norm(s.x, l.attn_norm.f32(), s.cur, n, c.n_embd, c.n_embd, c.rms_eps, st));
    if (!Attention(il, n, error)) return false;
    HY4_CHECK(hy4_hc_post(s.y, s.h, s.post, s.h, n, c.hc, c.n_embd, st));

    HY4_CHECK(hy4_hc_pre(s.h, l.hc_ffn_fn.f32(), l.hc_ffn_scale.f32(), l.hc_ffn_base.f32(), s.x, s.post, n, c.hc,
                         c.n_embd, c.rms_eps, c.hc_eps, c.hc_magnitude, st));
    HY4_CHECK(hy4_rms_norm(s.x, l.ffn_norm.f32(), s.cur, n, c.n_embd, c.n_embd, c.rms_eps, st));
    if (!Ffn(il, n, error)) return false;
    HY4_CHECK(hy4_rms_norm(s.y, l.ffn_post_norm.f32(), s.y, n, c.n_embd, c.n_embd, c.rms_eps, st));
    HY4_CHECK(hy4_hc_post(s.y, s.h, s.post, s.h, n, c.hc, c.n_embd, st));

    if (probe_) {
      HY4_CHECK(hipMemcpyAsync(s.host_h, s.h, std::size_t{n} * c.hc * c.n_embd * sizeof(float),
                               hipMemcpyDeviceToHost, st));
      HY4_CHECK(hipStreamSynchronize(st));
      probe_(probe_user_, static_cast<int>(il), s.host_h, n);
    }
  }

  // Head: reduce the streams, norm, project.
  HY4_CHECK(hy4_hc_pre(s.h, w.hc_head_fn.f32(), w.hc_head_scale.f32(), w.hc_head_base.f32(), s.x, nullptr, n, c.hc,
                       c.n_embd, c.rms_eps, c.hc_eps, 1.0f, st));
  HY4_CHECK(hy4_rms_norm(s.x, w.output_norm.f32(), s.cur, n, c.n_embd, c.n_embd, c.rms_eps, st));
  if (tap_hidden_) {
    const std::uint32_t r = std::min(n, kTapRows);
    HY4_CHECK(hipMemcpyAsync(tap_hidden_, s.cur + std::size_t{n - r} * c.n_embd, std::size_t{r} * c.n_embd * sizeof(float),
                             hipMemcpyDeviceToDevice, st));
    tap_rows_ = r;
    tap_row_ = r - 1;
    tap_ready_ = true;
  }
  if (logits) {
    const std::uint32_t first = all_logits ? 0 : n - 1;
    const std::uint32_t rows = n - first;
    if (!MatVec(w.output, s.cur + std::size_t{first} * c.n_embd, s.logits, c.n_vocab, c.n_embd, rows, ex, error)) {
      return false;
    }
    logits->resize(std::size_t{rows} * c.n_vocab);
    HY4_CHECK(hipMemcpyAsync(logits->data(), s.logits, logits->size() * sizeof(float), hipMemcpyDeviceToHost, st));
  }
  HY4_CHECK(hipStreamSynchronize(st));
  history_.insert(history_.end(), tokens.begin(), tokens.end());
  rope_.insert(rope_.end(), rope.begin(), rope.end());
  rope_delta_ = delta;
  return true;
}

bool Session::Attention(std::uint32_t il, std::uint32_t n, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *m.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_};
  auto* blas = static_cast<hipblasHandle_t>(m.blas_);
  const std::uint32_t heads = c.n_head[il];
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t nope = c.head_k - c.n_rot;
  const float base = c.is_swa[il] ? c.rope_base_swa : c.rope_base;

  if (!MatVec(l.wq_a, s.cur, s.qr, c.q_lora, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_rms_norm(s.qr, l.q_a_norm.f32(), s.qr, n, c.q_lora, c.q_lora, c.rms_eps, st));
  if (!MatVec(l.wq_b, s.qr, s.q, heads * c.head_k, c.q_lora, n, ex, error)) return false;
  if (!MatVec(l.wkv_a, s.cur, s.kva, dk, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_rms_norm(s.kva, l.kv_a_norm.f32(), s.kva, n, c.kv_lora, dk, c.rms_eps, st));
  const std::int32_t* hw = s.rope_3d ? s.rope_hw : nullptr;
  HY4_CHECK(hy4_rope_3d(s.kva, s.rope, hw, n, 1, dk, 0, c.kv_lora, c.n_rot,
                        base, st));
  HY4_CHECK(hy4_rope_3d(s.q, s.rope, hw, n, heads,
                        std::uint64_t{heads} * c.head_k, c.head_k, nope,
                        c.n_rot, base, st));

  // Absorb wk_b into the query: q_abs[h] = wk_b[h] (512 x 192) q_nope[h].
  if (l.wk_b.type != HY4_TYPE_Q8_0 || l.wv_b.type != HY4_TYPE_Q8_0) return Fail(error, "hy4: wk_b/wv_b must be Q8_0");
  if (n > Model::kVecMax) {
    // Per head: qfull[t][h][0:kv_lora] = wk_b[h] q[t][h][0:nope], column-major.
    const float one = 1.0f;
    const float zero = 0.0f;
    if (hipblasSgemmStridedBatched(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(c.kv_lora), static_cast<int>(n),
                                   static_cast<int>(nope), &one, l.wk_b_f32, static_cast<int>(nope),
                                   static_cast<hipblasStride>(c.kv_lora) * nope, s.q,
                                   static_cast<int>(heads * c.head_k), c.head_k, &zero, s.qfull,
                                   static_cast<int>(heads * dk), dk, static_cast<int>(heads)) != HIPBLAS_STATUS_SUCCESS) {
      return Fail(error, "hy4: wk_b gemm failed");
    }
  } else {
    HY4_CHECK(hy4_gemv_q8_0_grouped(l.wk_b.data, s.q, s.qfull, c.kv_lora, nope, heads, n,
                                    std::uint64_t{heads} * c.head_k, c.head_k, std::uint64_t{heads} * dk, dk, st));
  }
  HY4_CHECK(hipMemcpy2DAsync(s.qfull + c.kv_lora, dk * sizeof(float), s.q + nope, c.head_k * sizeof(float),
                             c.n_rot * sizeof(float), std::size_t{n} * heads, hipMemcpyDeviceToDevice, st));
  HY4_CHECK(hy4_store_rows(s.kva, dk, cache_[il], ring_[il], s.pos, n, dk, st));

  const std::int32_t* sel = nullptr;
  std::uint32_t n_sel = 0;
  if (c.has_indexer[il] && !Indexer(il, n, Position(), &sel, &n_sel, error)) return false;
  if (!tap_index_.empty() && tap_index_[il] >= 0) {
    const auto k = static_cast<std::uint32_t>(tap_index_[il]);
    const std::uint32_t r = std::min(n, kTapRows);
    tap_has_sel_[k] = sel != nullptr;
    if (sel) {
      if (n_sel != tap_sel_count_) return Fail(error, "hy4: tap selection size mismatch");
      HY4_CHECK(hipMemcpyAsync(tap_sel_[k], sel + std::size_t{n - r} * n_sel, std::size_t{r} * n_sel * sizeof(std::int32_t),
                               hipMemcpyDeviceToDevice, st));
    }
  }

  // Scores over every key the pass can see, as GEMMs over (token, head)
  // rows: MLA here is MQA over one latent row per key, so each key row is
  // read once per pass rather than once per head.
  const std::uint32_t window = c.is_swa[il] ? c.swa_window : 0;
  const std::int32_t last = static_cast<std::int32_t>(Position() + n - 1);
  std::int32_t lo = 0;
  const float* keys = cache_[il];
  if (window) {
    lo = std::max<std::int32_t>(0, static_cast<std::int32_t>(Position()) - static_cast<std::int32_t>(window) + 1);
    HY4_CHECK(hy4_gather_rows(cache_[il], ring_[il], lo, static_cast<std::uint32_t>(last - lo + 1), dk, s.keys, st));
    keys = s.keys;
  }
  const std::uint32_t n_keys = static_cast<std::uint32_t>(last - lo + 1);
  if (sel) HY4_CHECK(hy4_selection_mask(sel, n_sel, mask_, n, n_keys, st));
  const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_k));
  const float zero = 0.0f;
  const float one = 1.0f;
  if (n <= Model::kVecMax) {
    // Each token over exactly the keys a one-token pass would see, so a
    // verify pass computes what decoding its tokens one by one would.
    for (std::uint32_t t = 0; t < n; ++t) {
      const std::int32_t p = static_cast<std::int32_t>(Position() + t);
      const std::int32_t lo_t = window ? std::max<std::int32_t>(0, p - static_cast<std::int32_t>(window) + 1) : 0;
      const auto keys_t = static_cast<std::uint32_t>(p - lo_t + 1);
      const float* k = keys + std::size_t(lo_t - lo) * dk;
      const int rows = static_cast<int>(heads);
      if (hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(keys_t), rows, static_cast<int>(dk), &scale,
                       k, static_cast<int>(dk), s.qfull + std::size_t{t} * heads * dk, static_cast<int>(dk), &zero,
                       s.scores, static_cast<int>(keys_t)) != HIPBLAS_STATUS_SUCCESS) {
        return Fail(error, "hy4: attention score gemm failed");
      }
      HY4_CHECK(hy4_attn_softmax(s.scores, 1, heads, keys_t, lo_t, s.pos + t, window,
                                 sel ? mask_ + std::size_t{t} * n_keys + (lo_t - lo) : nullptr, l.attn_sinks.f32(),
                                 st));
      if (hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(c.kv_lora), rows, static_cast<int>(keys_t),
                       &one, k, static_cast<int>(dk), s.scores, static_cast<int>(keys_t), &zero,
                       s.att + std::size_t{t} * heads * c.kv_lora,
                       static_cast<int>(c.kv_lora)) != HIPBLAS_STATUS_SUCCESS) {
        return Fail(error, "hy4: attention value gemm failed");
      }
    }
  }
  const std::uint32_t per = static_cast<std::uint32_t>(
      std::max<std::size_t>(1, kScoreFloats / (std::size_t{heads} * n_keys)));
  for (std::uint32_t t0 = n <= Model::kVecMax ? n : 0; t0 < n; t0 += per) {
    const std::uint32_t nt = std::min(per, n - t0);
    const int rows = static_cast<int>(nt * heads);
    const float* q = s.qfull + std::size_t{t0} * heads * dk;
    // S (row-major [rows][n_keys]) = scale Q K^T.
    if (hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, static_cast<int>(n_keys), rows, static_cast<int>(dk), &scale,
                     keys, static_cast<int>(dk), q, static_cast<int>(dk), &zero, s.scores,
                     static_cast<int>(n_keys)) != HIPBLAS_STATUS_SUCCESS) {
      return Fail(error, "hy4: attention score gemm failed");
    }
    HY4_CHECK(hy4_attn_softmax(s.scores, nt, heads, n_keys, lo, s.pos + t0, window,
                               sel ? mask_ + std::size_t{t0} * n_keys : nullptr, l.attn_sinks.f32(), st));
    // O (row-major [rows][kv_lora]) = P V, V the first kv_lora columns of the key rows.
    if (hipblasSgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, static_cast<int>(c.kv_lora), rows, static_cast<int>(n_keys),
                     &one, keys, static_cast<int>(dk), s.scores, static_cast<int>(n_keys), &zero,
                     s.att + std::size_t{t0} * heads * c.kv_lora, static_cast<int>(c.kv_lora)) != HIPBLAS_STATUS_SUCCESS) {
      return Fail(error, "hy4: attention value gemm failed");
    }
  }
  if (n > Model::kVecMax) {
    HY4_CHECK(hy4_mmq_dense_grouped_typed(l.wv_b.type, l.wv_b.data, s.att, s.attv, static_cast<int>(c.head_v),
                                          static_cast<int>(n), static_cast<int>(c.kv_lora), static_cast<int>(heads),
                                          st));
  } else {
    HY4_CHECK(hy4_gemv_q8_0_grouped(l.wv_b.data, s.att, s.attv, c.head_v, c.kv_lora, heads, n,
                                    std::uint64_t{heads} * c.kv_lora, c.kv_lora, std::uint64_t{heads} * c.head_v,
                                    c.head_v, st));
  }
  if (!MatVec(l.attn_gate, s.cur, s.gate, heads * c.head_v, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_sigmoid_mul(s.attv, s.gate, std::uint64_t{n} * heads * c.head_v, st));
  if (!MatVec(l.wo, s.attv, s.y, c.n_embd, heads * c.head_v, n, ex, error)) return false;
  HY4_CHECK(hy4_rms_norm(s.y, l.attn_post_norm.f32(), s.y, n, c.n_embd, c.n_embd, c.rms_eps, st));
  return true;
}

bool Session::Indexer(std::uint32_t il, std::uint32_t n, std::uint32_t pos0, const std::int32_t** sel,
                      std::uint32_t* n_sel, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *m.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_};
  const std::uint32_t d = c.idx_dim;
  const float base = c.rope_base;  // indexer layers are the sparse ones

  // Key side: normed key, then its value and gate projections.
  if (!MatVec(l.idx_k, s.cur, s.ik, d, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_rms_norm(s.ik, l.idx_k_norm.f32(), s.ik, n, d, d, c.rms_eps, st));
  if (!MatVec(l.idx_value, s.ik, s.ffn_a, d, d, n, ex, error)) return false;
  if (!MatVec(l.idx_gate, s.ik, s.ffn_b, d, d, n, ex, error)) return false;
  HY4_CHECK(hipMemcpy2DAsync(s.ivg, 2 * d * sizeof(float), s.ffn_a, d * sizeof(float), d * sizeof(float), n,
                             hipMemcpyDeviceToDevice, st));
  HY4_CHECK(hipMemcpy2DAsync(s.ivg + d, 2 * d * sizeof(float), s.ffn_b, d * sizeof(float), d * sizeof(float), n,
                             hipMemcpyDeviceToDevice, st));
  HY4_CHECK(hy4_store_rows(s.ivg, 2 * d, idx_rows_[il], Model::kIndexerRing, s.pos, n, 2 * d, st));

  // Pool the groups this chunk completes; a pooled key is roped at its first
  // member's position.
  std::vector<std::int32_t> groups, group_pos, group_hw;
  bool group_3d = false;
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t p = pos0 + i;
    if (p % c.kpool == c.kpool - 1) {
      groups.push_back(static_cast<std::int32_t>(p / c.kpool));
      const auto r = PassRope(p / c.kpool * c.kpool, pos0);
      group_pos.push_back(r[0]);
      group_hw.push_back(r[1]);
      group_hw.push_back(r[2]);
      group_3d = group_3d || r[1] != r[0] || r[2] != r[0];
    }
  }
  const std::uint32_t pool_rows = max_context_ / c.kpool + 1;
  if (!groups.empty()) {
    const std::uint32_t ng = static_cast<std::uint32_t>(groups.size());
    HY4_CHECK(hipMemcpyAsync(s.groups, groups.data(), ng * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
    HY4_CHECK(hipMemcpyAsync(s.group_pos, group_pos.data(), ng * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
    if (group_3d) {
      HY4_CHECK(hipMemcpyAsync(s.group_hw, group_hw.data(),
                               2 * ng * sizeof(std::int32_t),
                               hipMemcpyHostToDevice, st));
    }
    HY4_CHECK(hy4_kpool_pool(idx_rows_[il], Model::kIndexerRing, s.groups, ng, c.kpool, l.idx_ape.f32(), l.idx_comp_norm.f32(),
                             c.rms_eps, idx_pooled_[il], pool_rows, d, st));
    HY4_CHECK(hy4_rope_3d(idx_pooled_[il] + std::size_t(groups[0]) * d,
                          s.group_pos, group_3d ? s.group_hw : nullptr, ng, 1,
                          d, 0, d - c.n_rot, c.n_rot, base, st));
  }

  // Selection, once a token sees more pools than the top-k keeps; below
  // that every pool is selected, which is plain causal attention.
  const std::uint32_t top = c.idx_top_k / c.kpool - 1;
  const std::uint32_t n_pool_max = (pos0 + n - 1) / c.kpool;
  if (n_pool_max <= top) return true;
  std::vector<std::int32_t> n_pool(n);
  for (std::uint32_t i = 0; i < n; ++i) n_pool[i] = static_cast<std::int32_t>((pos0 + i) / c.kpool);
  HY4_CHECK(hipMemcpyAsync(s.n_pool, n_pool.data(), n * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
  if (!MatVec(l.idx_q_b, s.qr, s.iq, c.idx_heads * d, c.q_lora, n, ex, error)) return false;
  HY4_CHECK(hy4_rope_3d(s.iq, s.rope, s.rope_3d ? s.rope_hw : nullptr, n,
                        c.idx_heads, std::uint64_t{c.idx_heads} * d, d,
                        d - c.n_rot, c.n_rot, base, st));
  // The 1 / sqrt(d heads) weight scale does not change the selection.
  if (!MatVec(l.idx_proj, s.cur, s.iw, c.idx_heads, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_indexer_score(s.iq, s.iw, idx_pooled_[il], s.n_pool, n_pool_max, score_, n, c.idx_heads, d, st));
  HY4_CHECK(hy4_kpool_select(score_, s.n_pool, n_pool_max, s.pos, top, c.kpool, sel_, n, st));
  *sel = sel_;
  *n_sel = (top + 1) * c.kpool;
  return true;
}

bool Session::Ffn(std::uint32_t il, std::uint32_t n, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *m.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_};

  if (il < c.n_dense_lead) {
    const std::uint32_t f = c.n_ff_dense;
    if (!MatVec(l.ffn_gate, s.cur, s.ffn_a, f, c.n_embd, n, ex, error)) return false;
    if (!MatVec(l.ffn_up, s.cur, s.ffn_b, f, c.n_embd, n, ex, error)) return false;
    HY4_CHECK(hy4_swiglu(s.ffn_a, s.ffn_b, nullptr, s.ffn_c, n, f, 0.0f, st));
    return MatVec(l.ffn_down, s.ffn_c, s.y, c.n_embd, f, n, ex, error);
  }

  const std::uint32_t f = c.n_ff_exp;
  const std::uint32_t used = c.n_expert_used;
  // Shared expert first, into its own buffer.
  if (!MatVec(l.gate_shexp, s.cur, s.ffn_a, f, c.n_embd, n, ex, error)) return false;
  if (!MatVec(l.up_shexp, s.cur, s.ffn_b, f, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_swiglu(s.ffn_a, s.ffn_b, nullptr, s.ffn_c, n, f, 0.0f, st));
  if (!MatVec(l.down_shexp, s.ffn_c, s.shared, c.n_embd, f, n, ex, error)) return false;

  // Router on the full-width input; experts in the latent space.
  if (!MatVec(l.gate_inp, s.cur, s.logits_r, c.n_expert, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_route(s.logits_r, l.exp_bias ? l.exp_bias.f32() : nullptr, s.ids, s.route_w, n, c.n_expert, used,
                      c.expert_norm ? 1 : 0, c.expert_scale, st));
  if (!MatVec(l.routed_down, s.cur, s.latent, c.n_latent, c.n_embd, n, ex, error)) return false;
  if (n > Model::kVecMax) {
    if (l.gate_exps.type != l.up_exps.type) return Fail(error, "hy4: gate and up experts differ in type");
    // Bound the routed grids by the real expert buckets (results unchanged).
    HY4_CHECK(hipMemcpyAsync(s.ids_host, s.ids, std::size_t{n} * used * sizeof(std::int32_t), hipMemcpyDeviceToHost,
                             st));
    HY4_CHECK(hipStreamSynchronize(st));
    std::vector<unsigned int> counts(c.n_expert, 0);
    for (std::uint32_t i = 0; i < n * used; ++i) ++counts[static_cast<std::uint32_t>(s.ids_host[i]) % c.n_expert];
    hy4_mmq_set_routed_max_expert_rows(static_cast<int>(*std::max_element(counts.begin(), counts.end())));
    hy4_mmq_set_routed_tile_cols(hy4_mmq_routed_tile_cols_for_counts(counts.data(), static_cast<int>(c.n_expert)));
    HY4_CHECK(hy4_mmq_moe_swiglu_typed(l.gate_exps.type, l.gate_exps.data, l.up_exps.data, s.latent, s.ids,
                                       s.route_w, s.ffn_a, s.ffn_c, static_cast<int>(f),
                                       static_cast<int>(c.n_latent), static_cast<int>(n),
                                       static_cast<int>(c.n_expert), static_cast<int>(used), c.swiglu_clamp[il], st));
    HY4_CHECK(hy4_mmq_moe_typed(l.down_exps.type, l.down_exps.data, s.ffn_c, s.ids, s.ffn_a,
                                static_cast<int>(c.n_latent), static_cast<int>(f), static_cast<int>(n * used),
                                static_cast<int>(c.n_expert), 1, st));
    hy4_mmq_set_routed_max_expert_rows(0);
    hy4_mmq_set_routed_tile_cols(0);
    HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
    if (!MatVec(l.routed_up, s.moe, s.y, c.n_embd, c.n_latent, n, ex, error)) return false;
    HY4_CHECK(hy4_add(s.y, s.shared, s.y, std::uint64_t{n} * c.n_embd, st));
    return true;
  }
  // One token per launch: the vendored launcher sums a multi-token batch in
  // another order than one token, and a verify pass must compute what
  // one-token decoding would (no weight row is shared across tokens anyway).
  for (std::uint32_t t = 0; t < n; ++t) {
    const float* x = s.latent + std::size_t{t} * c.n_latent;
    const std::int32_t* ids = s.ids + std::size_t{t} * used;
    const std::size_t at = std::size_t{t} * used * f;
    HY4_CHECK(hy4_mmq_moe_vec_typed(l.gate_exps.type, l.gate_exps.data, x, ids, s.ffn_a + at, static_cast<int>(f),
                                    static_cast<int>(c.n_latent), 1, static_cast<int>(c.n_expert),
                                    static_cast<int>(used), st));
    HY4_CHECK(hy4_mmq_moe_vec_typed(l.up_exps.type, l.up_exps.data, x, ids, s.ffn_b + at, static_cast<int>(f),
                                    static_cast<int>(c.n_latent), 1, static_cast<int>(c.n_expert),
                                    static_cast<int>(used), st));
  }
  HY4_CHECK(hy4_swiglu(s.ffn_a, s.ffn_b, s.route_w, s.ffn_c, n * used, f, c.swiglu_clamp[il], st));
  // Down: every slot has its own input, so each slot is a one-expert row.
  HY4_CHECK(hy4_mmq_moe_vec_typed(l.down_exps.type, l.down_exps.data, s.ffn_c, s.ids, s.ffn_a,
                                  static_cast<int>(c.n_latent), static_cast<int>(f), static_cast<int>(n * used),
                                  static_cast<int>(c.n_expert), 1, st));
  HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
  if (!MatVec(l.routed_up, s.moe, s.y, c.n_embd, c.n_latent, n, ex, error)) return false;
  HY4_CHECK(hy4_add(s.y, s.shared, s.y, std::uint64_t{n} * c.n_embd, st));
  return true;
}

}  // namespace gufo::models::hy4_flash
