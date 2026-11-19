/* Hy4 Flash kernels outside the vendored llama.cpp matvecs.
 *
 * Every entry mirrors one operation of the llama.cpp hy_v4_flash graph
 * (src/models/hy-v4-flash.cpp of ggml-org/llama.cpp#32517) on FP32 device
 * buffers, for `n` tokens stored one row per token. All launches go to the
 * given stream. Functions return 0 on success. */
#pragma once

#include <stdint.h>

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ggml type codes used here. */
enum {
  HY4_TYPE_F32 = 0,
  HY4_TYPE_F16 = 1,
  HY4_TYPE_Q8_0 = 8,
  HY4_TYPE_Q4_K = 12,
  HY4_TYPE_Q6_K = 14,
  HY4_TYPE_F8_E4M3 = 43,
};

/* out[t, 0:dim] = row tokens[t] of a Q8_0 table [rows][dim]. */
int hy4_embed_q8_0(const void* table, const int32_t* tokens, uint32_t n, uint32_t dim, float* out,
                   hipStream_t stream);

/* out[r] = x[r] / rms(x[r]) * weight (weight may be NULL), for `rows` rows of
 * `dim` floats at stride `stride` (in and out). */
int hy4_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t dim,
                 uint32_t stride, float eps, hipStream_t stream);

/* iHC pre-mix of one layer half (llama.cpp build_hc_pre) for n tokens of
 * `hc` streams h[t, hc, dim]: mixes = fn [2 hc][hc dim] . rms_norm(flat h)
 * (eps), pre = sigmoid(mix * scale[0] + base[0:hc]) + hc_eps, post =
 * sigmoid(mix * scale[1] + base[hc:2hc]) * magnitude + hc_eps; out[t] =
 * sum_i pre_i h[t, i]. post[t, hc] is written for hy4_hc_post. With
 * post == NULL this is the head reduce (fn [hc][hc dim], scale[0], base[hc]). */
int hy4_hc_pre(const float* h, const float* fn, const float* scale, const float* base, float* out,
               float* post, uint32_t n, uint32_t hc, uint32_t dim, float eps, float hc_eps,
               float magnitude, hipStream_t stream);

/* h[t, i] = residual[t, i] + y[t] * post[t, i] for each stream i (iHC: the
 * comb is the identity). h may alias residual. */
int hy4_hc_post(const float* y, const float* residual, const float* post, float* h, uint32_t n,
                uint32_t hc, uint32_t dim, hipStream_t stream);

/* h[t, i] = x[t] for every stream i. */
int hy4_hc_init(const float* x, float* h, uint32_t n, uint32_t hc, uint32_t dim, hipStream_t stream);

/* y[t, r] = W[r] . x[t] for F32 or F16 W [m][k] (row-major), n tokens. */
int hy4_gemv_float(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k,
                   uint32_t n, hipStream_t stream);

/* Q8_0 matvecs for any k (a multiple of 32), grouped: for group g < groups,
 * y[t, g, 0:m] = W_g . x[t, g, 0:k] with W_g the g-th [m][k] block of w.
 * x of token t, group g starts at x + t * x_tok + g * x_grp; y likewise. */
int hy4_gemv_q8_0_grouped(const void* w, const float* x, float* y, uint32_t m, uint32_t k,
                          uint32_t groups, uint32_t n, uint64_t x_tok, uint64_t x_grp, uint64_t y_tok,
                          uint64_t y_grp, hipStream_t stream);

/* Rotates consecutive pairs (rope type NORM) of the last n_rot values of
 * each of `heads` rows per token: row (t, h) starts at x + t * tok_stride +
 * h * head_stride + offset; pair i at position pos[t] turns by pos *
 * base^(-2i / n_rot). */
int hy4_rope(float* x, const int32_t* pos, uint32_t n, uint32_t heads, uint64_t tok_stride,
             uint64_t head_stride, uint32_t offset, uint32_t n_rot, float base, hipStream_t stream);

/* MLA attention with learnable sinks over latent rows (llama.cpp
 * build_attn_mha with wv_b applied by the caller). q[t, h, 0:dk] holds the
 * absorbed nope part (dv = kv_lora floats) and the roped tail; cache rows
 * are [dk] floats (the normed latent, then the roped key tail), stored at
 * row pos % ring. Token t at position pos[t] attends to positions
 * max(0, pos[t] - window + 1) .. pos[t] (window 0: no lower bound), or to
 * sel[t, 0:n_sel] (positions, -1 for an empty slot) when sel is not NULL.
 * out[t, h, 0:dv] = softmax(scale * q.k, sink[h]) . latent. */
int hy4_mla_attention(const float* q, const float* cache, uint32_t ring, const int32_t* pos,
                      uint32_t window, const int32_t* sel, uint32_t n_sel, const float* sinks,
                      float scale, float* out, uint32_t n, uint32_t heads, uint32_t dk, uint32_t dv,
                      hipStream_t stream);

/* Copies n rows of `dim` floats into a ring: dst row pos[t] % ring. */
int hy4_store_rows(const float* src, uint64_t src_stride, float* dst, uint32_t ring, const int32_t* pos,
                   uint32_t n, uint32_t dim, hipStream_t stream);

/* x[i] *= sigmoid(gate[i]) for n * dim values. */
int hy4_sigmoid_mul(float* x, const float* gate, uint64_t count, hipStream_t stream);

/* y[t] = silu(gate[t]) * up[t] (limit <= 0) or silu(min(gate, limit)) *
 * clamp(up, +-limit), times route_w[t] when route_w is not NULL, for rows
 * of dim floats. */
int hy4_swiglu(const float* gate, const float* up, const float* route_w, float* y, uint32_t rows,
               uint32_t dim, float limit, hipStream_t stream);

/* Sigmoid routing (llama.cpp gating 2 with expert bias): probs =
 * sigmoid(logits), pick the k largest probs + bias, weights = probs of the
 * picks normalized to sum 1 (norm) and times `scale`. ids/weights [n][k]. */
int hy4_route(const float* logits, const float* bias, int32_t* ids, float* weights, uint32_t n,
              uint32_t n_experts, uint32_t k, int norm, float scale, hipStream_t stream);

/* out[t] = sum_s y[t, s] over k slots of dim floats. */
int hy4_sum_slots(const float* y, float* out, uint32_t n, uint32_t k, uint32_t dim, hipStream_t stream);

/* out = a + b (count floats); out may alias either. */
int hy4_add(const float* a, const float* b, float* out, uint64_t count, hipStream_t stream);

/* Engram rows: out[t, c, 0:d] = rms_norm(fp8 rows[t, c]) * norm_w[c] *
 * valid[t, c], rows given as FP8 E4M3 bytes [n][cols][d]. */
int hy4_engram_rows(const uint8_t* rows, const float* valid, const float* norm_w, float* out, uint32_t n,
                    uint32_t cols, uint32_t d, float eps, hipStream_t stream);

/* Engram gate and update of the hc streams h[t, hc, dim] from kv[t, hc + 1,
 * dim] (bias added already): for stream i, dot = sum_c rms(h_i)[c] *
 * rms(key_i)[c] * gate_w[i, c] / sqrt(dim), g = sigmoid(sign(dot) *
 * sqrt(max(|dot|, 1e-6))); h_i += any[t] * rms(value) * gamma * (g +
 * hc_scale[i]). */
int hy4_engram_apply(float* h, const float* kv, const float* gate_w, const float* hc_scale,
                     const float* gamma, const float* any, uint32_t n, uint32_t hc, uint32_t dim,
                     float eps, hipStream_t stream);

/* out[t, r, 0:dim] = x[t, r, 0:dim] + bias[r] for rows_per_token rows. */
int hy4_add_bias_rows(float* x, const float* bias, uint32_t n, uint32_t rows_per_token, uint32_t dim,
                      hipStream_t stream);

/* kpool pooling of complete groups: for each group g in groups[0:n_groups],
 * pooled[g, c] = sum_j softmax_j(gate[g*kpool + j, c] + ape[j, c]) *
 * value[g*kpool + j, c], from rows[.., 2 dim] = value | gate stored at row
 * (g*kpool + j) % ring; then rms-normed with norm_w (eps). Output
 * pooled_out[g % pool_ring, dim]. */
int hy4_kpool_pool(const float* rows, uint32_t ring, const int32_t* groups, uint32_t n_groups,
                   uint32_t kpool, const float* ape, const float* norm_w, float eps, float* pooled_out,
                   uint32_t pool_ring, uint32_t dim, hipStream_t stream);

/* Indexer scores: score[t, g] = sum_h relu(q[t, h] . pooled[g]) * w[t, h]
 * for g < n_pool[t] (others -inf), heads of dim floats, pooled rows dim
 * floats, n_pool_max columns per token. */
int hy4_indexer_score(const float* q, const float* w, const float* pooled, const int32_t* n_pool,
                      uint32_t n_pool_max, float* score, uint32_t n, uint32_t heads, uint32_t dim,
                      hipStream_t stream);

/* Converts logits rows to the top-k selection of positions: for token t the
 * k highest finite scores' groups g expand to positions g*kpool + j, then
 * the token's own group positions (pos[t] - pos[t] % kpool .. pos[t]); the
 * rest of sel[t, 0:n_sel] is -1. n_sel = (k + 1) * kpool. */
int hy4_kpool_select(const float* score, const int32_t* n_pool, uint32_t n_pool_max, const int32_t* pos,
                     uint32_t k, uint32_t kpool, int32_t* sel, uint32_t n, hipStream_t stream);

/* Argmax over n rows of `dim` floats. */
/* Batched attention support (the scores come from a GEMM over every key).
 * Rows r = t * heads + h of `scores` [rows][n_keys] hold scaled q.k for the
 * keys at positions key_pos0 + j. In place: row softmax over the keys token t
 * sees (key position <= pos[t], within `window` when non-zero, and, with a
 * mask [n][n_keys], where it is 0), with the head's sink in the denominator.
 * Masked keys get probability 0. */
int hy4_attn_softmax(float* scores, uint32_t n, uint32_t heads, uint32_t n_keys, int32_t key_pos0,
                     const int32_t* pos, uint32_t window, const float* mask, const float* sinks,
                     hipStream_t stream);
/* mask [n][n_keys] = 0 at the selected positions sel [n][n_sel] (-1 unused), -inf elsewhere. */
int hy4_selection_mask(const int32_t* sel, uint32_t n_sel, float* mask, uint32_t n, uint32_t n_keys,
                       hipStream_t stream);
/* dst [count][dim] = rows of the positions first .. first + count - 1 of a ring. */
int hy4_gather_rows(const float* src, uint32_t ring, int32_t first, uint32_t count, uint32_t dim, float* dst,
                    hipStream_t stream);
int hy4_argmax(const float* x, uint32_t n, uint32_t dim, int32_t* out, hipStream_t stream);

#ifdef __cplusplus
}
#endif
