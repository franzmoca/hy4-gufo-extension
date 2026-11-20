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
/* hy4_rms_norm that also writes each output rounded to binary16 (FP16 bits,
 * the same layout) into out_half. */
int hy4_rms_norm_half(const float* x, const float* weight, float* out,
                      uint16_t* out_half, uint32_t rows, uint32_t dim,
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

/* hy4_hc_post(y, h, post_prev, h) followed by hy4_hc_pre(h, ...), with the
 * same results; batched passes update h inside the pre-mix's pass over it.
 * post_prev may be the same buffer as post. */
int hy4_hc_post_pre(const float* y, const float* post_prev, float* h,
                    const float* fn, const float* scale, const float* base,
                    float* out, float* post, uint32_t n, uint32_t hc,
                    uint32_t dim, float eps, float hc_eps, float magnitude,
                    hipStream_t stream);

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

/* Splits Q8_0 blocks into their values as binary16 bits (q, 32 per block)
 * and their scales (d, one per block), for hy4_mla_absorb. */
int hy4_q8_0_split(const void* src, uint64_t blocks, uint16_t* q, float* d,
                   hipStream_t stream);

/* MLA query absorption for prefill passes, on the matrix cores: out[t, h] =
 * [wk_b[h] . q[t, h, 0:nope], q[t, h, nope:nope + n_rot]] as binary16 bits,
 * [n][heads][kv_lora + n_rot], with wk_b [heads][kv_lora][nope] split by
 * hy4_q8_0_split and q [n][heads][nope + n_rot]. q is rounded to binary16,
 * the weights are exact, and each output sums in an order independent of n.
 * kv_lora must be 512, nope a multiple of 32 up to 256, n_rot a multiple of 8
 * up to 128. */
int hy4_mla_absorb(const uint16_t* wq, const float* wd, const float* q,
                   uint16_t* out, uint32_t n, uint32_t heads, uint32_t nope,
                   uint32_t n_rot, uint32_t kv_lora, hipStream_t stream);

/* Rotates consecutive pairs (rope type NORM) of the last n_rot values of
 * each of `heads` rows per token: row (t, h) starts at x + t * tok_stride +
 * h * head_stride + offset; pair i at position pos[t] turns by pos *
 * base^(-2i / n_rot). */
int hy4_rope(float* x, const int32_t* pos, uint32_t n, uint32_t heads, uint64_t tok_stride,
             uint64_t head_stride, uint32_t offset, uint32_t n_rot, float base, hipStream_t stream);
/* hy4_rope at 3D positions: `pos` is the temporal one and `pos_hw` [n][2]
 * the height and width ones (interleaved M-RoPE, sections [12, 10, 10]; n_rot
 * must be 64). A null `pos_hw` is hy4_rope. */
int hy4_rope_3d(float* x, const int32_t* pos, const int32_t* pos_hw, uint32_t n,
                uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
                uint32_t offset, uint32_t n_rot, float base,
                hipStream_t stream);

/* MLA attention with learnable sinks over latent rows (llama.cpp
 * build_attn_mha with wv_b applied by the caller). q[t, h, 0:dk] holds the
 * absorbed nope part (dv = kv_lora floats) and the roped tail; cache rows
 * are [dk] binary16 values (the normed latent, then the roped key tail),
 * stored at row pos % ring (hy4_store_rows_half); the arithmetic is FP32.
 * Token t at position pos[t] attends to positions
 * max(0, pos[t] - window + 1) .. pos[t] (window 0: no lower bound), or to
 * sel[t, 0:n_sel] (positions, -1 for an empty slot) when sel is not NULL.
 * out[t, h, 0:dv] = softmax(scale * q.k, sink[h]) . latent. Positions
 * above pos[t] and repeated selections are skipped. Each token's result
 * depends only on its own query and keys, so a verify pass computes what
 * one-token passes would. Hy4's shape only: dk 576, dv 512, heads a multiple
 * of 4. `work` holds hy4_mla_attention_work(n, heads, n_sel)
 * floats (n_sel 0 without a selection). */
uint64_t hy4_mla_attention_work(uint32_t n, uint32_t heads, uint32_t n_sel);
int hy4_mla_attention(const float* q, const uint16_t* cache, uint32_t ring,
                      const int32_t* pos, uint32_t window, const int32_t* sel,
                      uint32_t n_sel, const float* sinks, float scale,
                      float* out, uint32_t n, uint32_t heads, uint32_t dk,
                      uint32_t dv, float* work, uint64_t work_floats,
                      hipStream_t stream);

/* hy4_mla_attention for prefill on the matrix cores (binary16 operands, FP32
 * softmax and accumulation; q as binary16 bits from hy4_mla_absorb), each
 * token over exactly its own keys, so a token's result does not depend on the
 * pass it is in. Key position k is row k % ring of `keys` [ring][dk] (the
 * binary16 cache rows); token t sees max(lo, pos[t] - window + 1)
 * .. pos[t] (lo without a window), or the positions of sel[t, 0:n_sel] in that
 * range. out [n][heads][dv] is binary16 bits (the rounding the wv_b
 * projection applies to its input). dk must be 576, dv 512 and heads a
 * multiple of 32. */
int hy4_mla_prefill(const uint16_t* q, const uint16_t* keys, uint32_t ring,
                    int32_t lo, const int32_t* pos, uint32_t window,
                    const int32_t* sel, uint32_t n_sel, const float* sinks,
                    float scale, uint16_t* out, uint32_t n, uint32_t heads,
                    uint32_t dk, uint32_t dv, hipStream_t stream);

/* hy4_store_rows into binary16: dst row pos[t] % ring = src row t rounded
 * to binary16 (round to nearest even). */
int hy4_store_rows_half(const float* src, uint64_t src_stride, uint16_t* dst,
                        uint32_t ring, const int32_t* pos, uint32_t n,
                        uint32_t dim, hipStream_t stream);
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

/* x[t, r, 0:dim] += bias[r, 0:dim] for rows_per_token rows per token (bias
 * [rows_per_token][dim]). */
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
 * rest of sel[t, 0:n_sel] is -1. n_sel = (k + 1) * kpool. Groups are listed
 * by descending score, the lower group first among equal scores. For
 * k > 512 the picked scores are overwritten with -inf. */
int hy4_kpool_select(const float* score, const int32_t* n_pool, uint32_t n_pool_max, const int32_t* pos,
                     uint32_t k, uint32_t kpool, int32_t* sel, uint32_t n, hipStream_t stream);

/* Argmax over n rows of `dim` floats. */
/* Batched attention support (the scores come from a GEMM over every key).
 * Rows r = t * heads + h of `scores` [rows][n_keys] hold scaled q.k for the
 * keys at positions key_pos0 + j. In place: row softmax over the keys token t
 * sees (key position <= pos[t], within `window` when non-zero, and, with a
 * mask [n][n_keys], where it is 0), with the head's sink in the denominator.
 * Masked keys get probability 0. */
int hy4_attn_softmax(float* scores, uint32_t n, uint32_t heads, uint32_t n_keys,
                     int32_t key_pos0, const int32_t* pos, uint32_t window,
                     const float* mask, const float* sinks, hipStream_t stream);
/* dst [count][dim] = binary16 rows of the positions idx[j] of a ring, as
 * floats; a position < 0 or > limit is missing (a zero row). mask
 * [rows][n_keys] = -inf at the missing j, 0 at the other j < count and at every
 * j >= count. */
int hy4_gather_indexed(const uint16_t* src, uint32_t ring, const int32_t* idx,
                       uint32_t count, int32_t limit, uint32_t dim, float* dst,
                       float* mask, uint32_t rows, uint32_t n_keys,
                       hipStream_t stream);
/* x += a * b, elementwise over `count` floats; b repeats every `period` floats. */
int hy4_mul_add(float* x, const float* a, const float* b, uint64_t count, uint64_t period, hipStream_t stream);
/* out [blocks * 32] = dequantized Q8_0 blocks. */
int hy4_dequant_q8_0(const void* src, uint64_t blocks, float* out, hipStream_t stream);
/* *out = the index of the largest of x [dim] (dim > 0); ties go to the lower
 * index and NaN counts as -inf. work holds hy4_argmax_work_floats() floats. */
uint64_t hy4_argmax_work_floats(void);
int hy4_argmax(const float* x, uint32_t dim, float* work, int32_t* out,
               hipStream_t stream);
/* dst [blocks] = Q8_0 blocks of src [blocks * 32] binary16 values:
 * d = max |v| / 127 (binary16), q = v / d rounded to nearest even. */
int hy4_quantize_q8_0_half(const uint16_t* src, uint64_t blocks, void* dst,
                           hipStream_t stream);
/* y [n][m] = x [n][k] w^T for w [m][k], all FP32; each output sums its k
 * products in order, so a token's row does not depend on n. k must be a
 * multiple of 32. */
int hy4_sgemm_nt(const float* x, const float* w, float* y, uint32_t n,
                 uint32_t m, uint32_t k, hipStream_t stream);

/* Batched Q8_0, Q4_K and Q6_K projections on the FP16 matrix cores
 * (hy4_f16_mm.hip.cpp). For each group z < groups, out[t, z m + r] =
 * W_z[r] . x[t, z k : z k + k] over n tokens, W_z the z-th [m][k] matrix
 * stored back to back; x rows hold groups * k floats, rounded to FP16 as they
 * are staged, and out rows groups * m floats; non-finite outputs read as 0.
 * With HY4_F16_X_HALF the x rows are FP16 bits, with HY4_F16_OUT_HALF the out
 * rows are FP16 (rounded from the FP32 sums). m must be a multiple of 16 and
 * k of 256; returns 1 otherwise or for other types. */
#define HY4_F16_X_HALF 1u
#define HY4_F16_OUT_HALF 2u
int hy4_f16_dense(int type, const void* w, const void* x, void* out, uint32_t m,
                  uint32_t n, uint32_t k, uint32_t groups, uint32_t io,
                  hipStream_t stream);
/* hy4_f16_dense with each output multiplied by sigmoid(gate) before it is
 * stored (and rounded, with HY4_F16_OUT_HALF); gate is FP32 in out's layout,
 * [n][groups * m]. */
int hy4_f16_dense_gated(int type, const void* w, const void* x, void* out,
                        uint32_t m, uint32_t n, uint32_t k, uint32_t groups,
                        uint32_t io, const float* gate, hipStream_t stream);
/* out1 = W1 x and out2 = W2 x in one launch, the same results as two
 * hy4_f16_dense calls of one group: W1 [m1][k] and W2 [m2][k] of one type. */
int hy4_f16_dense_pair(int type, const void* w1, void* out1, uint32_t m1,
                       const void* w2, void* out2, uint32_t m2, const void* x,
                       uint32_t n, uint32_t k, uint32_t io, hipStream_t stream);
/* Batched routed experts on the FP16 matrix cores. rows lists the slots
 * grouped by expert, expert e owning rows[bounds[e] .. bounds[e + 1]), and
 * tiles[0:n_tiles] names 128 of them each as expert | index << 16 (see
 * hy4_f16_routed_plan); entries < 0 are skipped. A slot s reads x row
 * s / x_div (FP16 bits); expert e's matrices start e * expert_bytes into w
 * (and w_up). With w_up, y (FP16 bits) [s, r] = route_w[s] *
 * silu(min(gate, limit)) * clamp(up, +-limit) (no clamp for limit <= 0);
 * without it, y (FP32) [s, r] = (W x)[r]. Non-finite products read as 0. */
int hy4_f16_routed(int type, const void* w, const void* w_up,
                   uint64_t expert_bytes, const void* x, uint32_t x_div,
                   const int32_t* rows, const int32_t* bounds,
                   const int32_t* tiles, uint32_t n_tiles, const float* route_w,
                   float limit, void* y, uint32_t m, uint32_t k,
                   hipStream_t stream);
/* Host side: rows[slots], bounds[n_experts + 1] and tiles (at most
 * slots / 128 + n_experts entries) from the slot experts ids[slots]; ids
 * outside 0 .. n_experts - 1 are skipped. Returns the tile count. */
uint32_t hy4_f16_routed_plan(const int32_t* ids, uint32_t slots,
                             uint32_t n_experts, int32_t* rows, int32_t* bounds,
                             int32_t* tiles);

#ifdef __cplusplus
}
#endif
