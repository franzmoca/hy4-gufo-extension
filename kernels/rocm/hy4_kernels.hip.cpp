// Hy4 Flash kernels outside the vendored llama.cpp matvecs (hy4_kernels.h).
// Written for correctness first: one block per row or token, FP32
// arithmetic, shared-memory reductions.

#include "hy4_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr int kBlock = 256;

struct BlockQ8_0 {
  __half d;
  int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block layout");

__device__ __forceinline__ float WarpSum(float v) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
  return v;
}

__device__ __forceinline__ float WarpMax(float v) {
  for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o));
  return v;
}

// Sum over the block; every thread receives the result. `scratch` holds at
// least blockDim.x / warpSize floats.
__device__ float BlockSum(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  v = WarpSum(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = 0.0f;
  for (int i = 0; i < warps; ++i) total += scratch[i];
  return total;
}

__device__ float BlockMax(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  v = WarpMax(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = -INFINITY;
  for (int i = 0; i < warps; ++i) total = fmaxf(total, scratch[i]);
  return total;
}

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float Fp8E4M3(uint8_t v) {
  const int sign = v >> 7;
  const int exp = (v >> 3) & 0xF;
  const int man = v & 0x7;
  float r;
  if (exp == 0) {
    r = ldexpf((float)man, -9);  // man / 8 * 2^-6
  } else if (exp == 0xF && man == 0x7) {
    r = NAN;
  } else {
    r = ldexpf(1.0f + (float)man * 0.125f, exp - 7);
  }
  return sign ? -r : r;
}

int Status() { return hipGetLastError() == hipSuccess ? 0 : 1; }

// ---------------------------------------------------------------- embedding

__global__ void EmbedQ8_0(const BlockQ8_0* table, const int32_t* tokens, uint32_t dim, float* out) {
  const uint32_t t = blockIdx.x;
  const BlockQ8_0* row = table + (uint64_t)tokens[t] * (dim / 32);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const BlockQ8_0& b = row[i / 32];
    out[(uint64_t)t * dim + i] = __half2float(b.d) * (float)b.qs[i % 32];
  }
}

// --------------------------------------------------------------------- norm

__global__ void RmsNorm(const float* x, const float* weight, float* out, uint32_t dim, uint32_t stride,
                        float eps) {
  __shared__ float scratch[32];
  const float* xr = x + (uint64_t)blockIdx.x * stride;
  float* yr = out + (uint64_t)blockIdx.x * stride;
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / (float)dim + eps);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xr[i] * scale;
    yr[i] = weight ? v * weight[i] : v;
  }
}

// ------------------------------------------------------------- hyper-conn.

// One block per token. hc <= 8 (2 hc mixes at most 16).
__global__ void HcPre(const float* h, const float* fn, const float* scale, const float* base, float* out,
                      float* post, uint32_t hc, uint32_t dim, float eps, float hc_eps, float magnitude) {
  __shared__ float scratch[32];
  __shared__ float pre_s[8];
  const uint32_t t = blockIdx.x;
  const uint32_t flat = hc * dim;
  const float* ht = h + (uint64_t)t * flat;
  const uint32_t n_mix = post ? 2 * hc : hc;

  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < flat; i += blockDim.x) ss += ht[i] * ht[i];
  ss = BlockSum(ss, scratch);
  const float rs = rsqrtf(ss / (float)flat + eps);

  float mix[16];
  for (uint32_t m = 0; m < n_mix; ++m) mix[m] = 0.0f;
  for (uint32_t i = threadIdx.x; i < flat; i += blockDim.x) {
    const float v = ht[i] * rs;
    for (uint32_t m = 0; m < n_mix; ++m) mix[m] += fn[(uint64_t)m * flat + i] * v;
  }
  for (uint32_t m = 0; m < n_mix; ++m) mix[m] = BlockSum(mix[m], scratch);

  if (threadIdx.x == 0) {
    for (uint32_t i = 0; i < hc; ++i) {
      pre_s[i] = Sigmoid(mix[i] * scale[0] + base[i]) + hc_eps;
      if (post) {
        post[(uint64_t)t * hc + i] = Sigmoid(mix[hc + i] * scale[1] + base[hc + i]) * magnitude + hc_eps;
      }
    }
  }
  __syncthreads();
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < hc; ++i) acc += ht[(uint64_t)i * dim + c] * pre_s[i];
    out[(uint64_t)t * dim + c] = acc;
  }
}

__global__ void HcPost(const float* y, const float* residual, const float* post, float* h, uint32_t hc,
                       uint32_t dim) {
  const uint32_t t = blockIdx.y;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    const float yv = y[(uint64_t)t * dim + c];
    for (uint32_t i = 0; i < hc; ++i) {
      const uint64_t at = ((uint64_t)t * hc + i) * dim + c;
      h[at] = residual[at] + yv * post[(uint64_t)t * hc + i];
    }
  }
}

__global__ void HcInit(const float* x, float* h, uint32_t hc, uint32_t dim) {
  const uint32_t t = blockIdx.y;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    const float v = x[(uint64_t)t * dim + c];
    for (uint32_t i = 0; i < hc; ++i) h[((uint64_t)t * hc + i) * dim + c] = v;
  }
}

// --------------------------------------------------------------------- gemv

constexpr uint32_t kMaxTok = 8;

template <typename W>
__device__ __forceinline__ float LoadW(const W* w, uint64_t i);
template <>
__device__ __forceinline__ float LoadW<float>(const float* w, uint64_t i) {
  return w[i];
}
template <>
__device__ __forceinline__ float LoadW<__half>(const __half* w, uint64_t i) {
  return __half2float(w[i]);
}

// One block per output row, all tokens (n <= kMaxTok) per weight read.
template <typename W>
__global__ void GemvFloat(const W* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t n) {
  __shared__ float scratch[32];
  const uint32_t r = blockIdx.x;
  const W* wr = w + (uint64_t)r * k;
  float acc[kMaxTok];
  for (uint32_t t = 0; t < n; ++t) acc[t] = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const float wv = LoadW<W>(wr, i);
    for (uint32_t t = 0; t < n; ++t) acc[t] += wv * x[(uint64_t)t * k + i];
  }
  for (uint32_t t = 0; t < n; ++t) {
    const float s = BlockSum(acc[t], scratch);
    if (threadIdx.x == 0) y[(uint64_t)t * m + r] = s;
  }
}

// One block per (row, group), all tokens per weight read.
__global__ void GemvQ8_0Grouped(const BlockQ8_0* w, const float* x, float* y, uint32_t m, uint32_t k,
                                uint32_t n, uint64_t x_tok, uint64_t x_grp, uint64_t y_tok, uint64_t y_grp) {
  __shared__ float scratch[32];
  const uint32_t r = blockIdx.x;
  const uint32_t g = blockIdx.y;
  const uint32_t nb = k / 32;
  const BlockQ8_0* wr = w + ((uint64_t)g * m + r) * nb;
  float acc[kMaxTok];
  for (uint32_t t = 0; t < n; ++t) acc[t] = 0.0f;
  for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
    const BlockQ8_0& b = wr[i / 32];
    const float wv = __half2float(b.d) * (float)b.qs[i % 32];
    for (uint32_t t = 0; t < n; ++t) acc[t] += wv * x[t * x_tok + g * x_grp + i];
  }
  for (uint32_t t = 0; t < n; ++t) {
    const float s = BlockSum(acc[t], scratch);
    if (threadIdx.x == 0) y[t * y_tok + g * y_grp + r] = s;
  }
}

// --------------------------------------------------------------------- rope

__global__ void Rope(float* x, const int32_t* pos, uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
                     uint32_t offset, uint32_t n_rot, float base) {
  const uint32_t t = blockIdx.y;
  const uint32_t pairs = n_rot / 2;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * pairs) return;
  const uint32_t h = idx / pairs;
  const uint32_t i = idx % pairs;
  float* row = x + t * tok_stride + h * head_stride + offset;
  const float theta = (float)pos[t] * powf(base, -2.0f * (float)i / (float)n_rot);
  float s, c;
  sincosf(theta, &s, &c);
  const float x0 = row[2 * i];
  const float x1 = row[2 * i + 1];
  row[2 * i] = x0 * c - x1 * s;
  row[2 * i + 1] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- attention

constexpr uint32_t kMaxKeys = 4096;

// One block per (token, head). Keys are scored one warp per key.
__global__ void MlaAttention(const float* q, const float* cache, uint32_t ring, const int32_t* pos,
                             uint32_t window, const int32_t* sel, uint32_t n_sel, const float* sinks, float scale,
                             float* out, uint32_t heads, uint32_t dk, uint32_t dv) {
  extern __shared__ float smem[];
  __shared__ float scratch[32];
  float* qs = smem;          // dk
  float* sc = smem + dk;     // keys
  int32_t* kp = (int32_t*)(sc + kMaxKeys);  // key positions

  const uint32_t t = blockIdx.x;
  const uint32_t h = blockIdx.y;
  const float* qr = q + ((uint64_t)t * heads + h) * dk;
  for (uint32_t i = threadIdx.x; i < dk; i += blockDim.x) qs[i] = qr[i];

  // The key positions this token attends to.
  uint32_t n_keys = 0;
  const int32_t p = pos[t];
  if (sel) {
    if (threadIdx.x == 0) {
      for (uint32_t i = 0; i < n_sel && n_keys < kMaxKeys; ++i) {
        const int32_t s = sel[(uint64_t)t * n_sel + i];
        if (s >= 0) kp[n_keys++] = s;
      }
      sc[0] = (float)n_keys;  // passed through shared memory below
    }
    __syncthreads();
    n_keys = (uint32_t)sc[0];
    __syncthreads();
  } else {
    const int32_t lo = window ? max(0, p - (int32_t)window + 1) : 0;
    n_keys = (uint32_t)(p - lo + 1);
    if (n_keys > kMaxKeys) n_keys = kMaxKeys;  // callers keep spans below this
    const int32_t first = p - (int32_t)n_keys + 1;
    for (uint32_t i = threadIdx.x; i < n_keys; i += blockDim.x) kp[i] = first + (int32_t)i;
  }
  __syncthreads();

  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t warps = blockDim.x / warpSize;
  for (uint32_t j = warp; j < n_keys; j += warps) {
    const float* kr = cache + (uint64_t)((uint32_t)kp[j] % ring) * dk;
    float d = 0.0f;
    for (uint32_t i = lane; i < dk; i += warpSize) d += qs[i] * kr[i];
    d = WarpSum(d);
    if (lane == 0) sc[j] = d * scale;
  }
  __syncthreads();

  float mx = sinks ? sinks[h] : -INFINITY;
  for (uint32_t j = threadIdx.x; j < n_keys; j += blockDim.x) mx = fmaxf(mx, sc[j]);
  mx = BlockMax(mx, scratch);
  float den = 0.0f;
  for (uint32_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    const float e = expf(sc[j] - mx);
    sc[j] = e;
    den += e;
  }
  den = BlockSum(den, scratch);
  if (sinks) den += expf(sinks[h] - mx);
  __syncthreads();
  const float inv = 1.0f / den;
  float* orow = out + ((uint64_t)t * heads + h) * dv;
  for (uint32_t c = threadIdx.x; c < dv; c += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t j = 0; j < n_keys; ++j) acc += sc[j] * cache[(uint64_t)((uint32_t)kp[j] % ring) * dk + c];
    orow[c] = acc * inv;
  }
}

__global__ void StoreRows(const float* src, uint64_t src_stride, float* dst, uint32_t ring, const int32_t* pos,
                          uint32_t dim) {
  const uint32_t t = blockIdx.y;
  const uint64_t row = (uint64_t)((uint32_t)pos[t] % ring);
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    dst[row * dim + c] = src[t * src_stride + c];
  }
}

// ----------------------------------------------------------- elementwise

__global__ void SigmoidMul(float* x, const float* gate, uint64_t count) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    x[i] *= Sigmoid(gate[i]);
  }
}

__global__ void Swiglu(const float* gate, const float* up, const float* route_w, float* y, uint32_t dim,
                       float limit) {
  const uint32_t r = blockIdx.y;
  const float rw = route_w ? route_w[r] : 1.0f;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    float g = gate[(uint64_t)r * dim + c];
    float u = up[(uint64_t)r * dim + c];
    if (limit > 0.0f) {
      g = fminf(g, limit);
      u = fminf(fmaxf(u, -limit), limit);
    }
    y[(uint64_t)r * dim + c] = Silu(g) * u * rw;
  }
}

__global__ void Add(const float* a, const float* b, float* out, uint64_t count) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    out[i] = a[i] + b[i];
  }
}

__global__ void SumSlots(const float* y, float* out, uint32_t k, uint32_t dim) {
  const uint32_t t = blockIdx.y;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    float acc = 0.0f;
    for (uint32_t s = 0; s < k; ++s) acc += y[((uint64_t)t * k + s) * dim + c];
    out[(uint64_t)t * dim + c] = acc;
  }
}

__global__ void AddBiasRows(float* x, const float* bias, uint32_t rows_per_token, uint32_t dim) {
  const uint32_t t = blockIdx.y;
  const uint64_t total = (uint64_t)rows_per_token * dim;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < total;
       i += (uint64_t)gridDim.x * blockDim.x) {
    x[(uint64_t)t * total + i] += bias[i];
  }
}

// ---------------------------------------------------------------- routing

// One block per token, n_experts <= blockDim.x * 4.
__global__ void Route(const float* logits, const float* bias, int32_t* ids, float* weights, uint32_t n_experts,
                      uint32_t k, int norm, float scale) {
  extern __shared__ float rs[];
  float* prob = rs;
  float* sel = rs + n_experts;
  __shared__ float scratch[32];
  __shared__ int32_t picked;
  const uint32_t t = blockIdx.x;
  for (uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    const float p = Sigmoid(logits[(uint64_t)t * n_experts + e]);
    prob[e] = p;
    sel[e] = p + (bias ? bias[e] : 0.0f);
  }
  __syncthreads();
  float sum = 0.0f;
  for (uint32_t s = 0; s < k; ++s) {
    float best = -INFINITY;
    for (uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) best = fmaxf(best, sel[e]);
    best = BlockMax(best, scratch);
    if (threadIdx.x == 0) picked = -1;
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
      if (sel[e] == best) atomicMax(&picked, (int32_t)(n_experts - 1 - e));  // lowest index wins
    }
    __syncthreads();
    const int32_t e = (int32_t)n_experts - 1 - picked;
    if (threadIdx.x == 0) {
      ids[(uint64_t)t * k + s] = e;
      weights[(uint64_t)t * k + s] = prob[e];
      sel[e] = -INFINITY;
    }
    sum += prob[e];
    __syncthreads();
  }
  if (threadIdx.x < k) {
    float w = weights[(uint64_t)t * k + threadIdx.x];
    if (norm) w /= fmaxf(sum, 6.103515625e-5f);
    weights[(uint64_t)t * k + threadIdx.x] = w * scale;
  }
}

// ----------------------------------------------------------------- engram

// One block per (token, column) row of d values.
__global__ void EngramRows(const uint8_t* rows, const float* valid, const float* norm_w, float* out,
                           uint32_t cols, uint32_t d, float eps) {
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t c = blockIdx.x;
  const uint8_t* r = rows + ((uint64_t)t * cols + c) * d;
  float v = 0.0f;
  float ss = 0.0f;
  if (threadIdx.x < d) {
    v = Fp8E4M3(r[threadIdx.x]);
    ss = v * v;
  }
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / (float)d + eps);
  if (threadIdx.x < d) {
    out[((uint64_t)t * cols + c) * d + threadIdx.x] =
        v * scale * norm_w[(uint64_t)c * d + threadIdx.x] * valid[(uint64_t)t * cols + c];
  }
}

// One block per (token, stream).
__global__ void EngramApply(float* h, const float* kv, const float* gate_w, const float* hc_scale,
                            const float* gamma, const float* any, uint32_t hc, uint32_t dim, float eps) {
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t i = blockIdx.x;
  float* hi = h + ((uint64_t)t * hc + i) * dim;
  const float* key = kv + ((uint64_t)t * (hc + 1) + i) * dim;
  const float* value = kv + ((uint64_t)t * (hc + 1) + hc) * dim;
  float sh = 0.0f, sk = 0.0f, sv = 0.0f;
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    sh += hi[c] * hi[c];
    sk += key[c] * key[c];
    sv += value[c] * value[c];
  }
  sh = BlockSum(sh, scratch);
  sk = BlockSum(sk, scratch);
  sv = BlockSum(sv, scratch);
  const float rh = rsqrtf(sh / (float)dim + eps);
  const float rk = rsqrtf(sk / (float)dim + eps);
  const float rv = rsqrtf(sv / (float)dim + eps);
  float dot = 0.0f;
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    dot += (hi[c] * rh) * (key[c] * rk) * gate_w[(uint64_t)i * dim + c];
  }
  dot = BlockSum(dot, scratch) / sqrtf((float)dim);
  const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
  const float g = Sigmoid(dot < 0.0f ? -mag : mag);
  const float on = any[t] > 0.0f ? 1.0f : 0.0f;
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    const float signal = value[c] * rv * gamma[c];
    hi[c] += on * signal * (g + hc_scale[(uint64_t)i * dim + c]);
  }
}

// ------------------------------------------------------------------ kpool

// One block per pooled group; dim <= blockDim.x.
__global__ void KpoolPool(const float* rows, uint32_t ring, const int32_t* groups, uint32_t kpool, const float* ape,
                          const float* norm_w, float eps, float* pooled_out, uint32_t pool_ring, uint32_t dim) {
  __shared__ float scratch[32];
  const int32_t g = groups[blockIdx.x];
  const uint32_t c = threadIdx.x;
  float pooled = 0.0f;
  if (c < dim) {
    float mx = -INFINITY;
    for (uint32_t j = 0; j < kpool; ++j) {
      const float* r = rows + (uint64_t)(((uint32_t)g * kpool + j) % ring) * 2 * dim;
      mx = fmaxf(mx, r[dim + c] + ape[(uint64_t)j * dim + c]);
    }
    float den = 0.0f;
    for (uint32_t j = 0; j < kpool; ++j) {
      const float* r = rows + (uint64_t)(((uint32_t)g * kpool + j) % ring) * 2 * dim;
      const float e = expf(r[dim + c] + ape[(uint64_t)j * dim + c] - mx);
      den += e;
      pooled += e * r[c];
    }
    pooled /= den;
  }
  const float ss = BlockSum(c < dim ? pooled * pooled : 0.0f, scratch);
  const float scale = rsqrtf(ss / (float)dim + eps);
  if (c < dim) pooled_out[(uint64_t)((uint32_t)g % pool_ring) * dim + c] = pooled * scale * norm_w[c];
}

// One warp per (token, group).
__global__ void IndexerScore(const float* q, const float* w, const float* pooled, const int32_t* n_pool,
                             uint32_t n_pool_max, float* score, uint32_t heads, uint32_t dim) {
  const uint32_t t = blockIdx.y;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t g = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (g >= n_pool_max) return;
  float total = -INFINITY;
  if ((int32_t)g < n_pool[t]) {
    total = 0.0f;
    const float* k = pooled + (uint64_t)g * dim;
    for (uint32_t h = 0; h < heads; ++h) {
      const float* qh = q + ((uint64_t)t * heads + h) * dim;
      float d = 0.0f;
      for (uint32_t i = lane; i < dim; i += warpSize) d += qh[i] * k[i];
      d = WarpSum(d);
      total += fmaxf(d, 0.0f) * w[(uint64_t)t * heads + h];
    }
  }
  if (lane == 0) score[(uint64_t)t * n_pool_max + g] = total;
}

// One block per token: k rounds of block argmax over the visible groups.
// TODO(perf): a radix select instead of k argmax rounds.
__global__ void KpoolSelect(float* score, const int32_t* n_pool, uint32_t n_pool_max, const int32_t* pos,
                            uint32_t k, uint32_t kpool, int32_t* sel, uint32_t n_sel) {
  __shared__ float scratch[32];
  __shared__ int32_t picked;
  const uint32_t t = blockIdx.x;
  float* row = score + (uint64_t)t * n_pool_max;
  int32_t* out = sel + (uint64_t)t * n_sel;
  const uint32_t visible = (uint32_t)n_pool[t];
  const uint32_t take = visible < k ? visible : k;
  for (uint32_t i = threadIdx.x; i < n_sel; i += blockDim.x) out[i] = -1;
  __syncthreads();
  for (uint32_t s = 0; s < take; ++s) {
    float best = -INFINITY;
    for (uint32_t g = threadIdx.x; g < visible; g += blockDim.x) best = fmaxf(best, row[g]);
    best = BlockMax(best, scratch);
    if (threadIdx.x == 0) picked = (int32_t)n_pool_max;
    __syncthreads();
    for (uint32_t g = threadIdx.x; g < visible; g += blockDim.x) {
      if (row[g] == best) atomicMin(&picked, (int32_t)g);
    }
    __syncthreads();
    const int32_t g = picked;
    if (threadIdx.x < kpool) out[s * kpool + threadIdx.x] = g * (int32_t)kpool + (int32_t)threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) row[g] = -INFINITY;
    __syncthreads();
  }
  const int32_t p = pos[t];
  const int32_t first = p - p % (int32_t)kpool;
  if (threadIdx.x < kpool) {
    const int32_t at = first + (int32_t)threadIdx.x;
    out[k * kpool + threadIdx.x] = at <= p ? at : -1;
  }
}

__global__ void Argmax(const float* x, uint32_t dim, int32_t* out) {
  __shared__ float scratch[32];
  __shared__ int32_t picked;
  const float* row = x + (uint64_t)blockIdx.x * dim;
  float best = -INFINITY;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) best = fmaxf(best, row[i]);
  best = BlockMax(best, scratch);
  if (threadIdx.x == 0) picked = (int32_t)dim;
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    if (row[i] == best) atomicMin(&picked, (int32_t)i);
  }
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = picked == (int32_t)dim ? 0 : picked;
}

uint32_t Blocks(uint64_t count, uint32_t cap = 4096) {
  const uint64_t b = (count + kBlock - 1) / kBlock;
  return (uint32_t)(b < 1 ? 1 : (b > cap ? cap : b));
}

}  // namespace

extern "C" {

int hy4_embed_q8_0(const void* table, const int32_t* tokens, uint32_t n, uint32_t dim, float* out,
                   hipStream_t stream) {
  if (n == 0) return 0;
  if (dim % 32) return 1;
  EmbedQ8_0<<<n, kBlock, 0, stream>>>((const BlockQ8_0*)table, tokens, dim, out);
  return Status();
}

int hy4_rms_norm(const float* x, const float* weight, float* out, uint32_t rows, uint32_t dim, uint32_t stride,
                 float eps, hipStream_t stream) {
  if (rows == 0) return 0;
  RmsNorm<<<rows, kBlock, 0, stream>>>(x, weight, out, dim, stride, eps);
  return Status();
}

int hy4_hc_pre(const float* h, const float* fn, const float* scale, const float* base, float* out, float* post,
               uint32_t n, uint32_t hc, uint32_t dim, float eps, float hc_eps, float magnitude, hipStream_t stream) {
  if (n == 0) return 0;
  if (hc > 8) return 1;
  HcPre<<<n, kBlock, 0, stream>>>(h, fn, scale, base, out, post, hc, dim, eps, hc_eps, magnitude);
  return Status();
}

int hy4_hc_post(const float* y, const float* residual, const float* post, float* h, uint32_t n, uint32_t hc,
                uint32_t dim, hipStream_t stream) {
  if (n == 0) return 0;
  HcPost<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(y, residual, post, h, hc, dim);
  return Status();
}

int hy4_hc_init(const float* x, float* h, uint32_t n, uint32_t hc, uint32_t dim, hipStream_t stream) {
  if (n == 0) return 0;
  HcInit<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(x, h, hc, dim);
  return Status();
}

int hy4_gemv_float(int type, const void* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t n,
                   hipStream_t stream) {
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    if (type == HY4_TYPE_F32) {
      GemvFloat<float><<<m, kBlock, 0, stream>>>((const float*)w, x + (uint64_t)t0 * k, y + (uint64_t)t0 * m, m,
                                                 k, nt);
    } else if (type == HY4_TYPE_F16) {
      GemvFloat<__half><<<m, kBlock, 0, stream>>>((const __half*)w, x + (uint64_t)t0 * k, y + (uint64_t)t0 * m,
                                                  m, k, nt);
    } else {
      return 1;
    }
  }
  return Status();
}

int hy4_gemv_q8_0_grouped(const void* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t groups,
                          uint32_t n, uint64_t x_tok, uint64_t x_grp, uint64_t y_tok, uint64_t y_grp,
                          hipStream_t stream) {
  if (k % 32) return 1;
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    GemvQ8_0Grouped<<<dim3(m, groups), 128, 0, stream>>>((const BlockQ8_0*)w, x + t0 * x_tok, y + t0 * y_tok, m,
                                                         k, nt, x_tok, x_grp, y_tok, y_grp);
  }
  return Status();
}

int hy4_rope(float* x, const int32_t* pos, uint32_t n, uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
             uint32_t offset, uint32_t n_rot, float base, hipStream_t stream) {
  if (n == 0) return 0;
  const uint32_t work = heads * (n_rot / 2);
  Rope<<<dim3((work + kBlock - 1) / kBlock, n), kBlock, 0, stream>>>(x, pos, heads, tok_stride, head_stride,
                                                                     offset, n_rot, base);
  return Status();
}

int hy4_mla_attention(const float* q, const float* cache, uint32_t ring, const int32_t* pos, uint32_t window,
                      const int32_t* sel, uint32_t n_sel, const float* sinks, float scale, float* out, uint32_t n,
                      uint32_t heads, uint32_t dk, uint32_t dv, hipStream_t stream) {
  if (n == 0) return 0;
  const size_t shared = (dk + kMaxKeys) * sizeof(float) + kMaxKeys * sizeof(int32_t);
  MlaAttention<<<dim3(n, heads), kBlock, shared, stream>>>(q, cache, ring, pos, window, sel, n_sel, sinks, scale,
                                                           out, heads, dk, dv);
  return Status();
}

int hy4_store_rows(const float* src, uint64_t src_stride, float* dst, uint32_t ring, const int32_t* pos, uint32_t n,
                   uint32_t dim, hipStream_t stream) {
  if (n == 0) return 0;
  StoreRows<<<dim3(Blocks(dim, 16), n), kBlock, 0, stream>>>(src, src_stride, dst, ring, pos, dim);
  return Status();
}

int hy4_sigmoid_mul(float* x, const float* gate, uint64_t count, hipStream_t stream) {
  if (count == 0) return 0;
  SigmoidMul<<<Blocks(count), kBlock, 0, stream>>>(x, gate, count);
  return Status();
}

int hy4_swiglu(const float* gate, const float* up, const float* route_w, float* y, uint32_t rows, uint32_t dim,
               float limit, hipStream_t stream) {
  if (rows == 0) return 0;
  Swiglu<<<dim3(Blocks(dim, 64), rows), kBlock, 0, stream>>>(gate, up, route_w, y, dim, limit);
  return Status();
}

int hy4_route(const float* logits, const float* bias, int32_t* ids, float* weights, uint32_t n, uint32_t n_experts,
              uint32_t k, int norm, float scale, hipStream_t stream) {
  if (n == 0) return 0;
  if (k > kBlock) return 1;
  Route<<<n, kBlock, 2 * n_experts * sizeof(float), stream>>>(logits, bias, ids, weights, n_experts, k, norm,
                                                              scale);
  return Status();
}

int hy4_sum_slots(const float* y, float* out, uint32_t n, uint32_t k, uint32_t dim, hipStream_t stream) {
  if (n == 0) return 0;
  SumSlots<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(y, out, k, dim);
  return Status();
}

int hy4_add(const float* a, const float* b, float* out, uint64_t count, hipStream_t stream) {
  if (count == 0) return 0;
  Add<<<Blocks(count), kBlock, 0, stream>>>(a, b, out, count);
  return Status();
}

int hy4_engram_rows(const uint8_t* rows, const float* valid, const float* norm_w, float* out, uint32_t n,
                    uint32_t cols, uint32_t d, float eps, hipStream_t stream) {
  if (n == 0) return 0;
  if (d > 128) return 1;
  EngramRows<<<dim3(cols, n), 128, 0, stream>>>(rows, valid, norm_w, out, cols, d, eps);
  return Status();
}

int hy4_engram_apply(float* h, const float* kv, const float* gate_w, const float* hc_scale, const float* gamma,
                     const float* any, uint32_t n, uint32_t hc, uint32_t dim, float eps, hipStream_t stream) {
  if (n == 0) return 0;
  EngramApply<<<dim3(hc, n), kBlock, 0, stream>>>(h, kv, gate_w, hc_scale, gamma, any, hc, dim, eps);
  return Status();
}

int hy4_add_bias_rows(float* x, const float* bias, uint32_t n, uint32_t rows_per_token, uint32_t dim,
                      hipStream_t stream) {
  if (n == 0) return 0;
  AddBiasRows<<<dim3(Blocks((uint64_t)rows_per_token * dim, 64), n), kBlock, 0, stream>>>(x, bias, rows_per_token,
                                                                                          dim);
  return Status();
}

int hy4_kpool_pool(const float* rows, uint32_t ring, const int32_t* groups, uint32_t n_groups, uint32_t kpool,
                   const float* ape, const float* norm_w, float eps, float* pooled_out, uint32_t pool_ring,
                   uint32_t dim, hipStream_t stream) {
  if (n_groups == 0) return 0;
  if (dim > kBlock) return 1;
  KpoolPool<<<n_groups, dim < 64 ? 64 : dim, 0, stream>>>(rows, ring, groups, kpool, ape, norm_w, eps, pooled_out,
                                                          pool_ring, dim);
  return Status();
}

int hy4_indexer_score(const float* q, const float* w, const float* pooled, const int32_t* n_pool,
                      uint32_t n_pool_max, float* score, uint32_t n, uint32_t heads, uint32_t dim,
                      hipStream_t stream) {
  if (n == 0 || n_pool_max == 0) return 0;
  int dev = 0;
  int ws = 32;
  (void)hipGetDevice(&dev);
  (void)hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, dev);
  const uint32_t per_block = kBlock / (uint32_t)ws;
  IndexerScore<<<dim3((n_pool_max + per_block - 1) / per_block, n), kBlock, 0, stream>>>(q, w, pooled, n_pool,
                                                                                         n_pool_max, score, heads,
                                                                                         dim);
  return Status();
}

int hy4_kpool_select(const float* score, const int32_t* n_pool, uint32_t n_pool_max, const int32_t* pos, uint32_t k,
                     uint32_t kpool, int32_t* sel, uint32_t n, hipStream_t stream) {
  if (n == 0) return 0;
  KpoolSelect<<<n, kBlock, 0, stream>>>((float*)score, n_pool, n_pool_max, pos, k, kpool, sel, (k + 1) * kpool);
  return Status();
}

int hy4_argmax(const float* x, uint32_t n, uint32_t dim, int32_t* out, hipStream_t stream) {
  if (n == 0) return 0;
  Argmax<<<n, kBlock, 0, stream>>>(x, dim, out);
  return Status();
}

}  // extern "C"
