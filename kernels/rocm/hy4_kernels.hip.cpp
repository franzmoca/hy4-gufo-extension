// Hy4 Flash kernels outside the vendored llama.cpp matvecs (hy4_kernels.h).
// Written for correctness first: one block per row or token, FP32
// arithmetic, shared-memory reductions.

#include "hy4_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <climits>

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

// Hyper-connection mixes in two passes, so a single decoded token still
// spreads over the GPU. The mixes are rs * sum(fn h) with rs the RMS scale of
// h, so each block sums a slice of fn h and of h^2; HcApply reduces the
// slices in a fixed order (deterministic) and mixes the streams.
constexpr uint32_t kHcSlices = 48;
constexpr uint32_t kHcMaxMix = 16;

// Grid (kHcSlices, n). partial[t][slice][0..n_mix) dots, [n_mix] sum of squares.
__global__ void HcMix(const float* h, const float* fn, float* partial, uint32_t flat, uint32_t n_mix) {
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t per = (flat + kHcSlices - 1) / kHcSlices;
  const uint32_t lo = blockIdx.x * per;
  const uint32_t hi = min(flat, lo + per);
  const float* ht = h + (uint64_t)t * flat;
  float mix[kHcMaxMix];
  for (uint32_t m = 0; m < n_mix; ++m) mix[m] = 0.0f;
  float ss = 0.0f;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float v = ht[i];
    ss += v * v;
    for (uint32_t m = 0; m < n_mix; ++m) mix[m] += fn[(uint64_t)m * flat + i] * v;
  }
  float* out = partial + ((uint64_t)t * kHcSlices + blockIdx.x) * (kHcMaxMix + 1);
  for (uint32_t m = 0; m < n_mix; ++m) {
    const float r = BlockSum(mix[m], scratch);
    if (threadIdx.x == 0) out[m] = r;
  }
  const float r = BlockSum(ss, scratch);
  if (threadIdx.x == 0) out[n_mix] = r;
}

// HcMix for T tokens per block with NM mixes: each weight value loaded once
// serves T tokens. Per token the same thread partition, products and
// reduction order as HcMix (warp sums, then the warps' partials in order), so
// the results are identical. Grid (kHcSlices, token groups).
template<uint32_t T, uint32_t NM>
__global__ void __launch_bounds__(kBlock)
    HcMixTokens(const float* h, const float* fn, float* partial, uint32_t flat,
                uint32_t n) {
  constexpr uint32_t kValues = T * (NM + 1);
  __shared__ float scratch[kValues][kBlock / 32];
  const uint32_t t0 = blockIdx.y * T;
  const uint32_t nt = min(T, n - t0);
  const uint32_t per = (flat + kHcSlices - 1) / kHcSlices;
  const uint32_t lo = blockIdx.x * per;
  const uint32_t hi = min(flat, lo + per);
  float mix[T][NM + 1] = {};  // [NM] is the sum of squares
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float f[NM];
#pragma unroll
    for (uint32_t m = 0; m < NM; ++m)
      f[m] = fn[(uint64_t)m * flat + i];
#pragma unroll
    for (uint32_t tt = 0; tt < T; ++tt) {
      if (tt >= nt)
        break;
      const float v = h[(uint64_t)(t0 + tt) * flat + i];
      mix[tt][NM] += v * v;
#pragma unroll
      for (uint32_t m = 0; m < NM; ++m)
        mix[tt][m] += f[m] * v;
    }
  }
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t warps = (blockDim.x + warpSize - 1) / warpSize;
#pragma unroll
  for (uint32_t tt = 0; tt < T; ++tt)
#pragma unroll
    for (uint32_t m = 0; m <= NM; ++m) {
      const float v = WarpSum(mix[tt][m]);
      if (lane == 0)
        scratch[tt * (NM + 1) + m][warp] = v;
    }
  __syncthreads();
  for (uint32_t j = threadIdx.x; j < kValues; j += blockDim.x) {
    const uint32_t tt = j / (NM + 1), m = j % (NM + 1);
    if (tt >= nt)
      continue;
    float total = 0.0f;
    for (uint32_t w = 0; w < warps; ++w)
      total += scratch[j][w];
    partial[((uint64_t)(t0 + tt) * kHcSlices + blockIdx.x) * (kHcMaxMix + 1) +
            m] = total;
  }
}

// Grid (column blocks, n).
__global__ void HcApply(const float* h, const float* partial, const float* scale, const float* base, float* out,
                        float* post, uint32_t hc, uint32_t dim, float eps, float hc_eps, float magnitude) {
  __shared__ float mix_s[kHcMaxMix + 1];
  __shared__ float pre_s[8];
  const uint32_t t = blockIdx.y;
  const uint32_t n_mix = post ? 2 * hc : hc;
  const float* part = partial + (uint64_t)t * kHcSlices * (kHcMaxMix + 1);
  if (threadIdx.x <= n_mix) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < kHcSlices; ++b) acc += part[b * (kHcMaxMix + 1) + threadIdx.x];
    mix_s[threadIdx.x] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float rs = rsqrtf(mix_s[n_mix] / (float)(hc * dim) + eps);
    for (uint32_t i = 0; i < hc; ++i) {
      pre_s[i] = Sigmoid(mix_s[i] * rs * scale[0] + base[i]) + hc_eps;
      if (post && blockIdx.x == 0) {
        post[(uint64_t)t * hc + i] = Sigmoid(mix_s[hc + i] * rs * scale[1] + base[hc + i]) * magnitude + hc_eps;
      }
    }
  }
  __syncthreads();
  const float* ht = h + (uint64_t)t * hc * dim;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
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

// Eight consecutive weights as floats, from one 16-byte (F16) or two 16-byte
// (F32) loads.
template <typename W>
__device__ __forceinline__ void Load8(const W* w, float* out);
template <>
__device__ __forceinline__ void Load8<float>(const float* w, float* out) {
  const float4 a = *(const float4*)w;
  const float4 b = *(const float4*)(w + 4);
  out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
  out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}
template <>
__device__ __forceinline__ void Load8<__half>(const __half* w, float* out) {
  const uint4 raw = *(const uint4*)w;
  const __half2* h2 = (const __half2*)&raw;
  for (int i = 0; i < 4; ++i) {
    const float2 f = __half22float2(h2[i]);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}


// Warp per output row, all tokens (n <= kMaxTok) per weight read; k % 8 == 0
// and rows 16-byte aligned.
template <typename W>
__global__ void GemvFloat(const W* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t n) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (r >= m) return;
  const W* wr = w + (uint64_t)r * k;
  float acc[kMaxTok];
  for (uint32_t t = 0; t < kMaxTok; ++t) acc[t] = 0.0f;
  for (uint32_t i = lane * 8; i < k; i += warpSize * 8) {
    float wv[8];
    Load8<W>(wr + i, wv);
    for (uint32_t t = 0; t < n; ++t) {
      const float4 xa = *(const float4*)(x + (uint64_t)t * k + i);
      const float4 xb = *(const float4*)(x + (uint64_t)t * k + i + 4);
      acc[t] += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w + wv[4] * xb.x + wv[5] * xb.y +
                wv[6] * xb.z + wv[7] * xb.w;
    }
  }
  for (uint32_t t = 0; t < n; ++t) {
    const float s = WarpSum(acc[t]);
    if (lane == 0) y[(uint64_t)t * m + r] = s;
  }
}

// Warp per (row, group): each lane reads 4 quants (two 16-bit loads, the
// quants sit at an even offset) so 8 lanes cover a Q8_0 block.
__global__ void GemvQ8_0Grouped(const BlockQ8_0* w, const float* x, float* y, uint32_t m, uint32_t k,
                                uint32_t n, uint64_t x_tok, uint64_t x_grp, uint64_t y_tok, uint64_t y_grp) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t r = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const uint32_t g = blockIdx.y;
  if (r >= m) return;
  const uint32_t nb = k / 32;
  const BlockQ8_0* wr = w + ((uint64_t)g * m + r) * nb;
  float acc[kMaxTok];
  for (uint32_t t = 0; t < kMaxTok; ++t) acc[t] = 0.0f;
  const uint32_t sub = lane % 8;          // 4-quant slice within a block
  for (uint32_t b = lane / 8; b < nb; b += warpSize / 8) {
    const BlockQ8_0& blk = wr[b];
    const float d = __half2float(blk.d);
    const uint16_t* q16 = (const uint16_t*)blk.qs + sub * 2;
    const uint16_t lo = q16[0];
    const uint16_t hi = q16[1];
    const float w0 = d * (float)(int8_t)(lo & 0xff);
    const float w1 = d * (float)(int8_t)(lo >> 8);
    const float w2 = d * (float)(int8_t)(hi & 0xff);
    const float w3 = d * (float)(int8_t)(hi >> 8);
    const uint32_t i = b * 32 + sub * 4;
    for (uint32_t t = 0; t < n; ++t) {
      const float* xt = x + t * x_tok + g * x_grp + i;
      acc[t] += w0 * xt[0] + w1 * xt[1] + w2 * xt[2] + w3 * xt[3];
    }
  }
  for (uint32_t t = 0; t < n; ++t) {
    const float s = WarpSum(acc[t]);
    if (lane == 0) y[t * y_tok + g * y_grp + r] = s;
  }
}

// --------------------------------------------------------------------- rope

// Interleaved M-RoPE (mrope_section [12, 10, 10] of 32 pairs): pair i takes
// the height position when i % 3 == 1 and the width one when i % 3 == 2, for
// i < 30, else the temporal one. Without `pos_hw` every pair takes `pos`.
__global__ void Rope(float* x, const int32_t* pos, const int32_t* pos_hw,
                     uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
                     uint32_t offset, uint32_t n_rot, float base) {
  const uint32_t t = blockIdx.y;
  const uint32_t pairs = n_rot / 2;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * pairs) return;
  const uint32_t h = idx / pairs;
  const uint32_t i = idx % pairs;
  float* row = x + t * tok_stride + h * head_stride + offset;
  int32_t p = pos[t];
  if (pos_hw != nullptr && i < 30 && i % 3 != 0)
    p = pos_hw[2 * t + (i % 3 - 1)];
  const float theta = (float)p * powf(base, -2.0f * (float)i / (float)n_rot);
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

// One block per row (t, h).
__global__ void AttnSoftmax(float* scores, uint32_t heads, uint32_t n_keys, int32_t key_pos0, const int32_t* pos,
                            uint32_t window, const float* mask, const float* sinks) {
  __shared__ float scratch[32];
  const uint32_t r = blockIdx.x;
  const uint32_t t = r / heads;
  const uint32_t h = r % heads;
  float* row = scores + (uint64_t)r * n_keys;
  const float* mrow = mask ? mask + (uint64_t)t * n_keys : nullptr;
  const int32_t p = pos[t];
  const int32_t lo = window ? p - (int32_t)window + 1 : INT32_MIN;
  auto visible = [&](uint32_t j) {
    const int32_t kp = key_pos0 + (int32_t)j;
    return kp <= p && kp >= lo && (!mrow || mrow[j] == 0.0f);
  };
  float mx = sinks ? sinks[h] : -INFINITY;
  for (uint32_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    if (visible(j)) mx = fmaxf(mx, row[j]);
  }
  mx = BlockMax(mx, scratch);
  float den = 0.0f;
  for (uint32_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    const float e = visible(j) ? expf(row[j] - mx) : 0.0f;
    row[j] = e;
    den += e;
  }
  den = BlockSum(den, scratch);
  if (sinks) den += expf(sinks[h] - mx);
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  for (uint32_t j = threadIdx.x; j < n_keys; j += blockDim.x) row[j] *= inv;
}

__global__ void SelectionFill(float* mask, uint32_t n_keys) {
  const uint32_t t = blockIdx.y;
  float* row = mask + (uint64_t)t * n_keys;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n_keys; j += gridDim.x * blockDim.x) row[j] = -INFINITY;
}

__global__ void SelectionMaskSet(const int32_t* sel, uint32_t n_sel, float* mask, uint32_t n_keys) {
  const uint32_t t = blockIdx.y;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_sel; i += gridDim.x * blockDim.x) {
    const int32_t s = sel[(uint64_t)t * n_sel + i];
    if (s >= 0 && (uint32_t)s < n_keys) mask[(uint64_t)t * n_keys + s] = 0.0f;
  }
}

__global__ void GatherRows(const float* src, uint32_t ring, int32_t first, uint32_t dim, float* dst) {
  const uint32_t r = blockIdx.y;
  const uint64_t from = (uint64_t)((uint32_t)(first + (int32_t)r) % ring) * dim;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    dst[(uint64_t)r * dim + c] = src[from + c];
  }
}

__global__ void GatherIndexed(const float* src, uint32_t ring, const int32_t* idx, int32_t limit, uint32_t dim,
                              float* dst) {
  const uint32_t r = blockIdx.y;
  const int32_t p = idx[r];
  const bool ok = p >= 0 && p <= limit;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    dst[(uint64_t)r * dim + c] = ok ? src[(uint64_t)((uint32_t)p % ring) * dim + c] : 0.0f;
  }
}

__global__ void IndexMask(const int32_t* idx, uint32_t count, int32_t limit, float* mask, uint32_t n_keys) {
  const uint32_t t = blockIdx.y;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n_keys; j += gridDim.x * blockDim.x) {
    mask[(uint64_t)t * n_keys + j] = j < count && (idx[j] < 0 || idx[j] > limit) ? -INFINITY : 0.0f;
  }
}

__global__ void MulAdd(float* x, const float* a, const float* b, uint64_t count, uint64_t period) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x) {
    x[i] += a[i] * b[i % period];
  }
}

__global__ void DequantQ8_0(const BlockQ8_0* src, uint64_t blocks, float* out) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < blocks * 32;
       i += (uint64_t)gridDim.x * blockDim.x) {
    const BlockQ8_0& b = src[i / 32];
    out[i] = __half2float(b.d) * (float)b.qs[i % 32];
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

// ------------------------------------------------- fused prefill attention

// MLA prefill attention on the matrix cores, each token over exactly its own
// keys, so its result does not depend on the pass it is evaluated in.
// Block (token, 32 heads), eight waves in two groups of four, one group per
// 16 heads, so each staged key tile serves 32 heads. Keys come in 32-row
// tiles staged in LDS as binary16 [key][576]. Scores: wave w of a group
// multiplies its heads' Q (binary16 in registers) by K over dimensions
// 144 w .. 144 w + 143; the four partials meet in LDS. Thread h < 32 runs head
// h's FP32 online softmax, the sink seeding its maximum and denominator, and
// writes binary16 probabilities. Values: wave w of a group accumulates output
// dimensions 128 w .. 128 w + 127 in FP32.
// Fragment layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16
// contiguous k, B column L%16 and 16 contiguous k; accumulator element i is
// row 2i + L/16, column L%16.
constexpr uint32_t kMlaDk = 576;
constexpr uint32_t kMlaDv = 512;
constexpr uint32_t kMlaHeads = 32;  // two groups of 16, one per 4 waves
constexpr uint32_t kMlaWaves = 8;
constexpr uint32_t kMlaSplit = 4;  // waves per head group
constexpr uint32_t kMlaTile = 32;
constexpr uint32_t kMlaKtPitch = kMlaDk + 8;      // halves; 16-byte rows
constexpr uint32_t kMlaProbPitch = kMlaTile + 8;  // halves
constexpr uint32_t kMlaQkSteps = kMlaDk / kMlaSplit / 16;  // 9
constexpr uint32_t kMlaPvTiles = kMlaDv / kMlaSplit / 16;  // 8

using MlaHalf16 =
    __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using MlaFloat8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ MlaFloat8 MlaWmma(MlaHalf16 a, MlaHalf16 b,
                                             MlaFloat8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ MlaHalf16 MlaLoad16(const _Float16* p) {
  union {
    MlaHalf16 h;
    uint4 u[2];
  } v;
  v.u[0] = *reinterpret_cast<const uint4*>(p);
  v.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return v.h;
}

// The key position of slot j of token t, or -1 when the slot holds no visible
// key: a selection slot that is empty, in the future or outside the window.
__device__ __forceinline__ int32_t MlaKey(const int32_t* sel_t, int32_t first,
                                          uint32_t j, uint32_t count, int32_t p,
                                          int32_t lo_t) {
  if (j >= count)
    return -1;
  if (!sel_t)
    return first + (int32_t)j;
  const int32_t k = sel_t[j];
  return k >= lo_t && k <= p ? k : -1;
}

__global__ void __launch_bounds__(32 * kMlaWaves)
    MlaPrefill(const _Float16* q, const float* keys, int32_t lo,
               const int32_t* pos, uint32_t window, const int32_t* sel,
               uint32_t n_sel, const float* sinks, float scale, float* out,
               uint32_t heads) {
  __shared__ __align__(16) _Float16 kt[kMlaTile * kMlaKtPitch];
  __shared__ __align__(16) _Float16 probs[kMlaHeads * kMlaProbPitch];
  __shared__ float part[kMlaSplit][kMlaHeads][kMlaTile];
  __shared__ int32_t rows[kMlaTile];
  __shared__ float m_s[kMlaHeads], l_s[kMlaHeads], alpha_s[kMlaHeads];

  const uint32_t t = blockIdx.x;
  const uint32_t h0 = blockIdx.y * kMlaHeads;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t wave = tid / 32;
  const uint32_t row = lane % 16;
  const uint32_t half = lane / 16;
  const uint32_t hg = (wave / kMlaSplit) * 16;  // this wave's head group
  const uint32_t sub = wave % kMlaSplit;        // its slice of dimensions

  const int32_t p = pos[t];
  const int32_t lo_t = window ? max(lo, p - (int32_t)window + 1) : lo;
  const int32_t* sel_t = sel ? sel + (uint64_t)t * n_sel : nullptr;
  const uint32_t count = sel         ? n_sel
                         : p >= lo_t ? (uint32_t)(p - lo_t + 1)
                                     : 0u;

  // This wave's slice of the 16 heads' queries, binary16.
  MlaHalf16 qa[kMlaQkSteps];
  {
    const _Float16* qr = q + ((uint64_t)t * heads + h0 + hg + row) * kMlaDk +
                         sub * (kMlaDk / kMlaSplit);
#pragma unroll
    for (uint32_t s = 0; s < kMlaQkSteps; ++s)
      qa[s] = MlaLoad16(qr + s * 16);
  }
  if (tid < kMlaHeads) {
    m_s[tid] = sinks ? sinks[h0 + tid] : -INFINITY;
    l_s[tid] = sinks ? 1.0f : 0.0f;
  }
  MlaFloat8 acc[kMlaPvTiles] = {};

  for (uint32_t j0 = 0; j0 < count; j0 += kMlaTile) {
    if (tid < kMlaTile)
      rows[tid] = MlaKey(sel_t, lo_t, j0 + tid, count, p, lo_t);
    __syncthreads();
    // Stage the tile: each thread converts float4 groups of the 32 rows.
    for (uint32_t i = tid; i < kMlaTile * (kMlaDk / 4); i += blockDim.x) {
      const uint32_t r = i / (kMlaDk / 4);
      const uint32_t c = (i % (kMlaDk / 4)) * 4;
      float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      if (rows[r] >= 0)
        v = *reinterpret_cast<const float4*>(
            keys + (uint64_t)(rows[r] - lo) * kMlaDk + c);
      _Float16* d = kt + r * kMlaKtPitch + c;
      d[0] = (_Float16)v.x;
      d[1] = (_Float16)v.y;
      d[2] = (_Float16)v.z;
      d[3] = (_Float16)v.w;
    }
    __syncthreads();

    // Partial scores over this wave's dimensions.
#pragma unroll
    for (uint32_t b = 0; b < kMlaTile / 16; ++b) {
      MlaFloat8 sc = {};
      const _Float16* kr =
          kt + (b * 16 + row) * kMlaKtPitch + sub * (kMlaDk / kMlaSplit);
#pragma unroll
      for (uint32_t s = 0; s < kMlaQkSteps; ++s)
        sc = MlaWmma(qa[s], MlaLoad16(kr + s * 16), sc);
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i)
        part[sub][hg + 2 * i + half][b * 16 + row] = sc[i];
    }
    __syncthreads();

    // Online softmax, one thread per head.
    if (tid < kMlaHeads) {
      const uint32_t h = tid;
      float x[kMlaTile];
      float mx = m_s[h];
#pragma unroll
      for (uint32_t k = 0; k < kMlaTile; ++k) {
        float v = -INFINITY;
        if (rows[k] >= 0)
          v = (part[0][h][k] + part[1][h][k] + part[2][h][k] + part[3][h][k]) *
              scale;
        x[k] = v;
        mx = fmaxf(mx, v);
      }
      const float alpha = mx == -INFINITY ? 1.0f : __expf(m_s[h] - mx);
      float sum = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < kMlaTile; ++k) {
        const float e = x[k] == -INFINITY ? 0.0f : __expf(x[k] - mx);
        sum += e;
        probs[h * kMlaProbPitch + k] = (_Float16)e;
      }
      l_s[h] = l_s[h] * alpha + sum;
      m_s[h] = mx;
      alpha_s[h] = alpha;
    }
    __syncthreads();

    // Rescale and accumulate this wave's output dimensions.
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const float a = alpha_s[hg + 2 * i + half];
#pragma unroll
      for (uint32_t c = 0; c < kMlaPvTiles; ++c)
        acc[c][i] *= a;
    }
#pragma unroll
    for (uint32_t b = 0; b < kMlaTile / 16; ++b) {
      const MlaHalf16 pa =
          MlaLoad16(probs + (hg + row) * kMlaProbPitch + b * 16);
#pragma unroll
      for (uint32_t c = 0; c < kMlaPvTiles; ++c) {
        const uint32_t d = sub * (kMlaDv / kMlaSplit) + c * 16 + row;
        MlaHalf16 vb;
#pragma unroll
        for (uint32_t k = 0; k < 16; ++k)
          vb[k] = kt[(b * 16 + k) * kMlaKtPitch + d];
        acc[c] = MlaWmma(pa, vb, acc[c]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t h = hg + 2 * i + half;
    const float l = l_s[h];
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float* orow = out + ((uint64_t)t * heads + h0 + h) * kMlaDv +
                  sub * (kMlaDv / kMlaSplit) + row;
#pragma unroll
    for (uint32_t c = 0; c < kMlaPvTiles; ++c)
      orow[c * 16] = acc[c][i] * inv;
  }
}

// Splits Q8_0 blocks into their values as binary16 (exact for int8) and
// their scales as FP32.
__global__ void Q8_0Split(const uint8_t* src, uint64_t blocks, _Float16* q,
                          float* d) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks * 32)
    return;
  const uint8_t* blk = src + (i / 32) * 34;
  q[i] = (_Float16)(int8_t)blk[2 + i % 32];
  if (i % 32 == 0)
    d[i / 32] = (float)*reinterpret_cast<const _Float16*>(blk);
}

// MLA query absorption for prefill passes: out[t, h] = [wk_b[h] . q_nope,
// q_rope] in binary16, the operand layout MlaPrefill reads. wk_b is split by
// Q8_0Split. Block (head h, 64 tokens), sixteen waves; wave w owns outputs
// 32 w .. 32 w + 31 of the four 16-token tiles, so each weight fragment feeds
// four products. Heads vary fastest across the grid, so the blocks in flight
// read and write neighbouring rows (token-major order ran at half the
// bandwidth). q_nope is staged once in LDS as binary16, and the next step's
// weight fragments load during this step's products. A 16-wide k step lies
// inside one 32-value block, so each block's two steps accumulate apart and
// are scaled by the output row's block scale; every output sums in the same
// order whatever the token count. Results leave through LDS as whole rows.
constexpr uint32_t kAbsTok = 64;
constexpr uint32_t kAbsTiles = kAbsTok / 16;
constexpr uint32_t kAbsOut = 512;
constexpr uint32_t kAbsMaxK = 256;
constexpr uint32_t kAbsOutPitch = kAbsOut + 8;

__global__ void __launch_bounds__(512)
    MlaAbsorb(const _Float16* wq, const float* wd, const float* q,
              _Float16* out, uint32_t n, uint32_t heads, uint32_t nope,
              uint32_t n_rot) {
  __shared__ __align__(16) _Float16 xs[kAbsTok * (kAbsMaxK + 8)];
  __shared__ __align__(16) _Float16 os[16 * kAbsOutPitch];
  const uint32_t tid = threadIdx.x;
  const uint32_t pitch = nope + 8;
  const uint32_t lane = tid % 32;
  const uint32_t wave = tid / 32;
  const uint32_t row = lane % 16;
  const uint32_t half = lane / 16;
  const uint32_t g = blockIdx.x;
  const uint32_t t0 = blockIdx.y * kAbsTok;
  const uint32_t o0 = wave * 32;
  const uint32_t steps = nope / 16;
  const uint32_t head_k = nope + n_rot;
  const uint32_t row_out = kAbsOut + n_rot;

  // Stage q_nope: thread column c4 of 64 float4 columns, rows 8 apart.
  for (uint32_t c = tid % 64 * 4; c < nope; c += 256)
    for (uint32_t r = tid / 64; r < kAbsTok; r += 8) {
      const uint32_t t = min(t0 + r, n - 1);
      const float4 v = *reinterpret_cast<const float4*>(
          q + ((uint64_t)t * heads + g) * head_k + c);
      _Float16* d = xs + r * pitch + c;
      d[0] = (_Float16)v.x;
      d[1] = (_Float16)v.y;
      d[2] = (_Float16)v.z;
      d[3] = (_Float16)v.w;
    }

  const _Float16* wr[2];
  float dv[2][kAbsMaxK / 32];
#pragma unroll
  for (uint32_t c = 0; c < 2; ++c) {
    const uint64_t o = (uint64_t)g * kAbsOut + o0 + c * 16 + row;
    wr[c] = wq + o * nope;
#pragma unroll
    for (uint32_t b = 0; b < kAbsMaxK / 32; ++b)
      dv[c][b] = b < nope / 32 ? wd[o * (nope / 32) + b] : 0.0f;
  }
  MlaHalf16 bv[2] = {MlaLoad16(wr[0]), MlaLoad16(wr[1])};
  __syncthreads();

  MlaFloat8 acc[kAbsTiles][2] = {};
  MlaFloat8 part[kAbsTiles][2] = {};
  for (uint32_t s = 0; s < steps; ++s) {
    const uint32_t kk = s * 16;
    MlaHalf16 nb[2] = {bv[0], bv[1]};
    if (s + 1 < steps) {
      nb[0] = MlaLoad16(wr[0] + kk + 16);
      nb[1] = MlaLoad16(wr[1] + kk + 16);
    }
#pragma unroll
    for (uint32_t r = 0; r < kAbsTiles; ++r) {
      const MlaHalf16 a = MlaLoad16(xs + (r * 16 + row) * pitch + kk);
#pragma unroll
      for (uint32_t c = 0; c < 2; ++c)
        part[r][c] = MlaWmma(a, bv[c], part[r][c]);
    }
    if (s % 2 == 1) {
#pragma unroll
      for (uint32_t c = 0; c < 2; ++c) {
        float d = 0.0f;
#pragma unroll
        for (uint32_t b = 0; b < kAbsMaxK / 32; ++b)
          d = b == s / 2 ? dv[c][b] : d;
#pragma unroll
        for (uint32_t r = 0; r < kAbsTiles; ++r) {
#pragma unroll
          for (uint32_t i = 0; i < 8; ++i)
            acc[r][c][i] += d * part[r][c][i];
          part[r][c] = MlaFloat8{};
        }
      }
    }
    bv[0] = nb[0];
    bv[1] = nb[1];
  }

  // One 16-token tile at a time: fragments to LDS, then whole rows out, the
  // rope tail converted from q.
#pragma unroll
  for (uint32_t r = 0; r < kAbsTiles; ++r) {
#pragma unroll
    for (uint32_t c = 0; c < 2; ++c)
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i)
        os[(2 * i + half) * kAbsOutPitch + o0 + c * 16 + row] =
            (_Float16)acc[r][c][i];
    __syncthreads();
    for (uint32_t i = tid; i < 16 * (kAbsOut / 8); i += blockDim.x) {
      const uint32_t rr = i / (kAbsOut / 8), c = i % (kAbsOut / 8) * 8;
      const uint32_t t = t0 + r * 16 + rr;
      if (t < n)
        *reinterpret_cast<uint4*>(out + ((uint64_t)t * heads + g) * row_out +
                                  c) =
            *reinterpret_cast<const uint4*>(os + rr * kAbsOutPitch + c);
    }
    for (uint32_t i = tid; i < 16 * (n_rot / 4); i += blockDim.x) {
      const uint32_t rr = i / (n_rot / 4), c = i % (n_rot / 4) * 4;
      const uint32_t t = t0 + r * 16 + rr;
      if (t >= n)
        continue;
      const float4 v = *reinterpret_cast<const float4*>(
          q + ((uint64_t)t * heads + g) * head_k + nope + c);
      _Float16* d = out + ((uint64_t)t * heads + g) * row_out + kAbsOut + c;
      d[0] = (_Float16)v.x;
      d[1] = (_Float16)v.y;
      d[2] = (_Float16)v.z;
      d[3] = (_Float16)v.w;
    }
    __syncthreads();
  }
}

// Device warp size (32 on gfx1151), for grids of one warp per row.
uint32_t WarpSize() {
  static const uint32_t ws = [] {
    int dev = 0;
    int v = 32;
    (void)hipGetDevice(&dev);
    (void)hipDeviceGetAttribute(&v, hipDeviceAttributeWarpSize, dev);
    return (uint32_t)v;
  }();
  return ws;
}

// Rows per kBlock-thread block when each row takes one warp.
uint32_t RowsPerBlock() { return kBlock / WarpSize(); }

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
  static thread_local float* partial = nullptr;
  static thread_local uint32_t partial_tokens = 0;
  if (n > partial_tokens) {
    if (partial) (void)hipFree(partial);
    partial = nullptr;
    if (hipMalloc(&partial, (size_t)n * kHcSlices * (kHcMaxMix + 1) * sizeof(float)) != hipSuccess) return 1;
    partial_tokens = n;
  }
  const uint32_t n_mix = post ? 2 * hc : hc;
  constexpr uint32_t kTok = 8;
  const dim3 groups(kHcSlices, (n + kTok - 1) / kTok);
  if (n >= kTok && n_mix == 8)
    HcMixTokens<kTok, 8>
        <<<groups, kBlock, 0, stream>>>(h, fn, partial, hc * dim, n);
  else if (n >= kTok && n_mix == 4)
    HcMixTokens<kTok, 4>
        <<<groups, kBlock, 0, stream>>>(h, fn, partial, hc * dim, n);
  else
    HcMix<<<dim3(kHcSlices, n), kBlock, 0, stream>>>(h, fn, partial, hc * dim,
                                                     n_mix);
  HcApply<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(h, partial, scale, base, out, post, hc, dim, eps, hc_eps,
                                                         magnitude);
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
  if (k % 8) return 1;
  const uint32_t blocks = (m + RowsPerBlock() - 1) / RowsPerBlock();
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    if (type == HY4_TYPE_F32) {
      GemvFloat<float><<<blocks, kBlock, 0, stream>>>((const float*)w, x + (uint64_t)t0 * k,
                                                                  y + (uint64_t)t0 * m, m, k, nt);
    } else if (type == HY4_TYPE_F16) {
      GemvFloat<__half><<<blocks, kBlock, 0, stream>>>((const __half*)w, x + (uint64_t)t0 * k,
                                                                   y + (uint64_t)t0 * m, m, k, nt);
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
  const uint32_t blocks = (m + RowsPerBlock() - 1) / RowsPerBlock();
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    GemvQ8_0Grouped<<<dim3(blocks, groups), kBlock, 0, stream>>>(
        (const BlockQ8_0*)w, x + t0 * x_tok, y + t0 * y_tok, m, k, nt, x_tok, x_grp, y_tok, y_grp);
  }
  return Status();
}

int hy4_rope(float* x, const int32_t* pos, uint32_t n, uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
             uint32_t offset, uint32_t n_rot, float base, hipStream_t stream) {
  return hy4_rope_3d(x, pos, nullptr, n, heads, tok_stride, head_stride, offset,
                     n_rot, base, stream);
}

int hy4_rope_3d(float* x, const int32_t* pos, const int32_t* pos_hw, uint32_t n,
                uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
                uint32_t offset, uint32_t n_rot, float base,
                hipStream_t stream) {
  if (n == 0) return 0;
  if (pos_hw != nullptr && n_rot != 64)
    return 1;
  const uint32_t work = heads * (n_rot / 2);
  Rope<<<dim3((work + kBlock - 1) / kBlock, n), kBlock, 0, stream>>>(
      x, pos, pos_hw, heads, tok_stride, head_stride, offset, n_rot, base);
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
  const uint32_t per_block = RowsPerBlock();
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

int hy4_attn_softmax(float* scores, uint32_t n, uint32_t heads, uint32_t n_keys, int32_t key_pos0,
                     const int32_t* pos, uint32_t window, const float* mask, const float* sinks,
                     hipStream_t stream) {
  if (n == 0 || n_keys == 0) return 0;
  AttnSoftmax<<<n * heads, kBlock, 0, stream>>>(scores, heads, n_keys, key_pos0, pos, window, mask, sinks);
  return Status();
}

int hy4_selection_mask(const int32_t* sel, uint32_t n_sel, float* mask, uint32_t n, uint32_t n_keys,
                       hipStream_t stream) {
  if (n == 0) return 0;
  SelectionFill<<<dim3(Blocks(n_keys, 64), n), kBlock, 0, stream>>>(mask, n_keys);
  SelectionMaskSet<<<dim3(Blocks(n_sel, 16), n), kBlock, 0, stream>>>(sel, n_sel, mask, n_keys);
  return Status();
}

int hy4_gather_rows(const float* src, uint32_t ring, int32_t first, uint32_t count, uint32_t dim, float* dst,
                    hipStream_t stream) {
  if (count == 0) return 0;
  GatherRows<<<dim3(Blocks(dim, 16), count), kBlock, 0, stream>>>(src, ring, first, dim, dst);
  return Status();
}

int hy4_gather_indexed(const float* src, uint32_t ring, const int32_t* idx, uint32_t count, int32_t limit,
                       uint32_t dim, float* dst, float* mask, uint32_t rows, uint32_t n_keys, hipStream_t stream) {
  if (count) GatherIndexed<<<dim3(1, count), kBlock, 0, stream>>>(src, ring, idx, limit, dim, dst);
  if (rows && n_keys) {
    IndexMask<<<dim3((n_keys + kBlock - 1) / kBlock, rows), kBlock, 0, stream>>>(idx, count, limit, mask, n_keys);
  }
  return Status();
}

int hy4_mul_add(float* x, const float* a, const float* b, uint64_t count, uint64_t period, hipStream_t stream) {
  if (count == 0) return 0;
  MulAdd<<<Blocks(count), kBlock, 0, stream>>>(x, a, b, count, period);
  return Status();
}

int hy4_dequant_q8_0(const void* src, uint64_t blocks, float* out, hipStream_t stream) {
  if (blocks == 0) return 0;
  DequantQ8_0<<<Blocks(blocks * 32), kBlock, 0, stream>>>((const BlockQ8_0*)src, blocks, out);
  return Status();
}

int hy4_q8_0_split(const void* src, uint64_t blocks, uint16_t* q, float* d,
                   hipStream_t stream) {
  if (blocks == 0)
    return 0;
  const uint64_t count = blocks * 32;
  Q8_0Split<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
      static_cast<const uint8_t*>(src), blocks, reinterpret_cast<_Float16*>(q),
      d);
  return Status();
}

int hy4_mla_absorb(const uint16_t* wq, const float* wd, const float* q,
                   uint16_t* out, uint32_t n, uint32_t heads, uint32_t nope,
                   uint32_t n_rot, uint32_t kv_lora, hipStream_t stream) {
  if (kv_lora != kAbsOut || nope % 32 != 0 || nope == 0 || nope > kAbsMaxK ||
      n_rot % 8 != 0 || n_rot > 128)
    return 1;
  if (n == 0 || heads == 0)
    return 0;
  MlaAbsorb<<<dim3(heads, (n + kAbsTok - 1) / kAbsTok), 512, 0, stream>>>(
      reinterpret_cast<const _Float16*>(wq), wd, q,
      reinterpret_cast<_Float16*>(out), n, heads, nope, n_rot);
  return Status();
}

int hy4_mla_prefill(const uint16_t* q, const float* keys, int32_t lo,
                    const int32_t* pos, uint32_t window, const int32_t* sel,
                    uint32_t n_sel, const float* sinks, float scale, float* out,
                    uint32_t n, uint32_t heads, uint32_t dk, uint32_t dv,
                    hipStream_t stream) {
  if (n == 0)
    return 0;
  if (dk != kMlaDk || dv != kMlaDv || heads % kMlaHeads != 0)
    return 1;
  MlaPrefill<<<dim3(n, heads / kMlaHeads), 32 * kMlaWaves, 0, stream>>>(
      reinterpret_cast<const _Float16*>(q), keys, lo, pos, window, sel, n_sel,
      sinks, scale, out, heads);
  return Status();
}

int hy4_argmax(const float* x, uint32_t n, uint32_t dim, int32_t* out, hipStream_t stream) {
  if (n == 0) return 0;
  Argmax<<<n, kBlock, 0, stream>>>(x, dim, out);
  return Status();
}

}  // extern "C"
