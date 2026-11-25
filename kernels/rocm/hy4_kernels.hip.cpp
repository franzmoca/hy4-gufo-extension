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

using Half4 = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;

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

__global__ void GatherRows(const uint32_t* src, uint64_t words,
                           const int32_t* ids, uint32_t* dst) {
  const uint32_t* row = src + (uint64_t)ids[blockIdx.x] * words;
  uint32_t* out = dst + (uint64_t)blockIdx.x * words;
  for (uint64_t i = threadIdx.x; i < words; i += blockDim.x)
    out[i] = row[i];
}

__global__ void EmbedQ8_0(const BlockQ8_0* table, const int32_t* tokens, uint32_t dim, float* out) {
  const uint32_t t = blockIdx.x;
  const BlockQ8_0* row = table + (uint64_t)tokens[t] * (dim / 32);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const BlockQ8_0& b = row[i / 32];
    out[(uint64_t)t * dim + i] = __half2float(b.d) * (float)b.qs[i % 32];
  }
}

// --------------------------------------------------------------------- norm

// One row by one block: RmsNorm's body, shared by the fused decode kernels so
// that they compute its bits.
__device__ void RmsNormRow(const float* xr, const float* weight, float* yr,
                           uint32_t dim, float eps, float* scratch) {
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
    ss += xr[i] * xr[i];
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / (float)dim + eps);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xr[i] * scale;
    yr[i] = weight ? v * weight[i] : v;
  }
}

__global__ void RmsNorm(const float* x, const float* weight, float* out,
                        uint32_t dim, uint32_t stride, float eps,
                        _Float16* out_half = nullptr) {
  __shared__ float scratch[32];
  const float* xr = x + (uint64_t)blockIdx.x * stride;
  float* yr = out + (uint64_t)blockIdx.x * stride;
  if (!out_half) {
    RmsNormRow(xr, weight, yr, dim, eps, scratch);
    return;
  }
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
  ss = BlockSum(ss, scratch);
  const float scale = rsqrtf(ss / (float)dim + eps);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xr[i] * scale;
    float y = weight ? v * weight[i] : v;
    yr[i] = y;
    if (out_half) {
      // Round the product to FP32 first, as a consumer converting the FP32
      // output would (the compiler would otherwise fuse the multiply and the
      // conversion into one rounding).
      asm volatile("" : "+v"(y));
      out_half[(uint64_t)blockIdx.x * stride + i] = (_Float16)y;
    }
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
// the results are identical. With kPost the previous half's HcPost happens
// here: each h value becomes h + y * post (as HcPost computes it), is stored
// back and mixed, saving HcPost's pass over the streams. Grid (kHcSlices,
// token groups).
template<uint32_t T, uint32_t NM, bool kPost = false>
__global__ void __launch_bounds__(kBlock)
    HcMixTokens(float* h, const float* fn, float* partial, uint32_t flat,
                uint32_t n, const float* y = nullptr,
                const float* post = nullptr, uint32_t hc = 0,
                uint32_t dim = 0) {
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
      float v = h[(uint64_t)(t0 + tt) * flat + i];
      if constexpr (kPost) {
        const uint32_t t = t0 + tt, stream = i / dim;
        v = v + y[(uint64_t)t * dim + (i - stream * dim)] *
                    post[(uint64_t)t * hc + stream];
        h[(uint64_t)t * flat + i] = v;
      }
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

// Decode half-layer boundary in two launches (hy4_hc_decode). Each thread
// does what RmsNorm, HcPost, HcMix, HcApply and RmsNorm do, with their
// thread layouts and reduction orders, so the results are theirs bit for bit.
// HcPostMix, grid (kHcSlices, n): every block takes y's RMS scale (RmsNorm's
// loop and BlockSum), adds the normed y times post_prev into its slice of h,
// stores it and sums the slice's mixes (HcMix). y == NULL skips the post.
__global__ void __launch_bounds__(kBlock)
    HcPostMix(const float* y, const float* y_norm, const float* post_prev,
              float* h, const float* fn, float* partial, uint32_t hc,
              uint32_t dim, uint32_t n_mix, float eps) {
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t flat = hc * dim;
  float scale = 0.0f;
  const float* yt = y ? y + (uint64_t)t * dim : nullptr;
  if (yt) {
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
      ss += yt[i] * yt[i];
    ss = BlockSum(ss, scratch);
    scale = rsqrtf(ss / (float)dim + eps);
  }
  const uint32_t per = (flat + kHcSlices - 1) / kHcSlices;
  const uint32_t lo = blockIdx.x * per;
  const uint32_t hi = min(flat, lo + per);
  float* ht = h + (uint64_t)t * flat;
  float mix[kHcMaxMix];
  for (uint32_t m = 0; m < n_mix; ++m)
    mix[m] = 0.0f;
  float ss = 0.0f;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float v = ht[i];
    if (yt) {
      const uint32_t stream = i / dim, c = i - stream * dim;
      const float yv = yt[c] * scale;
      const float yn = y_norm ? yv * y_norm[c] : yv;
      v = v + yn * post_prev[(uint64_t)t * hc + stream];
      ht[i] = v;
    }
    ss += v * v;
    for (uint32_t m = 0; m < n_mix; ++m)
      mix[m] += fn[(uint64_t)m * flat + i] * v;
  }
  float* out =
      partial + ((uint64_t)t * kHcSlices + blockIdx.x) * (kHcMaxMix + 1);
  for (uint32_t m = 0; m < n_mix; ++m) {
    const float r = BlockSum(mix[m], scratch);
    if (threadIdx.x == 0)
      out[m] = r;
  }
  const float r = BlockSum(ss, scratch);
  if (threadIdx.x == 0)
    out[n_mix] = r;
}

// HcApplyNorm, grid (1, n): HcApply's mix for all of the token's columns in
// one block (each column sums its streams alone, so the layout does not
// change it), then RmsNorm of the mixed row into out (weight may be NULL). x
// keeps the mixed row.
__global__ void __launch_bounds__(kBlock)
    HcApplyNorm(const float* h, const float* partial, const float* scale,
                const float* base, float* x, float* out, const float* weight,
                float* post, uint32_t hc, uint32_t dim, float eps, float hc_eps,
                float magnitude) {
  __shared__ float mix_s[kHcMaxMix + 1];
  __shared__ float pre_s[8];
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t n_mix = post ? 2 * hc : hc;
  const float* part = partial + (uint64_t)t * kHcSlices * (kHcMaxMix + 1);
  if (threadIdx.x <= n_mix) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < kHcSlices; ++b)
      acc += part[b * (kHcMaxMix + 1) + threadIdx.x];
    mix_s[threadIdx.x] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float rs = rsqrtf(mix_s[n_mix] / (float)(hc * dim) + eps);
    for (uint32_t i = 0; i < hc; ++i) {
      pre_s[i] = Sigmoid(mix_s[i] * rs * scale[0] + base[i]) + hc_eps;
      if (post)
        post[(uint64_t)t * hc + i] =
            Sigmoid(mix_s[hc + i] * rs * scale[1] + base[hc + i]) * magnitude +
            hc_eps;
    }
  }
  __syncthreads();
  const float* ht = h + (uint64_t)t * hc * dim;
  float* xt = x + (uint64_t)t * dim;
  for (uint32_t c = threadIdx.x; c < dim; c += blockDim.x) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < hc; ++i)
      acc += ht[(uint64_t)i * dim + c] * pre_s[i];
    xt[c] = acc;
  }
  // RmsNorm reads the row back as written, in its own loop order.
  __syncthreads();
  float ss = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
    ss += xt[i] * xt[i];
  ss = BlockSum(ss, scratch);
  const float rscale = rsqrtf(ss / (float)dim + eps);
  float* yt = out + (uint64_t)t * dim;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = xt[i] * rscale;
    yt[i] = weight ? v * weight[i] : v;
  }
}

// HcPostMix and HcApplyNorm for NM mixes over HC streams, the model's
// shapes: the mixes stay in registers, their block sums share one barrier
// (each is BlockSum's warp sums, then the warps' partials in order), and
// HcApplyNorm keeps its columns (c = tid + kBlock r, kPer per thread) in
// registers for the norm. The same sums in the same order as the general
// kernels: their bits.
template<uint32_t NM>
__global__ void __launch_bounds__(kBlock)
    HcPostMixFast(const float* y, const float* y_norm, const float* post_prev,
                  float* h, const float* fn, float* partial, uint32_t hc,
                  uint32_t dim, float eps) {
  __shared__ float scratch[32];
  __shared__ float sums[NM + 1][kBlock / 32];
  const uint32_t t = blockIdx.y;
  const uint32_t flat = hc * dim;
  float scale = 0.0f;
  const float* yt = y ? y + (uint64_t)t * dim : nullptr;
  if (yt) {
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
      ss += yt[i] * yt[i];
    ss = BlockSum(ss, scratch);
    scale = rsqrtf(ss / (float)dim + eps);
  }
  const uint32_t per = (flat + kHcSlices - 1) / kHcSlices;
  const uint32_t lo = blockIdx.x * per;
  const uint32_t hi = min(flat, lo + per);
  float* ht = h + (uint64_t)t * flat;
  float mix[NM + 1] = {};  // [NM] is the sum of squares
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float f[NM];
#pragma unroll
    for (uint32_t m = 0; m < NM; ++m)
      f[m] = fn[(uint64_t)m * flat + i];
    float v = ht[i];
    if (yt) {
      const uint32_t stream = i / dim, c = i - stream * dim;
      const float yv = yt[c] * scale;
      const float yn = y_norm ? yv * y_norm[c] : yv;
      v = v + yn * post_prev[(uint64_t)t * hc + stream];
      ht[i] = v;
    }
    mix[NM] += v * v;
#pragma unroll
    for (uint32_t m = 0; m < NM; ++m)
      mix[m] += f[m] * v;
  }
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t warps = (blockDim.x + warpSize - 1) / warpSize;
#pragma unroll
  for (uint32_t m = 0; m <= NM; ++m) {
    const float v = WarpSum(mix[m]);
    if (lane == 0)
      sums[m][warp] = v;
  }
  __syncthreads();
  if (threadIdx.x <= NM) {
    float total = 0.0f;
    for (uint32_t w = 0; w < warps; ++w)
      total += sums[threadIdx.x][w];
    partial[((uint64_t)t * kHcSlices + blockIdx.x) * (kHcMaxMix + 1) +
            threadIdx.x] = total;
  }
}

template<uint32_t HC, uint32_t kPer>
__global__ void __launch_bounds__(kBlock)
    HcApplyNormFast(const float* h, const float* partial, const float* scale,
                    const float* base, float* x, float* out,
                    const float* weight, float* post, uint32_t dim, float eps,
                    float hc_eps, float magnitude) {
  __shared__ float mix_s[kHcMaxMix + 1];
  __shared__ float pre_s[HC];
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  const uint32_t n_mix = post ? 2 * HC : HC;
  const float* part = partial + (uint64_t)t * kHcSlices * (kHcMaxMix + 1);
  if (threadIdx.x <= n_mix) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < kHcSlices; ++b)
      acc += part[b * (kHcMaxMix + 1) + threadIdx.x];
    mix_s[threadIdx.x] = acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const float rs = rsqrtf(mix_s[n_mix] / (float)(HC * dim) + eps);
    for (uint32_t i = 0; i < HC; ++i) {
      pre_s[i] = Sigmoid(mix_s[i] * rs * scale[0] + base[i]) + hc_eps;
      if (post)
        post[(uint64_t)t * HC + i] =
            Sigmoid(mix_s[HC + i] * rs * scale[1] + base[HC + i]) * magnitude +
            hc_eps;
    }
  }
  __syncthreads();
  float pre[HC];
#pragma unroll
  for (uint32_t i = 0; i < HC; ++i)
    pre[i] = pre_s[i];
  const float* ht = h + (uint64_t)t * HC * dim;
  float* xt = x + (uint64_t)t * dim;
  float xv[kPer];
#pragma unroll
  for (uint32_t r = 0; r < kPer; ++r) {
    const uint32_t c = threadIdx.x + r * kBlock;
    float acc = 0.0f;
    if (c < dim) {
#pragma unroll
      for (uint32_t i = 0; i < HC; ++i)
        acc += ht[(uint64_t)i * dim + c] * pre[i];
      xt[c] = acc;
    }
    xv[r] = acc;
  }
  float ss = 0.0f;
#pragma unroll
  for (uint32_t r = 0; r < kPer; ++r)
    if (threadIdx.x + r * kBlock < dim)
      ss += xv[r] * xv[r];
  ss = BlockSum(ss, scratch);
  const float rscale = rsqrtf(ss / (float)dim + eps);
  float* yt = out + (uint64_t)t * dim;
#pragma unroll
  for (uint32_t r = 0; r < kPer; ++r) {
    const uint32_t c = threadIdx.x + r * kBlock;
    if (c < dim) {
      const float v = xv[r] * rscale;
      yt[c] = weight ? v * weight[c] : v;
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

// R output rows per warp, all tokens (n <= kMaxTok) per weight read; k % 8 ==
// 0 and rows 16-byte aligned. Each row sums as with R = 1; R > 1 shares each
// activation load among R rows, which multi-token passes of a large matrix
// (the LM head) are bound by.
template<typename W, uint32_t R>
__global__ void GemvFloat(const W* w, const float* x, float* y, uint32_t m,
                          uint32_t k, uint32_t n) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t r0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * R;
  if (r0 >= m)
    return;
  float acc[R][kMaxTok] = {};
  for (uint32_t i = lane * 8; i < k; i += warpSize * 8) {
    float wv[R][8];
#pragma unroll
    for (uint32_t j = 0; j < R; ++j)
      Load8<W>(w + (uint64_t)min(r0 + j, m - 1) * k + i, wv[j]);
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= n)
        break;
      const float4 xa = *(const float4*)(x + (uint64_t)t * k + i);
      const float4 xb = *(const float4*)(x + (uint64_t)t * k + i + 4);
#pragma unroll
      for (uint32_t j = 0; j < R; ++j)
        acc[j][t] += wv[j][0] * xa.x + wv[j][1] * xa.y + wv[j][2] * xa.z +
                     wv[j][3] * xa.w + wv[j][4] * xb.x + wv[j][5] * xb.y +
                     wv[j][6] * xb.z + wv[j][7] * xb.w;
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < R; ++j)
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= n)
        break;
      const float s = WarpSum(acc[j][t]);
      if (lane == 0 && r0 + j < m)
        y[(uint64_t)t * m + r0 + j] = s;
    }
}

// BF16 rows (the router), one row per block of kBf16Split waves: wave j sums
// the j-th of kBf16Split equal slices of k in order, 8 values per lane, then
// the slices add in order. Each token sums the same way at any n. A single
// wave per row left the router's 512 rows to 64 blocks on 40 CUs and read at
// about 130 GB/s; four waves per row reach about 180.
constexpr uint32_t kBf16Split = 4;
__device__ __forceinline__ void Load8Bf16(const uint16_t* w, float* out) {
  const uint4 raw = *(const uint4*)w;
  const uint32_t* u = (const uint32_t*)&raw;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = __uint_as_float(u[i] << 16);
    out[2 * i + 1] = __uint_as_float(u[i] & 0xffff0000u);
  }
}
__global__ void GemvBf16Split(const uint16_t* w, const float* x, float* y,
                              uint32_t m, uint32_t k, uint32_t n) {
  __shared__ float part[kBf16Split][kMaxTok];
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t wave = threadIdx.x / warpSize;
  const uint32_t r = blockIdx.x;
  const uint32_t len = k / kBf16Split, i0 = wave * len;
  const uint16_t* wr = w + (uint64_t)r * k;
  float acc[kMaxTok] = {};
  for (uint32_t i = i0 + lane * 8; i < i0 + len; i += warpSize * 8) {
    float wv[8];
    Load8Bf16(wr + i, wv);
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= n)
        break;
      const float4 xa = *(const float4*)(x + (uint64_t)t * k + i);
      const float4 xb = *(const float4*)(x + (uint64_t)t * k + i + 4);
      acc[t] += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w +
                wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
    }
  }
#pragma unroll
  for (uint32_t t = 0; t < kMaxTok; ++t) {
    if (t >= n)
      break;
    const float v = WarpSum(acc[t]);
    if (lane == 0)
      part[wave][t] = v;
  }
  __syncthreads();
  if (threadIdx.x < n) {
    float v = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < kBf16Split; ++j)
      v += part[j][threadIdx.x];
    y[(uint64_t)threadIdx.x * m + r] = v;
  }
}

__global__ void F32ToBf16(const float* src, uint64_t count, uint16_t* dst,
                          int32_t* inexact) {
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t bits = __float_as_uint(src[i]);
    if (bits & 0xffffu)
      *inexact = 1;
    dst[i] = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  }
}

// R rows per warp of group blockIdx.y: each lane reads 4 quants (two 16-bit
// loads, the quants sit at an even offset) so 8 lanes cover a Q8_0 block.
// Each row sums as with R = 1; R > 1 shares the activation loads.
template<uint32_t R>
__global__ void GemvQ8_0Grouped(const BlockQ8_0* w, const float* x, float* y,
                                uint32_t m, uint32_t k, uint32_t n,
                                uint64_t x_tok, uint64_t x_grp, uint64_t y_tok,
                                uint64_t y_grp) {
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t r0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * R;
  const uint32_t g = blockIdx.y;
  if (r0 >= m)
    return;
  const uint32_t nb = k / 32;
  float acc[R][kMaxTok] = {};
  const uint32_t sub = lane % 8;          // 4-quant slice within a block
  for (uint32_t b = lane / 8; b < nb; b += warpSize / 8) {
    float wq[R][4];
#pragma unroll
    for (uint32_t j = 0; j < R; ++j) {
      const BlockQ8_0& blk = w[((uint64_t)g * m + min(r0 + j, m - 1)) * nb + b];
      const float d = __half2float(blk.d);
      const uint16_t* q16 = (const uint16_t*)blk.qs + sub * 2;
      const uint16_t lo = q16[0];
      const uint16_t hi = q16[1];
      wq[j][0] = d * (float)(int8_t)(lo & 0xff);
      wq[j][1] = d * (float)(int8_t)(lo >> 8);
      wq[j][2] = d * (float)(int8_t)(hi & 0xff);
      wq[j][3] = d * (float)(int8_t)(hi >> 8);
    }
    const uint32_t i = b * 32 + sub * 4;
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= n)
        break;
      const float* xt = x + t * x_tok + g * x_grp + i;
      const float x0 = xt[0], x1 = xt[1], x2 = xt[2], x3 = xt[3];
#pragma unroll
      for (uint32_t j = 0; j < R; ++j)
        acc[j][t] +=
            wq[j][0] * x0 + wq[j][1] * x1 + wq[j][2] * x2 + wq[j][3] * x3;
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < R; ++j)
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= n)
        break;
      const float s = WarpSum(acc[j][t]);
      if (lane == 0 && r0 + j < m)
        y[t * y_tok + g * y_grp + r0 + j] = s;
    }
}

// Rows of whole Q8_0 blocks, k a multiple of 64 (so rows start 4-byte
// aligned): 8 lanes per row, lane l taking blocks l, l + 8, ... as nine
// aligned words around the block (its scale in the first word's low or high
// half), so a row is read in full 32-bit loads. A block's 32 rows share one
// group, whose nt input rows are staged in LDS. Each token sums alone: the
// same bits for any nt.
__device__ __forceinline__ void LoadQ8_0Words(const uint8_t* block, bool odd,
                                              uint32_t (&v)[9]) {
  const auto* words = reinterpret_cast<const uint32_t*>(block - (odd ? 2 : 0));
#pragma unroll
  for (int i = 0; i < 9; ++i)
    v[i] = words[i];
}

// LDS stride of a 32-value input block: 36 floats, so the 8 lanes of a row
// (blocks 0..7 of it) read 8 distinct 4-bank groups.
constexpr uint32_t kRowsXStride = 36;

__global__ void __launch_bounds__(kBlock)
    GemvQ8_0Rows(const uint8_t* w, const float* x, float* y, uint32_t m,
                 uint32_t k, uint32_t nt, uint64_t x_tok, uint64_t x_grp,
                 uint64_t y_tok, uint64_t y_grp) {
  extern __shared__ float xs[];  // [nt][k / 32][kRowsXStride]
  const uint32_t g = blockIdx.y;
  const uint32_t sub = threadIdx.x % 8;
  const uint32_t row = blockIdx.x * (kBlock / 8) + threadIdx.x / 8;
  const uint32_t nb = k / 32;
  const uint8_t* base = w + ((uint64_t)g * m + min(row, m - 1)) * nb * 34;
  // The first block's words are in flight while the inputs are staged.
  uint32_t v[9] = {};
  if (sub < nb)
    LoadQ8_0Words(base + 34 * sub, sub & 1, v);
  for (uint32_t i = threadIdx.x; i < nt * k; i += blockDim.x) {
    const uint32_t t = i / k, c = i - t * k;
    xs[(t * nb + c / 32) * kRowsXStride + c % 32] =
        x[t * x_tok + g * x_grp + c];
  }
  __syncthreads();
  float acc[kMaxTok] = {};
  for (uint32_t b = sub; b < nb; b += 8) {
    const bool odd = b & 1;  // the block starts 2 bytes into a word
    if (b != sub)
      LoadQ8_0Words(base + 34 * b, odd, v);
    const uint16_t dh = odd ? (uint16_t)(v[0] >> 16) : (uint16_t)v[0];
    const float d = __half2float(__ushort_as_half(dh));
    float q[32];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t word =
          odd ? v[i + 1] : __builtin_amdgcn_alignbit(v[i + 1], v[i], 16);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        q[4 * i + j] = (float)(int8_t)(word >> (8 * j));
    }
#pragma unroll
    for (uint32_t t = 0; t < kMaxTok; ++t) {
      if (t >= nt)
        break;
      const float4* xv =
          reinterpret_cast<const float4*>(xs + (t * nb + b) * kRowsXStride);
      float sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float4 xx = xv[i];
        sum += q[4 * i] * xx.x + q[4 * i + 1] * xx.y + q[4 * i + 2] * xx.z +
               q[4 * i + 3] * xx.w;
      }
      acc[t] += d * sum;
    }
  }
#pragma unroll
  for (uint32_t t = 0; t < kMaxTok; ++t) {
    if (t >= nt)
      break;
    float s = acc[t];
#pragma unroll
    for (int off = 4; off > 0; off >>= 1)
      s += __shfl_xor(s, off, 8);
    if (sub == 0 && row < m)
      y[t * y_tok + g * y_grp + row] = s;
  }
}

// --------------------------------------------------------------------- rope

// Interleaved M-RoPE (mrope_section [12, 10, 10] of 32 pairs): pair i takes
// the height position when i % 3 == 1 and the width one when i % 3 == 2, for
// i < 30, else the temporal one. Without `pos_hw` every pair takes `pos`.
// Rotates pair i of token t's row in place; dst, if given, gets a copy.
__device__ __forceinline__ void RopePair(float* row, float* dst,
                                         const int32_t* pos,
                                         const int32_t* pos_hw, uint32_t t,
                                         uint32_t i, uint32_t n_rot,
                                         float base) {
  int32_t p = pos[t];
  if (pos_hw != nullptr && i < 30 && i % 3 != 0)
    p = pos_hw[2 * t + (i % 3 - 1)];
  const float theta = (float)p * powf(base, -2.0f * (float)i / (float)n_rot);
  float s, c;
  sincosf(theta, &s, &c);
  const float x0 = row[2 * i];
  const float x1 = row[2 * i + 1];
  const float y0 = x0 * c - x1 * s;
  const float y1 = x0 * s + x1 * c;
  row[2 * i] = y0;
  row[2 * i + 1] = y1;
  if (dst) {
    dst[2 * i] = y0;
    dst[2 * i + 1] = y1;
  }
}

// With dst, each rotated pair is also stored at dst + t * dst_tok + h *
// dst_head (+ 2 i).
__global__ void Rope(float* x, const int32_t* pos, const int32_t* pos_hw,
                     uint32_t heads, uint64_t tok_stride, uint64_t head_stride,
                     uint32_t offset, uint32_t n_rot, float base,
                     float* dst = nullptr, uint64_t dst_tok = 0,
                     uint64_t dst_head = 0) {
  const uint32_t t = blockIdx.y;
  const uint32_t pairs = n_rot / 2;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * pairs) return;
  const uint32_t h = idx / pairs;
  const uint32_t i = idx % pairs;
  float* row = x + t * tok_stride + h * head_stride + offset;
  RopePair(row, dst ? dst + t * dst_tok + h * dst_head : nullptr, pos, pos_hw,
           t, i, n_rot, base);
}

// The decode MLA's normalisations after q_a and kv_a, grid (2, n): block 0
// norms the query latent (RmsNorm of q_lora values), block 1 the KV latent
// (the first kv_lora of dk values) in place and rotates the key's n_rot
// values after it (Rope, one head).
__global__ void __launch_bounds__(kBlock)
    MlaPrep(float* qr, const float* q_norm, uint32_t q_lora, float* kva,
            const float* kv_norm, uint32_t kv_lora, uint32_t dk,
            const int32_t* pos, const int32_t* pos_hw, uint32_t n_rot,
            float base, float eps) {
  __shared__ float scratch[32];
  const uint32_t t = blockIdx.y;
  if (blockIdx.x == 0) {
    float* row = qr + (uint64_t)t * q_lora;
    RmsNormRow(row, q_norm, row, q_lora, eps, scratch);
    return;
  }
  float* row = kva + (uint64_t)t * dk;
  RmsNormRow(row, kv_norm, row, kv_lora, eps, scratch);
  for (uint32_t i = threadIdx.x; i < n_rot / 2; i += blockDim.x)
    RopePair(row + kv_lora, nullptr, pos, pos_hw, t, i, n_rot, base);
}

// ---------------------------------------------------------------- attention

// Decode attention: each token over its own keys in splits of whole
// 32-key chunks, a warp holding kAttnHpw heads' queries in registers. Lane
// l owns latent columns 128 i + 4 l .. + 3 (i < 4) and rope columns
// 512 + 2 l, 2 l + 1. A split keeps a running max, denominator and value
// sum per head; MlaCombine merges the splits with the sink. A token's split
// size depends only on its own key count, so its result does not depend on
// the other tokens of the pass.
constexpr uint32_t kAttnDk = 576, kAttnDv = 512;
constexpr uint32_t kAttnHpw = 1, kAttnWarps = 8, kAttnUnroll = 4,
                   kAttnHeadsPerBlock = kAttnHpw * kAttnWarps;
constexpr uint32_t kAttnChunk = 32;

// Cap partial buffers to 512 head-splits per token (1 MiB/token): 8 splits at
// 64 heads (SWA, 32 blocks/token) and 16 at 32 heads (sparse, 32 blocks/token).
// Depends only on `heads` so each token sums identically for any batch size.
__host__ __device__ __forceinline__ uint32_t AttnSplits(uint32_t heads) {
  return heads > 32 ? 8u : 16u;
}

__device__ __forceinline__ uint32_t AttnSplitKeys(uint32_t count,
                                                  uint32_t max_splits) {
  const uint32_t per = kAttnChunk * max_splits;
  return kAttnChunk * ((count + per - 1) / per);
}

// The token's keys: list[t] when selected, else first .. first + count - 1.
__device__ __forceinline__ uint32_t AttnKeys(const int32_t* pos,
                                             uint32_t window,
                                             const uint32_t* list_count,
                                             uint32_t t, int32_t* first) {
  const int32_t p = pos[t];
  *first = window ? max(0, p - (int32_t)window + 1) : 0;
  return list_count ? list_count[t] : (uint32_t)(p - *first + 1);
}

// One block per token: the selected positions <= pos[t], ascending and
// without repeats, so that selecting every key gives the plain causal order.
// Positions are marked in an LDS bitmap kKeyTile at a time, then each
// thread writes its words' positions after a block scan of their counts.
constexpr uint32_t kKeyListThreads = 1024, kKeyTileWords = 2 * kKeyListThreads;
constexpr uint32_t kKeyTile = 32 * kKeyTileWords;

__global__ void __launch_bounds__(kKeyListThreads)
    AttnKeyList(const int32_t* sel, uint32_t n_sel, const int32_t* pos,
                int32_t* list, uint32_t* count) {
  __shared__ uint32_t bits[kKeyTileWords];
  __shared__ uint32_t warp_total[kKeyListThreads / 32];
  const uint32_t t = blockIdx.x;
  const int32_t p = pos[t];
  const int32_t* st = sel + (uint64_t)t * n_sel;
  int32_t* out = list + (uint64_t)t * n_sel;
  const uint32_t lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  uint32_t written = 0;
  for (int32_t tile = 0; tile <= p; tile += (int32_t)kKeyTile) {
    for (uint32_t w = threadIdx.x; w < kKeyTileWords; w += blockDim.x)
      bits[w] = 0;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n_sel; i += blockDim.x) {
      const int32_t s = st[i];
      if (s >= tile && s <= p && s - tile < (int32_t)kKeyTile)
        atomicOr(&bits[(uint32_t)(s - tile) / 32],
                 1u << ((uint32_t)(s - tile) % 32));
    }
    __syncthreads();
    // Thread x owns words 2x and 2x + 1.
    const uint32_t w0 = bits[2 * threadIdx.x], w1 = bits[2 * threadIdx.x + 1];
    const uint32_t mine = __popc(w0) + __popc(w1);
    uint32_t incl = mine;
    for (uint32_t o = 1; o < (uint32_t)warpSize; o <<= 1) {
      const uint32_t x = __shfl_up(incl, o);
      if (lane >= o)
        incl += x;
    }
    if (lane == (uint32_t)warpSize - 1)
      warp_total[warp] = incl;
    __syncthreads();
    uint32_t before = 0, total = 0;
    for (uint32_t w = 0; w < blockDim.x / warpSize; ++w) {
      if (w < warp)
        before += warp_total[w];
      total += warp_total[w];
    }
    uint32_t at = written + before + incl - mine;
    const int32_t base = tile + (int32_t)(64 * threadIdx.x);
    for (uint32_t v = w0; v; v &= v - 1)
      out[at++] = base + __ffs(v) - 1;
    for (uint32_t v = w1; v; v &= v - 1)
      out[at++] = base + 32 + __ffs(v) - 1;
    written += total;
    __syncthreads();
  }
  if (threadIdx.x == 0)
    count[t] = written;
}

__global__ void __launch_bounds__(kAttnWarps * 32)
    MlaDecode(const float* q, const _Float16* cache, uint32_t ring,
              const int32_t* pos, uint32_t window, const int32_t* list,
              const uint32_t* list_count, uint32_t list_stride, float scale,
              float* part_ml, float* part_acc, uint32_t heads) {
  const uint32_t split = blockIdx.x, t = blockIdx.z;
  const uint32_t max_splits = AttnSplits(heads);
  int32_t first;
  const uint32_t count = AttnKeys(pos, window, list_count, t, &first);
  const uint32_t keys = AttnSplitKeys(count, max_splits);
  const uint32_t begin = split * keys;
  if (begin >= count)
    return;
  const uint32_t end = min(count, begin + keys);
  const uint32_t lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  const uint32_t h0 = blockIdx.y * kAttnHeadsPerBlock + warp * kAttnHpw;
  if (h0 >= heads)
    return;

  float4 qv[kAttnHpw][4];
  float2 qt[kAttnHpw];
  for (uint32_t h = 0; h < kAttnHpw; ++h) {
    const float* qh = q + ((uint64_t)t * heads + h0 + h) * kAttnDk;
    for (uint32_t i = 0; i < 4; ++i)
      qv[h][i] = *(const float4*)(qh + 128 * i + 4 * lane);
    qt[h] = *(const float2*)(qh + kAttnDv + 2 * lane);
  }
  float m[kAttnHpw], l[kAttnHpw];
  float4 acc[kAttnHpw][4];
  for (uint32_t h = 0; h < kAttnHpw; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.0f;
    for (uint32_t i = 0; i < 4; ++i)
      acc[h][i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  const int32_t* lt = list ? list + (uint64_t)t * list_stride : nullptr;
  auto row = [&](uint32_t j) {
    const int32_t kp = lt ? lt[j] : first + (int32_t)j;
    return cache + (uint64_t)((uint32_t)kp % ring) * kAttnDk;
  };
  // Four binary16 values (8 bytes) or two (4 bytes) as floats.
  auto load4 = [](const _Float16* x) {
    const Half4 h = *(const Half4*)x;
    return make_float4((float)h[0], (float)h[1], (float)h[2], (float)h[3]);
  };
  auto load2 = [](const _Float16* x) {
    return make_float2((float)x[0], (float)x[1]);
  };
  for (uint32_t c0 = begin; c0 < end; c0 += kAttnChunk) {
    const uint32_t nk = min(kAttnChunk, end - c0);
    // Lane j keeps key c0 + j's scores. kAttnUnroll keys at a time, their
    // loads issued together; keys past the chunk repeat its last key and
    // are dropped.
    float sc[kAttnHpw];
    for (uint32_t h = 0; h < kAttnHpw; ++h)
      sc[h] = -INFINITY;
    for (uint32_t j = 0; j < nk; j += kAttnUnroll) {
      float4 kv[kAttnUnroll][4];
      float2 kt[kAttnUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u) {
        const _Float16* kr = row(c0 + min(j + u, nk - 1));
        for (uint32_t i = 0; i < 4; ++i)
          kv[u][i] = load4(kr + 128 * i + 4 * lane);
        kt[u] = load2(kr + kAttnDv + 2 * lane);
      }
      float d[kAttnUnroll][kAttnHpw];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kAttnHpw; ++h) {
          float x = qt[h].x * kt[u].x;
          x = fmaf(qt[h].y, kt[u].y, x);
          for (uint32_t i = 0; i < 4; ++i) {
            x = fmaf(qv[h][i].x, kv[u][i].x, x);
            x = fmaf(qv[h][i].y, kv[u][i].y, x);
            x = fmaf(qv[h][i].z, kv[u][i].z, x);
            x = fmaf(qv[h][i].w, kv[u][i].w, x);
          }
          d[u][h] = x;
        }
      for (int o = warpSize / 2; o > 0; o >>= 1)
#pragma unroll
        for (uint32_t u = 0; u < kAttnUnroll; ++u)
          for (uint32_t h = 0; h < kAttnHpw; ++h)
            d[u][h] += __shfl_xor(d[u][h], o);
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kAttnHpw; ++h)
          if (lane == j + u && j + u < nk)
            sc[h] = d[u][h] * scale;
    }
    float e[kAttnHpw];
    for (uint32_t h = 0; h < kAttnHpw; ++h) {
      const float mx = fmaxf(m[h], WarpMax(sc[h]));
      const float alpha = expf(m[h] - mx);
      e[h] = lane < nk ? expf(sc[h] - mx) : 0.0f;
      l[h] = l[h] * alpha + WarpSum(e[h]);
      m[h] = mx;
      for (uint32_t i = 0; i < 4; ++i) {
        acc[h][i].x *= alpha;
        acc[h][i].y *= alpha;
        acc[h][i].z *= alpha;
        acc[h][i].w *= alpha;
      }
    }
    for (uint32_t j = 0; j < nk; j += kAttnUnroll) {
      float4 kv[kAttnUnroll][4];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u) {
        const _Float16* kr = row(c0 + min(j + u, nk - 1));
        for (uint32_t i = 0; i < 4; ++i)
          kv[u][i] = load4(kr + 128 * i + 4 * lane);
      }
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kAttnHpw; ++h) {
          // Lanes past the chunk hold e = 0.
          const float pj = __shfl(e[h], min(j + u, kAttnChunk - 1));
          if (j + u >= nk)
            continue;
          for (uint32_t i = 0; i < 4; ++i) {
            acc[h][i].x = fmaf(pj, kv[u][i].x, acc[h][i].x);
            acc[h][i].y = fmaf(pj, kv[u][i].y, acc[h][i].y);
            acc[h][i].z = fmaf(pj, kv[u][i].z, acc[h][i].z);
            acc[h][i].w = fmaf(pj, kv[u][i].w, acc[h][i].w);
          }
        }
    }
  }
  for (uint32_t h = 0; h < kAttnHpw; ++h) {
    const uint64_t r = ((uint64_t)t * max_splits + split) * heads + h0 + h;
    if (lane == 0) {
      part_ml[2 * r] = m[h];
      part_ml[2 * r + 1] = l[h];
    }
    for (uint32_t i = 0; i < 4; ++i)
      *(float4*)(part_acc + r * kAttnDv + 128 * i + 4 * lane) = acc[h][i];
  }
}

// One block per (head, token), a float4 of the output per thread.
__global__ void __launch_bounds__(kAttnDv / 4)
    MlaCombine(const float* part_ml, const float* part_acc, const int32_t* pos,
               uint32_t window, const uint32_t* list_count, const float* sinks,
               float* out, uint32_t heads) {
  const uint32_t h = blockIdx.x, t = blockIdx.y;
  const uint32_t max_splits = AttnSplits(heads);
  int32_t first;
  const uint32_t count = AttnKeys(pos, window, list_count, t, &first);
  const uint32_t keys = AttnSplitKeys(count, max_splits);
  const uint32_t splits = keys ? (count + keys - 1) / keys : 0;
  const uint64_t r0 = (uint64_t)t * max_splits * heads + h;
  float mx = sinks ? sinks[h] : -INFINITY;
  for (uint32_t s = 0; s < splits; ++s)
    mx = fmaxf(mx, part_ml[2 * (r0 + (uint64_t)s * heads)]);
  float den = sinks ? expf(sinks[h] - mx) : 0.0f;
  float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint32_t s = 0; s < splits; ++s) {
    const uint64_t r = r0 + (uint64_t)s * heads;
    const float w = expf(part_ml[2 * r] - mx);
    den = fmaf(part_ml[2 * r + 1], w, den);
    const float4 a = *(const float4*)(part_acc + r * kAttnDv + 4 * threadIdx.x);
    o.x = fmaf(a.x, w, o.x);
    o.y = fmaf(a.y, w, o.y);
    o.z = fmaf(a.z, w, o.z);
    o.w = fmaf(a.w, w, o.w);
  }
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  *(float4*)(out + ((uint64_t)t * heads + h) * kAttnDv + 4 * threadIdx.x) =
      make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
}

// DFly2 block MLA attention: all `n` queries in a block attend to every valid
// indexed binary16 cache row sel[0..sel_count) (0 <= sel[j] <= limit) plus the
// block's own `n` FP32 latent rows kva[0..n). Uses 8 splits so 8 tokens * 2
// head groups * 8 splits = 128 workgroups saturate all CUs while partial
// buffers stay within ~4 MiB.
constexpr uint32_t kDflyAttnSplits = 8;
constexpr uint32_t kDflyAttnHpw = 4, kDflyAttnWarps = 4,
                   kDflyAttnHeadsPerBlock = kDflyAttnHpw * kDflyAttnWarps;

__global__ void __launch_bounds__(kDflyAttnWarps * 32)
    DflyAttnDecode(const float* q, const _Float16* cache, uint32_t ring,
                   const int32_t* sel, uint32_t sel_count, int32_t limit,
                   const float* kva, float scale, float* part_ml,
                   float* part_acc, uint32_t n, uint32_t heads) {
  const uint32_t split = blockIdx.x, t = blockIdx.z;
  const uint32_t count = sel_count + n;
  const uint32_t keys = AttnSplitKeys(count, kDflyAttnSplits);
  const uint32_t begin = split * keys;
  if (begin >= count)
    return;
  const uint32_t end = min(count, begin + keys);
  const uint32_t lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  const uint32_t h0 = blockIdx.y * kDflyAttnHeadsPerBlock + warp * kDflyAttnHpw;
  if (h0 >= heads)
    return;

  float4 qv[kDflyAttnHpw][4];
  float2 qt[kDflyAttnHpw];
  for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
    const float* qh = q + ((uint64_t)t * heads + h0 + h) * kAttnDk;
    for (uint32_t i = 0; i < 4; ++i)
      qv[h][i] = *(const float4*)(qh + 128 * i + 4 * lane);
    qt[h] = *(const float2*)(qh + kAttnDv + 2 * lane);
  }
  float m[kDflyAttnHpw], l[kDflyAttnHpw];
  float4 acc[kDflyAttnHpw][4];
  for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
    m[h] = -INFINITY;
    l[h] = 0.0f;
    for (uint32_t i = 0; i < 4; ++i)
      acc[h][i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  auto load4h = [](const _Float16* x) {
    const Half4 h = *(const Half4*)x;
    return make_float4((float)h[0], (float)h[1], (float)h[2], (float)h[3]);
  };
  auto load2h = [](const _Float16* x) {
    return make_float2((float)x[0], (float)x[1]);
  };
  auto load_key = [&](uint32_t idx, float4(&kv)[4], float2& kt, bool& valid) {
    if (idx < sel_count) {
      const int32_t kp = sel[idx];
      valid = kp >= 0 && kp <= limit;
      if (valid) {
        const _Float16* kr = cache + (uint64_t)((uint32_t)kp % ring) * kAttnDk;
        for (uint32_t i = 0; i < 4; ++i)
          kv[i] = load4h(kr + 128 * i + 4 * lane);
        kt = load2h(kr + kAttnDv + 2 * lane);
      } else {
        for (uint32_t i = 0; i < 4; ++i)
          kv[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        kt = make_float2(0.0f, 0.0f);
      }
    } else {
      valid = true;
      const float* kr = kva + (uint64_t)(idx - sel_count) * kAttnDk;
      for (uint32_t i = 0; i < 4; ++i)
        kv[i] = *(const float4*)(kr + 128 * i + 4 * lane);
      kt = *(const float2*)(kr + kAttnDv + 2 * lane);
    }
  };
  auto load_val = [&](uint32_t idx, float4(&kv)[4]) {
    if (idx < sel_count) {
      const int32_t kp = sel[idx];
      if (kp >= 0 && kp <= limit) {
        const _Float16* kr = cache + (uint64_t)((uint32_t)kp % ring) * kAttnDk;
        for (uint32_t i = 0; i < 4; ++i)
          kv[i] = load4h(kr + 128 * i + 4 * lane);
      } else {
        for (uint32_t i = 0; i < 4; ++i)
          kv[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    } else {
      const float* kr = kva + (uint64_t)(idx - sel_count) * kAttnDk;
      for (uint32_t i = 0; i < 4; ++i)
        kv[i] = *(const float4*)(kr + 128 * i + 4 * lane);
    }
  };

  for (uint32_t c0 = begin; c0 < end; c0 += kAttnChunk) {
    const uint32_t nk = min(kAttnChunk, end - c0);
    float sc[kDflyAttnHpw];
    for (uint32_t h = 0; h < kDflyAttnHpw; ++h)
      sc[h] = -INFINITY;
    for (uint32_t j = 0; j < nk; j += kAttnUnroll) {
      float4 kv[kAttnUnroll][4];
      float2 kt[kAttnUnroll];
      bool ok[kAttnUnroll];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        load_key(c0 + min(j + u, nk - 1), kv[u], kt[u], ok[u]);
      float d[kAttnUnroll][kDflyAttnHpw];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
          float x = qt[h].x * kt[u].x;
          x = fmaf(qt[h].y, kt[u].y, x);
          for (uint32_t i = 0; i < 4; ++i) {
            x = fmaf(qv[h][i].x, kv[u][i].x, x);
            x = fmaf(qv[h][i].y, kv[u][i].y, x);
            x = fmaf(qv[h][i].z, kv[u][i].z, x);
            x = fmaf(qv[h][i].w, kv[u][i].w, x);
          }
          d[u][h] = x;
        }
      for (int o = warpSize / 2; o > 0; o >>= 1)
#pragma unroll
        for (uint32_t u = 0; u < kAttnUnroll; ++u)
          for (uint32_t h = 0; h < kDflyAttnHpw; ++h)
            d[u][h] += __shfl_xor(d[u][h], o);
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kDflyAttnHpw; ++h)
          if (lane == j + u && j + u < nk && ok[u])
            sc[h] = d[u][h] * scale;
    }
    float e[kDflyAttnHpw];
    for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
      const float mx = fmaxf(m[h], WarpMax(sc[h]));
      const float alpha = m[h] == -INFINITY ? 0.0f : expf(m[h] - mx);
      e[h] = (lane < nk && sc[h] > -INFINITY) ? expf(sc[h] - mx) : 0.0f;
      l[h] = l[h] * alpha + WarpSum(e[h]);
      m[h] = mx;
      for (uint32_t i = 0; i < 4; ++i) {
        acc[h][i].x *= alpha;
        acc[h][i].y *= alpha;
        acc[h][i].z *= alpha;
        acc[h][i].w *= alpha;
      }
    }
    for (uint32_t j = 0; j < nk; j += kAttnUnroll) {
      float4 kv[kAttnUnroll][4];
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        load_val(c0 + min(j + u, nk - 1), kv[u]);
#pragma unroll
      for (uint32_t u = 0; u < kAttnUnroll; ++u)
        for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
          const float pj = __shfl(e[h], min(j + u, kAttnChunk - 1));
          if (j + u >= nk)
            continue;
          for (uint32_t i = 0; i < 4; ++i) {
            acc[h][i].x = fmaf(pj, kv[u][i].x, acc[h][i].x);
            acc[h][i].y = fmaf(pj, kv[u][i].y, acc[h][i].y);
            acc[h][i].z = fmaf(pj, kv[u][i].z, acc[h][i].z);
            acc[h][i].w = fmaf(pj, kv[u][i].w, acc[h][i].w);
          }
        }
    }
  }
  for (uint32_t h = 0; h < kDflyAttnHpw; ++h) {
    const uint64_t r = ((uint64_t)t * kDflyAttnSplits + split) * heads + h0 + h;
    if (lane == 0) {
      part_ml[2 * r] = m[h];
      part_ml[2 * r + 1] = l[h];
    }
    for (uint32_t i = 0; i < 4; ++i)
      *(float4*)(part_acc + r * kAttnDv + 128 * i + 4 * lane) = acc[h][i];
  }
}

__global__ void __launch_bounds__(kAttnDv / 4)
    DflyAttnCombine(const float* part_ml, const float* part_acc, uint32_t count,
                    const float* sinks, float* out, uint32_t heads) {
  const uint32_t h = blockIdx.x, t = blockIdx.y;
  const uint32_t keys = AttnSplitKeys(count, kDflyAttnSplits);
  const uint32_t splits = keys ? (count + keys - 1) / keys : 0;
  const uint64_t r0 = (uint64_t)t * kDflyAttnSplits * heads + h;
  float mx = sinks ? sinks[h] : -INFINITY;
  for (uint32_t s = 0; s < splits; ++s)
    mx = fmaxf(mx, part_ml[2 * (r0 + (uint64_t)s * heads)]);
  float den = sinks ? expf(sinks[h] - mx) : 0.0f;
  float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  for (uint32_t s = 0; s < splits; ++s) {
    const uint64_t r = r0 + (uint64_t)s * heads;
    const float ms = part_ml[2 * r];
    const float w = ms == -INFINITY ? 0.0f : expf(ms - mx);
    den = fmaf(part_ml[2 * r + 1], w, den);
    const float4 a = *(const float4*)(part_acc + r * kAttnDv + 4 * threadIdx.x);
    o.x = fmaf(a.x, w, o.x);
    o.y = fmaf(a.y, w, o.y);
    o.z = fmaf(a.z, w, o.z);
    o.w = fmaf(a.w, w, o.w);
  }
  const float inv = den > 0.0f ? 1.0f / den : 0.0f;
  *(float4*)(out + ((uint64_t)t * heads + h) * kAttnDv + 4 * threadIdx.x) =
      make_float4(o.x * inv, o.y * inv, o.z * inv, o.w * inv);
}

__global__ void StoreRows(const float* src, uint64_t src_stride, float* dst, uint32_t ring, const int32_t* pos,
                          uint32_t dim) {
  const uint32_t t = blockIdx.y;
  const uint64_t row = (uint64_t)((uint32_t)pos[t] % ring);
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    dst[row * dim + c] = src[t * src_stride + c];
  }
}

__global__ void StoreRowsHalf(const float* src, uint64_t src_stride,
                              _Float16* dst, uint32_t ring, const int32_t* pos,
                              uint32_t dim) {
  const uint32_t t = blockIdx.y;
  const uint64_t row = (uint64_t)((uint32_t)pos[t] % ring);
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim;
       c += gridDim.x * blockDim.x) {
    dst[row * dim + c] = (_Float16)src[t * src_stride + c];
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

// IndexerScore with one thread per (token, group) for 128-wide rows: a block
// takes 256 groups of kTok tokens, whose queries it shares in LDS, and
// each thread keeps its group's row in registers. Each head's dot product is
// summed as IndexerScore's warp sums it (lane l's partial over i = l + 32 j,
// then the xor tree, pairs commute), so the scores are identical.
constexpr uint32_t kIdxDim = 128, kIdxMaxHeads = 16;

template<uint32_t kTok>
__global__ void __launch_bounds__(kBlock)
    IndexerScoreRows(const float* q, const float* w, const float* pooled,
                     const int32_t* n_pool, uint32_t n_pool_max, float* score,
                     uint32_t n, uint32_t heads) {
  static_assert(kIdxDim == 4 * 32, "four values per lane partial");
  __shared__ __align__(16) float qs[kTok * kIdxMaxHeads * kIdxDim];
  __shared__ float ws[kTok * kIdxMaxHeads];
  const uint32_t t0 = blockIdx.y * kTok;
  const uint32_t nt = min(kTok, n - t0);
  const float4* qt =
      reinterpret_cast<const float4*>(q + (uint64_t)t0 * heads * kIdxDim);
  for (uint32_t i = threadIdx.x; i < nt * heads * kIdxDim / 4; i += blockDim.x)
    reinterpret_cast<float4*>(qs)[i] = qt[i];
  if (threadIdx.x < nt * heads)
    ws[threadIdx.x] = w[(uint64_t)t0 * heads + threadIdx.x];
  __syncthreads();
  const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_pool_max)
    return;
  float k[kIdxDim];
  const float4* kr =
      reinterpret_cast<const float4*>(pooled + (uint64_t)g * kIdxDim);
#pragma unroll
  for (uint32_t i = 0; i < kIdxDim / 4; ++i) {
    const float4 v = kr[i];
    k[4 * i] = v.x;
    k[4 * i + 1] = v.y;
    k[4 * i + 2] = v.z;
    k[4 * i + 3] = v.w;
  }
  for (uint32_t tt = 0; tt < nt; ++tt) {
    const uint32_t t = t0 + tt;
    float total = -INFINITY;
    if ((int32_t)g < n_pool[t]) {
      total = 0.0f;
#pragma unroll 1
      for (uint32_t h = 0; h < heads; ++h) {
        const float* qh = qs + (tt * heads + h) * kIdxDim;
        float p[32];
#pragma unroll
        for (uint32_t l = 0; l < 32; ++l) {
          float d = 0.0f;
#pragma unroll
          for (uint32_t j = 0; j < kIdxDim / 32; ++j)
            d += qh[l + 32 * j] * k[l + 32 * j];
          p[l] = d;
        }
#pragma unroll
        for (uint32_t o = 16; o > 0; o >>= 1)
#pragma unroll
          for (uint32_t l = 0; l < o; ++l)
            p[l] = p[l] + p[l + o];
        total += fmaxf(p[0], 0.0f) * ws[tt * heads + h];
      }
    }
    score[(uint64_t)t * n_pool_max + g] = total;
  }
}

// One block per token: k rounds of block argmax over the visible groups
// (kept for k > kSelMax; KpoolSelectSort gives the same selection).
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

constexpr uint32_t kSelMax = 512;
// KpoolSelect's selection in its order without its k argmax rounds: the
// take-th largest key by a byte-wise radix select, then the keys at or above
// it sorted in LDS. One block per token.
//
// Descending order of (score, -group): the score's bits made unsigned-
// orderable (-0 as +0, as the argmax's comparisons treat them), the group
// complemented below them, so that keys are distinct and a lower group sorts
// first among equal scores.
__device__ __forceinline__ uint64_t SelKey(float v, uint32_t g) {
  uint32_t u = __float_as_uint(v == 0.0f ? 0.0f : v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (uint64_t)u << 32 | (0xffffffffu - g);
}
__global__ void __launch_bounds__(kBlock)
    KpoolSelectSort(const float* score, const int32_t* n_pool,
                    uint32_t n_pool_max, const int32_t* pos, uint32_t k,
                    uint32_t kpool, int32_t* sel, uint32_t n_sel) {
  static_assert(kBlock == 256, "a histogram bin per thread");
  __shared__ uint32_t hist[256];
  __shared__ uint64_t keys[kSelMax];
  __shared__ uint64_t prefix_s;
  __shared__ uint32_t want_s, count_s;
  const uint32_t t = blockIdx.x;
  const float* row = score + (uint64_t)t * n_pool_max;
  int32_t* out = sel + (uint64_t)t * n_sel;
  const uint32_t visible = (uint32_t)n_pool[t];
  const uint32_t take = visible < k ? visible : k;
  for (uint32_t i = threadIdx.x; i < n_sel; i += blockDim.x)
    out[i] = -1;
  // The take-th largest key, a byte at a time from the top.
  uint64_t prefix = 0, mask = 0;
  if (take < visible) {
    if (threadIdx.x == 0)
      want_s = take;
    for (int shift = 56; shift >= 0; shift -= 8) {
      hist[threadIdx.x] = 0;
      __syncthreads();
      for (uint32_t g = threadIdx.x; g < visible; g += blockDim.x) {
        const uint64_t key = SelKey(row[g], g);
        if ((key & mask) == prefix)
          atomicAdd(&hist[(key >> shift) & 0xff], 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        uint32_t want = want_s, b = 255;
        for (;; --b) {
          if (hist[b] >= want)
            break;
          want -= hist[b];
        }
        want_s = want;
        prefix_s = prefix | (uint64_t)b << shift;
      }
      __syncthreads();
      prefix = prefix_s;
      mask |= (uint64_t)0xff << shift;
    }
  }
  // Collect the keys at or above it (all of them when every group is taken).
  if (threadIdx.x == 0)
    count_s = 0;
  for (uint32_t i = threadIdx.x; i < kSelMax; i += blockDim.x)
    keys[i] = 0;
  __syncthreads();
  for (uint32_t g = threadIdx.x; g < visible; g += blockDim.x) {
    const uint64_t key = SelKey(row[g], g);
    if (take == visible || key >= prefix)
      keys[atomicAdd(&count_s, 1u)] = key;
  }
  __syncthreads();
  // Bitonic sort, descending; the zero padding sorts last.
  for (uint32_t size = 2; size <= kSelMax; size <<= 1)
    for (uint32_t stride = size / 2; stride > 0; stride >>= 1) {
      for (uint32_t i = threadIdx.x; i < kSelMax; i += blockDim.x) {
        const uint32_t j = i ^ stride;
        if (j > i) {
          const bool desc = (i & size) == 0;
          const uint64_t a = keys[i], b = keys[j];
          if (desc ? a < b : a > b) {
            keys[i] = b;
            keys[j] = a;
          }
        }
      }
      __syncthreads();
    }
  for (uint32_t i = threadIdx.x; i < take * kpool; i += blockDim.x) {
    const int32_t g = (int32_t)(0xffffffffu - (uint32_t)keys[i / kpool]);
    out[i] = g * (int32_t)kpool + (int32_t)(i % kpool);
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

__global__ void GatherIndexed(const _Float16* src, uint32_t ring,
                              const int32_t* idx, int32_t limit, uint32_t dim,
                              float* dst) {
  const uint32_t r = blockIdx.y;
  const int32_t p = idx[r];
  const bool ok = p >= 0 && p <= limit;
  for (uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; c < dim; c += gridDim.x * blockDim.x) {
    dst[(uint64_t)r * dim + c] =
        ok ? (float)src[(uint64_t)((uint32_t)p % ring) * dim + c] : 0.0f;
  }
}

__global__ void IndexMask(const int32_t* idx, uint32_t count, int32_t limit, float* mask, uint32_t n_keys) {
  const uint32_t t = blockIdx.y;
  for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < n_keys; j += gridDim.x * blockDim.x) {
    mask[(uint64_t)t * n_keys + j] = j < count && (idx[j] < 0 || idx[j] > limit) ? -INFINITY : 0.0f;
  }
}

__global__ void Copy2d(float* dst, uint64_t dst_pitch, const float* src,
                       uint64_t src_pitch, uint32_t width, uint64_t count) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    const uint64_t r = i / width;
    const uint64_t c = i % width;
    dst[r * dst_pitch + c] = src[r * src_pitch + c];
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

// y[t][r] = sum_c x[t][c] w[r][c] in FP32 for batched passes (the router and
// the indexer's FP32 projections). Block: 64 tokens x 128 rows, 256 threads
// of 4 tokens x 8 rows; K in steps of 32 through LDS, transposed so each
// thread reads its tokens and rows as float4. Every output sums c = 0 .. k-1
// in order in one accumulator, so a token's result does not depend on the
// pass or the other tokens. k must be a multiple of 32.
constexpr uint32_t kSgTok = 64, kSgRows = 128, kSgK = 32;
__device__ __forceinline__ float4 Load4(const float* w) {
  return *reinterpret_cast<const float4*>(w);
}
__device__ __forceinline__ float4 Load4(const uint16_t* w) {
  const uint2 raw = *reinterpret_cast<const uint2*>(w);
  return make_float4(
      __uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
      __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
}
template<typename W>
__global__ void __launch_bounds__(256)
    SgemmNt(const float* __restrict__ x, const W* __restrict__ w,
            float* __restrict__ y, uint32_t n, uint32_t m, uint32_t k) {
  __shared__ __align__(16) float xs[kSgK][kSgTok + 4];
  __shared__ __align__(16) float ws[kSgK][kSgRows + 4];
  const uint32_t tid = threadIdx.x;
  const uint32_t t0 = blockIdx.y * kSgTok, r0 = blockIdx.x * kSgRows;
  const uint32_t tt = (tid % 16) * 4, rr = (tid / 16) * 8;
  // Staging: thread tid loads values 4 (tid % 8) .. + 3 of tokens tid / 8
  // and 32 + tid / 8, and of rows tid / 8 + 32 i; the next step's loads are
  // issued before this step's products.
  const uint32_t lrow = tid / 8, lcol = (tid % 8) * 4;
  float4 xv[2], wv[4];
  const auto load = [&](uint32_t c0) {
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t t = t0 + lrow + 32 * i;
      xv[i] = t < n ? *reinterpret_cast<const float4*>(x + (uint64_t)t * k +
                                                       c0 + lcol)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t r = r0 + lrow + 32 * i;
      wv[i] = r < m ? Load4(w + (uint64_t)r * k + c0 + lcol)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
    }
  };
  float acc[4][8] = {};
  load(0);
  for (uint32_t c0 = 0; c0 < k; c0 += kSgK) {
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
      xs[lcol][lrow + 32 * i] = xv[i].x;
      xs[lcol + 1][lrow + 32 * i] = xv[i].y;
      xs[lcol + 2][lrow + 32 * i] = xv[i].z;
      xs[lcol + 3][lrow + 32 * i] = xv[i].w;
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      ws[lcol][lrow + 32 * i] = wv[i].x;
      ws[lcol + 1][lrow + 32 * i] = wv[i].y;
      ws[lcol + 2][lrow + 32 * i] = wv[i].z;
      ws[lcol + 3][lrow + 32 * i] = wv[i].w;
    }
    __syncthreads();
    if (c0 + kSgK < k)
      load(c0 + kSgK);
#pragma unroll 8
    for (uint32_t c = 0; c < kSgK; ++c) {
      const float4 a = *reinterpret_cast<const float4*>(&xs[c][tt]);
      const float4 b0 = *reinterpret_cast<const float4*>(&ws[c][rr]);
      const float4 b1 = *reinterpret_cast<const float4*>(&ws[c][rr + 4]);
      const float av[4] = {a.x, a.y, a.z, a.w};
      const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t t = t0 + tt + i;
    if (t >= n)
      continue;
#pragma unroll
    for (int j = 0; j < 8; ++j)
      if (r0 + rr + j < m)
        y[(uint64_t)t * m + r0 + rr + j] = acc[i][j];
  }
}

// The index of a row's largest value, the lower index on ties (NaN counts as
// -inf), in two launches: kArgBlocks blocks each reduce a chunk to its best
// (value, index), then one block reduces those.
constexpr uint32_t kArgBlocks = 64;

__device__ __forceinline__ bool Better(float a, int32_t ia, float b,
                                       int32_t ib) {
  return a > b || (a == b && ia < ib);
}

// Block b reduces x[b chunk .. (b + 1) chunk) (indices idx, else positions)
// to out_v/out_i[b].
__global__ void ArgmaxRows(const float* x, const int32_t* idx, uint32_t count,
                           uint32_t chunk, float* out_v, int32_t* out_i) {
  __shared__ float sv[kBlock / 32];
  __shared__ int32_t si[kBlock / 32];
  float v = -INFINITY;
  int32_t id = INT32_MAX;
  const uint32_t lo = blockIdx.x * chunk, hi = min(count, lo + chunk);
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    float e = x[i];
    if (e != e)
      e = -INFINITY;
    const int32_t ie = idx ? idx[i] : (int32_t)i;
    if (Better(e, ie, v, id)) {
      v = e;
      id = ie;
    }
  }
  for (int o = warpSize / 2; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o);
    const int32_t oi = __shfl_xor(id, o);
    if (Better(ov, oi, v, id)) {
      v = ov;
      id = oi;
    }
  }
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  if (lane == 0) {
    sv[warp] = v;
    si[warp] = id;
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  for (int w = 1; w < (int)((blockDim.x + warpSize - 1) / warpSize); ++w)
    if (Better(sv[w], si[w], v, id)) {
      v = sv[w];
      id = si[w];
    }
  out_v[blockIdx.x] = v;
  out_i[blockIdx.x] = id;
}

// Batched finite argmax over `rows` rows of [dim] floats (blockIdx.y = row).
__device__ __forceinline__ bool SampleBetter(float v, int32_t a, float best,
                                             int32_t best_a) {
  return a >= 0 && (best_a < 0 || v > best || (v == best && a < best_a));
}

__global__ void ArgmaxBatchPass1(const float* x, uint32_t dim, uint32_t chunk,
                                 float* part_v, int32_t* part_i) {
  __shared__ float sv[kBlock / 32];
  __shared__ int32_t si[kBlock / 32];
  const uint32_t row = blockIdx.y;
  const float* xr = x + (uint64_t)row * dim;
  float v = -INFINITY;
  int32_t id = -1;
  const uint32_t lo = blockIdx.x * chunk, hi = min(dim, lo + chunk);
  for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float e = xr[i];
    if (isfinite(e) && SampleBetter(e, (int32_t)i, v, id)) {
      v = e;
      id = (int32_t)i;
    }
  }
  for (int o = warpSize / 2; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o);
    const int32_t oi = __shfl_xor(id, o);
    if (SampleBetter(ov, oi, v, id)) {
      v = ov;
      id = oi;
    }
  }
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  if (lane == 0) {
    sv[warp] = v;
    si[warp] = id;
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  for (int w = 1; w < (int)((blockDim.x + warpSize - 1) / warpSize); ++w)
    if (SampleBetter(sv[w], si[w], v, id)) {
      v = sv[w];
      id = si[w];
    }
  part_v[(uint64_t)row * kArgBlocks + blockIdx.x] = v;
  part_i[(uint64_t)row * kArgBlocks + blockIdx.x] = id;
}

__global__ void ArgmaxBatchPass2(const float* part_v, const int32_t* part_i,
                                 int32_t* out) {
  const uint32_t row = blockIdx.x;
  const float* pv = part_v + (uint64_t)row * kArgBlocks;
  const int32_t* pi = part_i + (uint64_t)row * kArgBlocks;
  float v = -INFINITY;
  int32_t id = -1;
  for (uint32_t i = threadIdx.x; i < kArgBlocks; i += blockDim.x) {
    if (SampleBetter(pv[i], pi[i], v, id)) {
      v = pv[i];
      id = pi[i];
    }
  }
  for (int o = warpSize / 2; o > 0; o >>= 1) {
    const float ov = __shfl_xor(v, o);
    const int32_t oi = __shfl_xor(id, o);
    if (SampleBetter(ov, oi, v, id)) {
      v = ov;
      id = oi;
    }
  }
  __shared__ float sv[kBlock / 32];
  __shared__ int32_t si[kBlock / 32];
  const int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
  if (lane == 0) {
    sv[warp] = v;
    si[warp] = id;
  }
  __syncthreads();
  if (threadIdx.x != 0)
    return;
  for (int w = 1; w < (int)((blockDim.x + warpSize - 1) / warpSize); ++w)
    if (SampleBetter(sv[w], si[w], v, id)) {
      v = sv[w];
      id = si[w];
    }
  out[row] = id;
}

// Exact device sampling on one logit row, adapted from ds41-gufo-extension
// (ds41_v41_sample_row, E131).
constexpr uint32_t kSampleBlocks = 128, kSampleThreads = 256, kSamplePer = 8;
constexpr double kSampleTiny = 1e-300;

struct SampleScratch {
  hy4_sample_result result;
  float max[kSampleBlocks];
  int32_t arg[kSampleBlocks];
  int32_t big[kSampleBlocks];
  double sum[kSampleBlocks];
  float raw_max[kSampleBlocks];
  int32_t raw_arg[kSampleBlocks];
};
static_assert(sizeof(SampleScratch) <= HY4_SAMPLE_SCRATCH_BYTES);

__device__ __forceinline__ float SampleLogit(const float* logits,
                                             const uint32_t* mask, uint32_t i) {
  return !mask || ((mask[i >> 5] >> (i & 31)) & 1u) ? logits[i] : -INFINITY;
}

__device__ void SampleSpan(uint32_t n, uint32_t b, uint32_t* lo, uint32_t* hi) {
  const uint32_t span = (n + kSampleBlocks - 1) / kSampleBlocks;
  *lo = min(n, b * span);
  *hi = min(n, *lo + span);
}

__global__ void __launch_bounds__(kSampleThreads)
    SampleMaxKernel(const float* logits, const uint32_t* mask, uint32_t n,
                    SampleScratch* s) {
  __shared__ float sv[kSampleThreads], rv[kSampleThreads];
  __shared__ int32_t sa[kSampleThreads], ra[kSampleThreads];
  uint32_t lo, hi;
  SampleSpan(n, blockIdx.x, &lo, &hi);
  float v = -INFINITY, w = -INFINITY;
  int32_t a = -1, b = -1;
  for (uint32_t i = lo + threadIdx.x; i < hi; i += kSampleThreads) {
    const float x = logits[i];
    if (!isfinite(x))
      continue;
    if (SampleBetter(x, (int32_t)i, w, b))
      w = x, b = (int32_t)i;
    if (isfinite(SampleLogit(logits, mask, i)) &&
        SampleBetter(x, (int32_t)i, v, a))
      v = x, a = (int32_t)i;
  }
  sv[threadIdx.x] = v;
  sa[threadIdx.x] = a;
  rv[threadIdx.x] = w;
  ra[threadIdx.x] = b;
  __syncthreads();
  for (uint32_t o = kSampleThreads / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o &&
        SampleBetter(sv[threadIdx.x + o], sa[threadIdx.x + o], sv[threadIdx.x],
                     sa[threadIdx.x]))
      sv[threadIdx.x] = sv[threadIdx.x + o],
      sa[threadIdx.x] = sa[threadIdx.x + o];
    if (threadIdx.x < o &&
        SampleBetter(rv[threadIdx.x + o], ra[threadIdx.x + o], rv[threadIdx.x],
                     ra[threadIdx.x]))
      rv[threadIdx.x] = rv[threadIdx.x + o],
      ra[threadIdx.x] = ra[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s->max[blockIdx.x] = sv[0];
    s->arg[blockIdx.x] = sa[0];
    s->raw_max[blockIdx.x] = rv[0];
    s->raw_arg[blockIdx.x] = ra[0];
  }
}

__device__ void SampleGlobalMax(const float* max, const int32_t* arg, float* v,
                                int32_t* a) {
  *v = -INFINITY;
  *a = -1;
  for (uint32_t b = 0; b < kSampleBlocks; ++b)
    if (SampleBetter(max[b], arg[b], *v, *a))
      *v = max[b], *a = arg[b];
}

__global__ void SampleArgmaxFinishKernel(const uint32_t* mask,
                                         SampleScratch* s) {
  if (threadIdx.x != 0)
    return;
  hy4_sample_result r{-1, 0, -1};
  float mx;
  int32_t arg;
  SampleGlobalMax(s->max, s->arg, &mx, &arg);
  r.token = arg;
  r.degenerate = arg < 0 ? 1 : 0;
  r.argmax = arg;
  if (mask) {
    float raw;
    SampleGlobalMax(s->raw_max, s->raw_arg, &raw, &r.argmax);
  }
  s->result = r;
}

__device__ __forceinline__ double SampleTerm(float x, double mx, double t) {
  return isfinite(x) ? exp(((double)x - mx) / t) : 0.0;
}

__global__ void __launch_bounds__(kSampleThreads)
    SampleSumKernel(const float* logits, const uint32_t* mask, uint32_t n,
                    float temperature, SampleScratch* s) {
  __shared__ double ss[kSampleThreads];
  __shared__ int32_t sb[kSampleThreads];
  __shared__ float smax;
  if (threadIdx.x == 0) {
    int32_t a;
    SampleGlobalMax(s->max, s->arg, &smax, &a);
  }
  __syncthreads();
  uint32_t lo, hi;
  SampleSpan(n, blockIdx.x, &lo, &hi);
  const double mx = smax, t = temperature;
  double sum = 0.0;
  int32_t big = 0;
  if (isfinite(smax)) {
    for (uint32_t i = lo + threadIdx.x; i < hi; i += kSampleThreads) {
      const double e = SampleTerm(SampleLogit(logits, mask, i), mx, t);
      sum += e;
      big += e > kSampleTiny;
    }
  }
  ss[threadIdx.x] = sum;
  sb[threadIdx.x] = big;
  __syncthreads();
  for (uint32_t o = kSampleThreads / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) {
      ss[threadIdx.x] += ss[threadIdx.x + o];
      sb[threadIdx.x] += sb[threadIdx.x + o];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    s->sum[blockIdx.x] = ss[0];
    s->big[blockIdx.x] = sb[0];
  }
}

__global__ void __launch_bounds__(kSampleThreads)
    SamplePickKernel(const float* logits, const uint32_t* mask, uint32_t n,
                     float temperature, double u, SampleScratch* s) {
  __shared__ double scan[kSampleThreads];
  __shared__ double sh_total, sh_start;
  __shared__ float sh_max;
  __shared__ int32_t sh_owner, sh_found, sh_last, sh_done;
  if (threadIdx.x == 0) {
    hy4_sample_result r{-1, 0, -1};
    int32_t arg;
    SampleGlobalMax(s->max, s->arg, &sh_max, &arg);
    r.argmax = arg;
    if (mask) {
      float raw;
      SampleGlobalMax(s->raw_max, s->raw_arg, &raw, &r.argmax);
    }
    double total = 0.0;
    int32_t big = 0;
    for (uint32_t b = 0; b < kSampleBlocks; ++b) {
      total += s->sum[b];
      big += s->big[b];
    }
    sh_done = 1;
    sh_total = total;
    if (arg < 0 || !(total > 0.0) || big < 2) {
      r.degenerate = 1;
    } else {
      sh_done = 0;
      double left = u;
      int32_t owner = -1, last = -1;
      for (uint32_t b = 0; b < kSampleBlocks; ++b) {
        const double w = s->sum[b] / total;
        if (!(w > 0.0))
          continue;
        last = (int32_t)b;
        if (left - w < 0.0) {
          owner = (int32_t)b;
          break;
        }
        left -= w;
      }
      sh_owner = owner >= 0 ? owner : last;
      sh_start = owner >= 0 ? left : INFINITY;
      if (sh_owner < 0)
        r.degenerate = sh_done = 1;
    }
    s->result = r;
  }
  __syncthreads();
  if (sh_done)
    return;
  uint32_t lo, hi;
  SampleSpan(n, (uint32_t)sh_owner, &lo, &hi);
  const double mx = sh_max, t = temperature, total = sh_total;
  const uint32_t first = lo + threadIdx.x * kSamplePer;
  double run[kSamplePer];
  double local = 0.0;
#pragma unroll
  for (uint32_t j = 0; j < kSamplePer; ++j) {
    const uint32_t idx = first + j;
    run[j] = idx < hi
                 ? SampleTerm(SampleLogit(logits, mask, idx), mx, t) / total
                 : 0.0;
    local += run[j];
  }
  scan[threadIdx.x] = local;
  if (threadIdx.x == 0) {
    sh_found = INT32_MAX;
    sh_last = -1;
  }
  __syncthreads();
  for (uint32_t o = 1; o < kSampleThreads; o <<= 1) {
    const double add = threadIdx.x >= o ? scan[threadIdx.x - o] : 0.0;
    __syncthreads();
    scan[threadIdx.x] += add;
    __syncthreads();
  }
  double left = sh_start - (scan[threadIdx.x] - local);
  int32_t found = -1, last = -1;
#pragma unroll
  for (uint32_t j = 0; j < kSamplePer; ++j) {
    if (!(run[j] > 0.0))
      continue;
    last = (int32_t)(first + j);
    left -= run[j];
    if (found < 0 && left < 0.0)
      found = (int32_t)(first + j);
  }
  if (found >= 0)
    atomicMin(&sh_found, found);
  if (last >= 0)
    atomicMax(&sh_last, last);
  __syncthreads();
  if (threadIdx.x == 0)
    s->result.token = sh_found != INT32_MAX ? sh_found : sh_last;
}

// Q8_0 blocks of 32 binary16 values: d = max |v| / 127, q = round(v / d).
__global__ void QuantizeQ8Half(const _Float16* src, BlockQ8_0* dst,
                               uint64_t blocks) {
  const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  float v[32];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    v[i] = (float)src[b * 32 + i];
    amax = fmaxf(amax, fabsf(v[i]));
  }
  const float d = amax / 127.0f;
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  dst[b].d = (__half)d;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    dst[b].qs[i] = (int8_t)rintf(v[i] * inv);
}

// Q8_0 blocks of 32 bfloat16 values (the trunk's LM head): the same d and q
// as the binary16 form above, the weights read as BF16.
__global__ void QuantizeQ8Bf16(const uint16_t* src, BlockQ8_0* dst,
                               uint64_t blocks) {
  const uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= blocks)
    return;
  float v[32];
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 32; i += 8) {
    Load8Bf16(src + b * 32 + i, v + i);
  }
#pragma unroll
  for (int i = 0; i < 32; ++i)
    amax = fmaxf(amax, fabsf(v[i]));
  const float d = amax / 127.0f;
  const float inv = d > 0.0f ? 1.0f / d : 0.0f;
  dst[b].d = (__half)d;
#pragma unroll
  for (int i = 0; i < 32; ++i)
    dst[b].qs[i] = (int8_t)rintf(v[i] * inv);
}

// ------------------------------------------------- fused prefill attention

// MLA prefill attention on the matrix cores, each token over exactly its own
// keys, so its result does not depend on the pass it is evaluated in.
// Block (token, 32 heads), eight waves in two groups of four, one group per
// 16 heads, so each staged key tile serves 32 heads. Keys arrive as binary16
// rows (converted once per pass) and are staged in 32-row tiles of LDS
// [key][576]. Scores: wave w of a group
// multiplies its heads' Q (binary16 in registers) by K over dimensions
// 144 w .. 144 w + 143; the four partials meet in LDS. Threads 8 h .. 8 h + 7
// run head h's FP32 online softmax, four keys each, the sink seeding its
// maximum and denominator, and write binary16 probabilities. Values: wave w of
// a group accumulates output dimensions 128 w .. 128 w + 127 in FP32. Fragment
// layout (wave32 v_wmma_f32_16x16x16_f16): A holds row L%16 and 16 contiguous
// k, B column L%16 and 16 contiguous k; accumulator element i is row 2i + L/16,
// column L%16.
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
    MlaPrefill(const _Float16* q, const _Float16* keys, uint32_t ring,
               int32_t lo, const int32_t* pos, uint32_t window,
               const int32_t* sel, uint32_t n_sel, const float* sinks,
               float scale, _Float16* out, uint32_t heads) {
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
  // A selection has n_sel = (top + 1) kpool slots, so a token below
  // position n_sel sees at most top groups and selects every earlier
  // position: it reads them in causal order, as in a pass with no selection,
  // so that its result does not depend on the pass it is in.
  const int32_t* sel_t =
      sel && p >= (int32_t)n_sel ? sel + (uint64_t)t * n_sel : nullptr;
  const uint32_t count = sel_t       ? n_sel
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
    // The tile's ring rows, -1 for a key the token does not see.
    if (tid < kMlaTile) {
      const int32_t k = MlaKey(sel_t, lo_t, j0 + tid, count, p, lo_t);
      rows[tid] = k >= 0 ? (int32_t)((uint32_t)k % ring) : -1;
    }
    __syncthreads();
    // Stage the tile: 16-byte groups of the binary16 rows, every load
    // issued before the stores.
    {
      constexpr uint32_t kThreads = 32 * kMlaWaves;
      constexpr uint32_t kGroups = kMlaTile * (kMlaDk / 8) / kThreads;  // 9
      static_assert(kMlaTile * (kMlaDk / 8) % kThreads == 0);
      uint4 v[kGroups];
#pragma unroll
      for (uint32_t g = 0; g < kGroups; ++g) {
        const uint32_t i = tid + g * kThreads;
        const uint32_t r = i / (kMlaDk / 8);
        const uint32_t c = (i % (kMlaDk / 8)) * 8;
        v[g] = make_uint4(0, 0, 0, 0);
        if (rows[r] >= 0)
          v[g] = *reinterpret_cast<const uint4*>(
              keys + (uint64_t)rows[r] * kMlaDk + c);
      }
#pragma unroll
      for (uint32_t g = 0; g < kGroups; ++g) {
        const uint32_t i = tid + g * kThreads;
        *reinterpret_cast<uint4*>(kt + i / (kMlaDk / 8) * kMlaKtPitch +
                                  (i % (kMlaDk / 8)) * 8) = v[g];
      }
    }
    __syncthreads();

    // Partial scores over this wave's dimensions.
    // The two 16-key halves' products interleave (independent chains).
    {
      constexpr uint32_t kB = kMlaTile / 16;
      MlaFloat8 sc[kB] = {};
      const _Float16* kr = kt + row * kMlaKtPitch + sub * (kMlaDk / kMlaSplit);
#pragma unroll
      for (uint32_t s = 0; s < kMlaQkSteps; ++s)
#pragma unroll
        for (uint32_t b = 0; b < kB; ++b)
          sc[b] = MlaWmma(qa[s], MlaLoad16(kr + b * 16 * kMlaKtPitch + s * 16),
                          sc[b]);
#pragma unroll
      for (uint32_t b = 0; b < kB; ++b)
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i)
          part[sub][hg + 2 * i + half][b * 16 + row] = sc[b][i];
    }
    __syncthreads();

    // Online softmax: eight lanes per head, four keys per lane.
    {
      constexpr uint32_t kLanes = 8, kKeys = kMlaTile / kLanes;
      static_assert(kMlaHeads * kLanes == 32 * kMlaWaves);
      const uint32_t h = tid / kLanes;
      const uint32_t k0 = tid % kLanes * kKeys;
      const float m_old = m_s[h];
      float x[kKeys];
      float mx = m_old;
#pragma unroll
      for (uint32_t k = 0; k < kKeys; ++k) {
        float v = -INFINITY;
        if (rows[k0 + k] >= 0)
          v = (part[0][h][k0 + k] + part[1][h][k0 + k] + part[2][h][k0 + k] +
               part[3][h][k0 + k]) *
              scale;
        x[k] = v;
        mx = fmaxf(mx, v);
      }
#pragma unroll
      for (uint32_t o = 1; o < kLanes; o <<= 1)
        mx = fmaxf(mx, __shfl_xor(mx, o));
      float sum = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < kKeys; ++k) {
        const float e = x[k] == -INFINITY ? 0.0f : __expf(x[k] - mx);
        sum += e;
        probs[h * kMlaProbPitch + k0 + k] = (_Float16)e;
      }
#pragma unroll
      for (uint32_t o = 1; o < kLanes; o <<= 1)
        sum += __shfl_xor(sum, o);
      if (tid % kLanes == 0) {
        const float alpha = mx == -INFINITY ? 1.0f : __expf(m_old - mx);
        l_s[h] = l_s[h] * alpha + sum;
        m_s[h] = mx;
        alpha_s[h] = alpha;
      }
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
      for (uint32_t c = 0; c < kMlaPvTiles; c += 2) {
        // Tiles c and c + 1 hold the even and odd dimensions of 32: lane
        // row's word of each key carries both.
        const uint32_t d = sub * (kMlaDv / kMlaSplit) + c * 16 + 2 * row;
        uint32_t ev[8], od[8];
#pragma unroll
        for (uint32_t k = 0; k < 16; k += 2) {
          const uint32_t w0 = *reinterpret_cast<const uint32_t*>(
              kt + (b * 16 + k) * kMlaKtPitch + d);
          const uint32_t w1 = *reinterpret_cast<const uint32_t*>(
              kt + (b * 16 + k + 1) * kMlaKtPitch + d);
          ev[k / 2] = __builtin_amdgcn_perm(w1, w0, 0x05040100u);
          od[k / 2] = __builtin_amdgcn_perm(w1, w0, 0x07060302u);
        }
        acc[c] = MlaWmma(pa, __builtin_bit_cast(MlaHalf16, ev), acc[c]);
        acc[c + 1] = MlaWmma(pa, __builtin_bit_cast(MlaHalf16, od), acc[c + 1]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t h = hg + 2 * i + half;
    const float l = l_s[h];
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    _Float16* orow = out + ((uint64_t)t * heads + h0 + h) * kMlaDv +
                     sub * (kMlaDv / kMlaSplit) + 2 * row;
#pragma unroll
    for (uint32_t c = 0; c < kMlaPvTiles; c += 2) {
      const _Float16 v[2] = {(_Float16)(acc[c][i] * inv),
                             (_Float16)(acc[c + 1][i] * inv)};
      *reinterpret_cast<uint32_t*>(orow + c * 16) =
          __builtin_bit_cast(uint32_t, v);
    }
  }
}

// 64-query MlaPrefill: 16 waves (512 threads) = 4 groups of 16 queries x 4
// dim-splits share each 32-key kt tile in LDS, halving KV loads vs the
// 32-query kernel.
// - kHeads == 64: all 64 heads of 1 token (supports dense, SWA and sparse sel).
// - kHeads == 32: 2 consecutive tokens x 32 heads (dense/SWA, sel == nullptr),
//   with key tiles aligned to 32-key boundaries for pass independence.
template<uint32_t kHeads>
__global__ void __launch_bounds__(512)
    MlaPrefill64(const _Float16* q, const _Float16* keys, uint32_t ring,
                 int32_t lo, const int32_t* pos, uint32_t window,
                 const int32_t* sel, uint32_t n_sel, const float* sinks,
                 float scale, _Float16* out, uint32_t n) {
  constexpr uint32_t kQ = 64;
  constexpr uint32_t kToks = kHeads == 32 ? 2u : 1u;
  constexpr uint32_t kThreads = 512;
  constexpr uint32_t kVecLoads = kMlaTile * (kMlaDk / 8);              // 2304
  constexpr uint32_t kGroups = (kVecLoads + kThreads - 1) / kThreads;  // 5
  __shared__ __align__(16) _Float16 kt[kMlaTile * kMlaKtPitch];
  __shared__ __align__(16) _Float16 probs[kQ * kMlaProbPitch];
  __shared__ float part[kMlaSplit][kQ][16];
  __shared__ int32_t rows[kMlaTile];
  __shared__ float m_s[kQ], l_s[kQ], alpha_s[kQ];

  const uint32_t t0 = blockIdx.x * kToks;
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t wave = tid / 32;
  const uint32_t row = lane % 16;
  const uint32_t half = lane / 16;
  const uint32_t qg = (wave / kMlaSplit) * 16;  // 0, 16, 32, 48
  const uint32_t sub = wave % kMlaSplit;

  const uint32_t t_wave = kToks == 2 ? t0 + (qg / 32) : t0;
  const uint32_t h_wave = kToks == 2 ? (qg % 32) : qg;
  const uint32_t t_valid = t_wave < n ? t_wave : (n - 1);

  MlaHalf16 qa[kMlaQkSteps];
  {
    const _Float16* qr = q +
                         ((uint64_t)t_valid * kHeads + h_wave + row) * kMlaDk +
                         sub * (kMlaDk / kMlaSplit);
#pragma unroll
    for (uint32_t s = 0; s < kMlaQkSteps; ++s)
      qa[s] = MlaLoad16(qr + s * 16);
  }
  if (tid < kQ) {
    const uint32_t h = kToks == 2 ? (tid % 32) : tid;
    m_s[tid] = sinks ? sinks[h] : -INFINITY;
    l_s[tid] = sinks ? 1.0f : 0.0f;
  }
  MlaFloat8 acc[kMlaPvTiles] = {};

  const int32_t p0 = pos[t0];
  const int32_t p1 = (kToks == 2 && t0 + 1 < n) ? pos[t0 + 1] : p0;
  const int32_t lo0 = window ? max(lo, p0 - (int32_t)window + 1) : lo;
  const int32_t lo1 = window ? max(lo, p1 - (int32_t)window + 1) : lo;

  const int32_t* sel_t = (kToks == 1 && sel && p0 >= (int32_t)n_sel)
                             ? sel + (uint64_t)t0 * n_sel
                             : nullptr;
  const int32_t lo_min = min(lo0, lo1);
  const int32_t p_max = max(p0, p1);
  const int32_t k_base0 = (kToks == 2 && window) ? (lo_min & ~31) : lo_min;
  const uint32_t steps = sel_t             ? n_sel
                         : p_max >= lo_min ? (uint32_t)(p_max - k_base0 + 1)
                                           : 0u;

  const uint32_t qi = tid / 8;
  const uint32_t kl = tid % 8;
  const uint32_t k0 = kl * 4;
  const bool is_t1 = kToks == 2 && qi >= 32;
  const int32_t my_lo = is_t1 ? lo1 : lo0;
  const int32_t my_p = is_t1 ? p1 : p0;
  const bool wave_t1 = kToks == 2 && qg >= 32;
  const int32_t wave_lo = wave_t1 ? lo1 : lo0;
  const int32_t wave_p = wave_t1 ? p1 : p0;

  for (uint32_t j0 = 0; j0 < steps; j0 += kMlaTile) {
    if (tid < kMlaTile) {
      int32_t k = -1;
      if (sel_t) {
        k = MlaKey(sel_t, lo0, j0 + tid, steps, p0, lo0);
      } else {
        const int32_t cand = k_base0 + (int32_t)(j0 + tid);
        if (cand >= lo_min && cand <= p_max)
          k = cand;
      }
      rows[tid] = k >= 0 ? (int32_t)((uint32_t)k % ring) : -1;
    }
    __syncthreads();
    {
      uint4 v[kGroups];
#pragma unroll
      for (uint32_t g = 0; g < kGroups; ++g) {
        const uint32_t i = tid + g * kThreads;
        v[g] = make_uint4(0, 0, 0, 0);
        if (i < kVecLoads) {
          const uint32_t r = i / (kMlaDk / 8);
          const uint32_t c = (i % (kMlaDk / 8)) * 8;
          if (rows[r] >= 0)
            v[g] = *reinterpret_cast<const uint4*>(
                keys + (uint64_t)rows[r] * kMlaDk + c);
        }
      }
#pragma unroll
      for (uint32_t g = 0; g < kGroups; ++g) {
        const uint32_t i = tid + g * kThreads;
        if (i < kVecLoads)
          *reinterpret_cast<uint4*>(kt + i / (kMlaDk / 8) * kMlaKtPitch +
                                    (i % (kMlaDk / 8)) * 8) = v[g];
      }
    }
    __syncthreads();

    MlaFloat8 sc[2] = {};
    {
      const _Float16* kr = kt + row * kMlaKtPitch + sub * (kMlaDk / kMlaSplit);
#pragma unroll
      for (uint32_t s = 0; s < kMlaQkSteps; ++s)
#pragma unroll
        for (uint32_t b = 0; b < 2; ++b)
          sc[b] = MlaWmma(qa[s], MlaLoad16(kr + b * 16 * kMlaKtPitch + s * 16),
                          sc[b]);
    }
    // Exchange 16 keys at a time so part[4][64][16] stays 16 KiB.
    // Lanes 0..3 collect b=0 (keys 0..15) and lanes 4..7 collect b=1 (keys
    // 16..31) so every lane owns the exact same 4 contiguous keys as
    // MlaPrefill, keeping softmax reduction bit-identical.
    float x[4];
#pragma unroll
    for (uint32_t b = 0; b < 2; ++b) {
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i)
        part[sub][qg + 2 * i + half][row] = sc[b][i];
      __syncthreads();
      if ((kl >> 2) == b) {
        const uint32_t base_k = (kl & 3) * 4;
#pragma unroll
        for (uint32_t kk = 0; kk < 4; ++kk) {
          const uint32_t kidx = base_k + kk;
          bool valid = rows[b * 16 + kidx] >= 0;
          if (kToks == 2 && valid) {
            const int32_t kpos = k_base0 + (int32_t)(j0 + b * 16 + kidx);
            valid = kpos >= my_lo && kpos <= my_p;
          }
          x[kk] = valid ? (part[0][qi][kidx] + part[1][qi][kidx] +
                           part[2][qi][kidx] + part[3][qi][kidx]) *
                              scale
                        : -INFINITY;
        }
      }
      if (b == 0)
        __syncthreads();
    }
    {
      const float m_old = m_s[qi];
      float mx = m_old;
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k)
        mx = fmaxf(mx, x[k]);
#pragma unroll
      for (uint32_t o = 1; o < 8; o <<= 1)
        mx = fmaxf(mx, __shfl_xor(mx, o));
      float sum = 0.0f;
#pragma unroll
      for (uint32_t k = 0; k < 4; ++k) {
        const float e = x[k] == -INFINITY ? 0.0f : __expf(x[k] - mx);
        sum += e;
        probs[qi * kMlaProbPitch + k0 + k] = (_Float16)e;
      }
#pragma unroll
      for (uint32_t o = 1; o < 8; o <<= 1)
        sum += __shfl_xor(sum, o);
      if (kl == 0) {
        const bool active_q =
            kToks == 1 || (k_base0 + (int32_t)j0 <= my_p &&
                           k_base0 + (int32_t)j0 + (int32_t)kMlaTile > my_lo);
        if (active_q) {
          const float alpha = mx == -INFINITY ? 1.0f : __expf(m_old - mx);
          l_s[qi] = l_s[qi] * alpha + sum;
          m_s[qi] = mx;
          alpha_s[qi] = alpha;
        } else {
          alpha_s[qi] = 1.0f;
        }
      }
    }
    __syncthreads();

    const bool active_w =
        kToks == 1 || (k_base0 + (int32_t)j0 <= wave_p &&
                       k_base0 + (int32_t)j0 + (int32_t)kMlaTile > wave_lo);
    if (active_w) {
#pragma unroll
      for (uint32_t i = 0; i < 8; ++i) {
        const float a = alpha_s[qg + 2 * i + half];
#pragma unroll
        for (uint32_t c = 0; c < kMlaPvTiles; ++c)
          acc[c][i] *= a;
      }
#pragma unroll
      for (uint32_t b = 0; b < 2; ++b) {
        const MlaHalf16 pa =
            MlaLoad16(probs + (qg + row) * kMlaProbPitch + b * 16);
#pragma unroll
        for (uint32_t c = 0; c < kMlaPvTiles; c += 2) {
          const uint32_t d = sub * (kMlaDv / kMlaSplit) + c * 16 + 2 * row;
          uint32_t ev[8], od[8];
#pragma unroll
          for (uint32_t k = 0; k < 16; k += 2) {
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(
                kt + (b * 16 + k) * kMlaKtPitch + d);
            const uint32_t w1 = *reinterpret_cast<const uint32_t*>(
                kt + (b * 16 + k + 1) * kMlaKtPitch + d);
            ev[k / 2] = __builtin_amdgcn_perm(w1, w0, 0x05040100u);
            od[k / 2] = __builtin_amdgcn_perm(w1, w0, 0x07060302u);
          }
          acc[c] = MlaWmma(pa, __builtin_bit_cast(MlaHalf16, ev), acc[c]);
          acc[c + 1] =
              MlaWmma(pa, __builtin_bit_cast(MlaHalf16, od), acc[c + 1]);
        }
      }
    }
    __syncthreads();
  }

  if (t_wave >= n)
    return;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t qidx = qg + 2 * i + half;
    const uint32_t h = h_wave + 2 * i + half;
    const float l = l_s[qidx];
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    _Float16* orow = out + ((uint64_t)t_wave * kHeads + h) * kMlaDv +
                     sub * (kMlaDv / kMlaSplit) + 2 * row;
#pragma unroll
    for (uint32_t c = 0; c < kMlaPvTiles; c += 2) {
      const _Float16 v[2] = {(_Float16)(acc[c][i] * inv),
                             (_Float16)(acc[c + 1][i] * inv)};
      *reinterpret_cast<uint32_t*>(orow + c * 16) =
          __builtin_bit_cast(uint32_t, v);
    }
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

int hy4_gather_rows(const void* src, uint64_t row_bytes, const int32_t* ids,
                    uint32_t n, void* dst, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (row_bytes % 4)
    return 1;
  GatherRows<<<n, kBlock, 0, stream>>>((const uint32_t*)src, row_bytes / 4, ids,
                                       (uint32_t*)dst);
  return Status();
}

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

int hy4_rms_norm_half(const float* x, const float* weight, float* out,
                      uint16_t* out_half, uint32_t rows, uint32_t dim,
                      uint32_t stride, float eps, hipStream_t stream) {
  if (rows == 0)
    return 0;
  RmsNorm<<<rows, kBlock, 0, stream>>>(x, weight, out, dim, stride, eps,
                                       reinterpret_cast<_Float16*>(out_half));
  return Status();
}

// The mixes' per-slice partial sums, grown to n tokens on this thread.
static float* HcPartial(uint32_t n) {
  static thread_local float* partial = nullptr;
  static thread_local uint32_t partial_tokens = 0;
  if (n > partial_tokens) {
    if (partial) (void)hipFree(partial);
    partial = nullptr;
    partial_tokens = 0;
    if (hipMalloc(&partial, (size_t)n * kHcSlices * (kHcMaxMix + 1) *
                                sizeof(float)) != hipSuccess)
      return nullptr;
    partial_tokens = n;
  }
  return partial;
}

int hy4_hc_pre(const float* h, const float* fn, const float* scale,
               const float* base, float* out, float* post, uint32_t n,
               uint32_t hc, uint32_t dim, float eps, float hc_eps,
               float magnitude, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (hc > 8)
    return 1;
  float* partial = HcPartial(n);
  if (!partial)
    return 1;
  const uint32_t n_mix = post ? 2 * hc : hc;
  constexpr uint32_t kTok = 8;
  const dim3 groups(kHcSlices, (n + kTok - 1) / kTok);
  if (n >= kTok && n_mix == 8)
    HcMixTokens<kTok, 8><<<groups, kBlock, 0, stream>>>(
        const_cast<float*>(h), fn, partial, hc * dim, n);
  else if (n >= kTok && n_mix == 4)
    HcMixTokens<kTok, 4><<<groups, kBlock, 0, stream>>>(
        const_cast<float*>(h), fn, partial, hc * dim, n);
  else
    HcMix<<<dim3(kHcSlices, n), kBlock, 0, stream>>>(h, fn, partial, hc * dim,
                                                     n_mix);
  HcApply<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(h, partial, scale, base, out, post, hc, dim, eps, hc_eps,
                                                         magnitude);
  return Status();
}

int hy4_hc_post_pre(const float* y, const float* post_prev, float* h,
                    const float* fn, const float* scale, const float* base,
                    float* out, float* post, uint32_t n, uint32_t hc,
                    uint32_t dim, float eps, float hc_eps, float magnitude,
                    hipStream_t stream) {
  if (n == 0)
    return 0;
  const uint32_t n_mix = post ? 2 * hc : hc;
  constexpr uint32_t kTok = 8;
  if (n < kTok || (n_mix != 8 && n_mix != 4) || hc > 8) {
    if (int rc = hy4_hc_post(y, h, post_prev, h, n, hc, dim, stream))
      return rc;
    return hy4_hc_pre(h, fn, scale, base, out, post, n, hc, dim, eps, hc_eps,
                      magnitude, stream);
  }
  float* partial = HcPartial(n);
  if (!partial)
    return 1;
  const dim3 groups(kHcSlices, (n + kTok - 1) / kTok);
  if (n_mix == 8)
    HcMixTokens<kTok, 8, true><<<groups, kBlock, 0, stream>>>(
        h, fn, partial, hc * dim, n, y, post_prev, hc, dim);
  else
    HcMixTokens<kTok, 4, true><<<groups, kBlock, 0, stream>>>(
        h, fn, partial, hc * dim, n, y, post_prev, hc, dim);
  HcApply<<<dim3(Blocks(dim, 64), n), kBlock, 0, stream>>>(
      h, partial, scale, base, out, post, hc, dim, eps, hc_eps, magnitude);
  return Status();
}

int hy4_hc_decode(const float* y, const float* y_norm, const float* post_prev,
                  float* h, const float* fn, const float* scale,
                  const float* base, float* x, float* out,
                  const float* out_norm, float* post, uint32_t n, uint32_t hc,
                  uint32_t dim, float eps, float hc_eps, float magnitude,
                  hipStream_t stream) {
  if (n == 0)
    return 0;
  if (hc > 8)
    return 1;
  float* partial = HcPartial(n);
  if (!partial)
    return 1;
  const uint32_t n_mix = post ? 2 * hc : hc;
  const dim3 slices(kHcSlices, n);
  if (hc == 4 && n_mix == 8)
    HcPostMixFast<8><<<slices, kBlock, 0, stream>>>(y, y_norm, post_prev, h, fn,
                                                    partial, hc, dim, eps);
  else if (hc == 4 && n_mix == 4)
    HcPostMixFast<4><<<slices, kBlock, 0, stream>>>(y, y_norm, post_prev, h, fn,
                                                    partial, hc, dim, eps);
  else
    HcPostMix<<<slices, kBlock, 0, stream>>>(y, y_norm, post_prev, h, fn,
                                             partial, hc, dim, n_mix, eps);
  constexpr uint32_t kPer = 12;
  if (hc == 4 && dim <= kPer * kBlock)
    HcApplyNormFast<4, kPer><<<dim3(1, n), kBlock, 0, stream>>>(
        h, partial, scale, base, x, out, out_norm, post, dim, eps, hc_eps,
        magnitude);
  else
    HcApplyNorm<<<dim3(1, n), kBlock, 0, stream>>>(h, partial, scale, base, x,
                                                   out, out_norm, post, hc, dim,
                                                   eps, hc_eps, magnitude);
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
  if (type != HY4_TYPE_F32 && type != HY4_TYPE_F16 && type != HY4_TYPE_BF16)
    return 1;
  if (type == HY4_TYPE_BF16) {
    if (k % (kBf16Split * 8))
      return 1;
    for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
      const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
      GemvBf16Split<<<m, kBf16Split * 32, 0, stream>>>(
          (const uint16_t*)w, x + (uint64_t)t0 * k, y + (uint64_t)t0 * m, m, k,
          nt);
    }
    return Status();
  }
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    const float* xt = x + (uint64_t)t0 * k;
    float* yt = y + (uint64_t)t0 * m;
    // Four rows per warp from two tokens on: the same sums, the activations
    // loaded once per four rows.
    const uint32_t rows = nt > 1 ? 4 : 1;
    const uint32_t blocks =
        (m + RowsPerBlock() * rows - 1) / (RowsPerBlock() * rows);
    if (type == HY4_TYPE_F32) {
      if (rows == 4)
        GemvFloat<float, 4>
            <<<blocks, kBlock, 0, stream>>>((const float*)w, xt, yt, m, k, nt);
      else
        GemvFloat<float, 1>
            <<<blocks, kBlock, 0, stream>>>((const float*)w, xt, yt, m, k, nt);
    } else {
      if (rows == 4)
        GemvFloat<__half, 4>
            <<<blocks, kBlock, 0, stream>>>((const __half*)w, xt, yt, m, k, nt);
      else
        GemvFloat<__half, 1>
            <<<blocks, kBlock, 0, stream>>>((const __half*)w, xt, yt, m, k, nt);
    }
  }
  return Status();
}

int hy4_gemv_q8_0_grouped(const void* w, const float* x, float* y, uint32_t m, uint32_t k, uint32_t groups,
                          uint32_t n, uint64_t x_tok, uint64_t x_grp, uint64_t y_tok, uint64_t y_grp,
                          hipStream_t stream) {
  if (k % 32)
    return 1;
  // Whole-block rows when rows start aligned and a group's inputs fit LDS.
  const bool rows_ok =
      k % 64 == 0 && m % (kBlock / 8) == 0 && (uintptr_t)w % 4 == 0 &&
      (uint64_t)kMaxTok * (k / 32) * kRowsXStride * sizeof(float) <= 32768;
  for (uint32_t t0 = 0; t0 < n; t0 += kMaxTok) {
    const uint32_t nt = n - t0 < kMaxTok ? n - t0 : kMaxTok;
    if (rows_ok) {
      GemvQ8_0Rows<<<dim3(m / (kBlock / 8), groups), kBlock,
                     nt*(k / 32) * kRowsXStride * sizeof(float), stream>>>(
          (const uint8_t*)w, x + t0 * x_tok, y + t0 * y_tok, m, k, nt, x_tok,
          x_grp, y_tok, y_grp);
      continue;
    }
    // Four rows per warp from two tokens on, as in hy4_gemv_float.
    const uint32_t rows = nt > 1 ? 4 : 1;
    const uint32_t blocks =
        (m + RowsPerBlock() * rows - 1) / (RowsPerBlock() * rows);
    if (rows == 4)
      GemvQ8_0Grouped<4><<<dim3(blocks, groups), kBlock, 0, stream>>>(
          (const BlockQ8_0*)w, x + t0 * x_tok, y + t0 * y_tok, m, k, nt, x_tok,
          x_grp, y_tok, y_grp);
    else
      GemvQ8_0Grouped<1><<<dim3(blocks, groups), kBlock, 0, stream>>>(
          (const BlockQ8_0*)w, x + t0 * x_tok, y + t0 * y_tok, m, k, nt, x_tok,
          x_grp, y_tok, y_grp);
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

uint64_t hy4_mla_attention_work(uint32_t n, uint32_t heads, uint32_t n_sel) {
  const uint64_t lists = ((uint64_t)n * n_sel + n + 63) / 64 * 64;
  return lists + (uint64_t)n * AttnSplits(heads) * heads * (2 + kAttnDv);
}

int hy4_mla_attention(const float* q, const uint16_t* cache, uint32_t ring,
                      const int32_t* pos, uint32_t window, const int32_t* sel,
                      uint32_t n_sel, const float* sinks, float scale,
                      float* out, uint32_t n, uint32_t heads, uint32_t dk,
                      uint32_t dv, float* work, uint64_t work_floats,
                      hipStream_t stream) {
  if (n == 0) return 0;
  if (dk != kAttnDk || dv != kAttnDv || heads % 4 != 0 ||
      work_floats < hy4_mla_attention_work(n, heads, sel ? n_sel : 0))
    return 1;
  int32_t* list = nullptr;
  uint32_t* count = nullptr;
  float* part = work;
  if (sel) {
    list = reinterpret_cast<int32_t*>(work);
    count = reinterpret_cast<uint32_t*>(work + (uint64_t)n * n_sel);
    part = work + ((uint64_t)n * n_sel + n + 63) / 64 * 64;
    AttnKeyList<<<n, kKeyListThreads, 0, stream>>>(sel, n_sel, pos, list,
                                                   count);
  }
  const uint32_t max_splits = AttnSplits(heads);
  float* part_ml = part;
  float* part_acc = part + (uint64_t)n * max_splits * heads * 2;
  const uint32_t groups = (heads + kAttnHeadsPerBlock - 1) / kAttnHeadsPerBlock;
  MlaDecode<<<dim3(max_splits, groups, n), kAttnWarps * 32, 0, stream>>>(
      q, reinterpret_cast<const _Float16*>(cache), ring, pos, window, list,
      count, n_sel, scale, part_ml, part_acc, heads);
  MlaCombine<<<dim3(heads, n), kAttnDv / 4, 0, stream>>>(
      part_ml, part_acc, pos, window, count, sinks, out, heads);
  return Status();
}

uint64_t hy4_dfly_attention_work(uint32_t n, uint32_t heads) {
  return (uint64_t)n * kDflyAttnSplits * heads * (2 + kAttnDv);
}

int hy4_dfly_attention(const float* q, const uint16_t* cache, uint32_t ring,
                       const int32_t* sel, uint32_t sel_count, int32_t limit,
                       const float* kva, const float* sinks, float scale,
                       float* out, uint32_t n, uint32_t heads, uint32_t dk,
                       uint32_t dv, float* work, uint64_t work_floats,
                       hipStream_t stream) {
  if (n == 0)
    return 0;
  if (dk != kAttnDk || dv != kAttnDv || heads % 4 != 0 ||
      work_floats < hy4_dfly_attention_work(n, heads))
    return 1;
  float* part_ml = work;
  float* part_acc = work + (uint64_t)n * kDflyAttnSplits * heads * 2;
  const uint32_t groups =
      (heads + kDflyAttnHeadsPerBlock - 1) / kDflyAttnHeadsPerBlock;
  DflyAttnDecode<<<dim3(kDflyAttnSplits, groups, n), kDflyAttnWarps * 32, 0,
                   stream>>>(q, reinterpret_cast<const _Float16*>(cache), ring,
                             sel, sel_count, limit, kva, scale, part_ml,
                             part_acc, n, heads);
  DflyAttnCombine<<<dim3(heads, n), kAttnDv / 4, 0, stream>>>(
      part_ml, part_acc, sel_count + n, sinks, out, heads);
  return Status();
}

int hy4_store_rows_half(const float* src, uint64_t src_stride, uint16_t* dst,
                        uint32_t ring, const int32_t* pos, uint32_t n,
                        uint32_t dim, hipStream_t stream) {
  if (n == 0)
    return 0;
  StoreRowsHalf<<<dim3(Blocks(dim, 16), n), kBlock, 0, stream>>>(
      src, src_stride, reinterpret_cast<_Float16*>(dst), ring, pos, dim);
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
  if (k > kBlock || k > n_experts)
    return 1;
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
  if (dim == kIdxDim && heads <= kIdxMaxHeads) {
    // Four tokens per block share their group rows' loads; with few blocks
    // of groups (short contexts, verify passes) a token per block spreads
    // the work.
    const uint32_t blocks = (n_pool_max + kBlock - 1) / kBlock;
    if (blocks * n >= 64)
      IndexerScoreRows<4><<<dim3(blocks, (n + 3) / 4), kBlock, 0, stream>>>(
          q, w, pooled, n_pool, n_pool_max, score, n, heads);
    else
      IndexerScoreRows<1><<<dim3(blocks, n), kBlock, 0, stream>>>(
          q, w, pooled, n_pool, n_pool_max, score, n, heads);
    return Status();
  }
  const uint32_t per_block = RowsPerBlock();
  IndexerScore<<<dim3((n_pool_max + per_block - 1) / per_block, n), kBlock, 0, stream>>>(q, w, pooled, n_pool,
                                                                                         n_pool_max, score, heads,
                                                                                         dim);
  return Status();
}

int hy4_kpool_select(const float* score, const int32_t* n_pool, uint32_t n_pool_max, const int32_t* pos, uint32_t k,
                     uint32_t kpool, int32_t* sel, uint32_t n, hipStream_t stream) {
  if (n == 0) return 0;
  if (k <= kSelMax)
    KpoolSelectSort<<<n, kBlock, 0, stream>>>(score, n_pool, n_pool_max, pos, k,
                                              kpool, sel, (k + 1) * kpool);
  else
    KpoolSelect<<<n, kBlock, 0, stream>>>((float*)score, n_pool, n_pool_max,
                                          pos, k, kpool, sel, (k + 1) * kpool);
  return Status();
}

int hy4_attn_softmax(float* scores, uint32_t n, uint32_t heads, uint32_t n_keys, int32_t key_pos0,
                     const int32_t* pos, uint32_t window, const float* mask, const float* sinks,
                     hipStream_t stream) {
  if (n == 0 || n_keys == 0) return 0;
  AttnSoftmax<<<n * heads, kBlock, 0, stream>>>(scores, heads, n_keys, key_pos0, pos, window, mask, sinks);
  return Status();
}

int hy4_gather_indexed(const uint16_t* src, uint32_t ring, const int32_t* idx,
                       uint32_t count, int32_t limit, uint32_t dim, float* dst,
                       float* mask, uint32_t rows, uint32_t n_keys,
                       hipStream_t stream) {
  if (count)
    GatherIndexed<<<dim3(1, count), kBlock, 0, stream>>>(
        reinterpret_cast<const _Float16*>(src), ring, idx, limit, dim, dst);
  if (rows && n_keys) {
    IndexMask<<<dim3((n_keys + kBlock - 1) / kBlock, rows), kBlock, 0, stream>>>(idx, count, limit, mask, n_keys);
  }
  return Status();
}

int hy4_mla_prep(float* qr, const float* q_norm, uint32_t q_lora, float* kva,
                 const float* kv_norm, uint32_t kv_lora, uint32_t dk,
                 const int32_t* pos, const int32_t* pos_hw, uint32_t n_rot,
                 float base, float eps, uint32_t n, hipStream_t stream) {
  if (n == 0)
    return 0;
  if ((pos_hw != nullptr && n_rot != 64) || kv_lora + n_rot > dk)
    return 1;
  MlaPrep<<<dim3(2, n), kBlock, 0, stream>>>(qr, q_norm, q_lora, kva, kv_norm,
                                             kv_lora, dk, pos, pos_hw, n_rot,
                                             base, eps);
  return Status();
}

int hy4_rope_3d_copy(float* x, const int32_t* pos, const int32_t* pos_hw,
                     uint32_t n, uint32_t heads, uint64_t tok_stride,
                     uint64_t head_stride, uint32_t offset, uint32_t n_rot,
                     float base, float* dst, uint64_t dst_tok,
                     uint64_t dst_head, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (pos_hw != nullptr && n_rot != 64)
    return 1;
  const uint32_t work = heads * (n_rot / 2);
  Rope<<<dim3((work + kBlock - 1) / kBlock, n), kBlock, 0, stream>>>(
      x, pos, pos_hw, heads, tok_stride, head_stride, offset, n_rot, base, dst,
      dst_tok, dst_head);
  return Status();
}

int hy4_copy_2d(float* dst, uint64_t dst_pitch, const float* src,
                uint64_t src_pitch, uint32_t width, uint64_t rows,
                hipStream_t stream) {
  const uint64_t count = rows * width;
  if (count == 0)
    return 0;
  Copy2d<<<Blocks(count), kBlock, 0, stream>>>(dst, dst_pitch, src, src_pitch,
                                               width, count);
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

int hy4_mla_prefill(const uint16_t* q, const uint16_t* keys, uint32_t ring,
                    int32_t lo, const int32_t* pos, uint32_t window,
                    const int32_t* sel, uint32_t n_sel, const float* sinks,
                    float scale, uint16_t* out, uint32_t n, uint32_t heads,
                    uint32_t dk, uint32_t dv, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (dk != kMlaDk || dv != kMlaDv || heads == 0 || heads % kMlaHeads != 0 ||
      ring == 0)
    return 1;
  const auto* qh = reinterpret_cast<const _Float16*>(q);
  const auto* kh = reinterpret_cast<const _Float16*>(keys);
  auto* oh = reinterpret_cast<_Float16*>(out);
  if (heads == 64) {
    MlaPrefill64<64><<<dim3(n, 1), 512, 0, stream>>>(
        qh, kh, ring, lo, pos, window, sel, n_sel, sinks, scale, oh, n);
  } else if (heads == 32 && window > 0 && sel == nullptr && n >= 2) {
    MlaPrefill64<32><<<dim3((n + 1) / 2, 1), 512, 0, stream>>>(
        qh, kh, ring, lo, pos, window, nullptr, 0, sinks, scale, oh, n);
  } else {
    MlaPrefill<<<dim3(n, heads / kMlaHeads), 32 * kMlaWaves, 0, stream>>>(
        qh, kh, ring, lo, pos, window, sel, n_sel, sinks, scale, oh, heads);
  }
  return Status();
}

int hy4_sgemm_nt(const float* x, const float* w, float* y, uint32_t n,
                 uint32_t m, uint32_t k, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (k % kSgK != 0)
    return 1;
  SgemmNt<float><<<dim3((m + kSgRows - 1) / kSgRows, (n + kSgTok - 1) / kSgTok),
                   256, 0, stream>>>(x, w, y, n, m, k);
  return Status();
}

int hy4_sgemm_nt_bf16(const float* x, const uint16_t* w, float* y, uint32_t n,
                      uint32_t m, uint32_t k, hipStream_t stream) {
  if (n == 0)
    return 0;
  if (k % kSgK != 0)
    return 1;
  SgemmNt<uint16_t>
      <<<dim3((m + kSgRows - 1) / kSgRows, (n + kSgTok - 1) / kSgTok), 256, 0,
         stream>>>(x, w, y, n, m, k);
  return Status();
}

int hy4_f32_to_bf16(const float* src, uint64_t count, uint16_t* dst,
                    int32_t* inexact, hipStream_t stream) {
  if (count == 0)
    return 0;
  const uint64_t blocks = (count + kBlock - 1) / kBlock;
  F32ToBf16<<<(uint32_t)(blocks < 4096 ? blocks : 4096), kBlock, 0, stream>>>(
      src, count, dst, inexact);
  return Status();
}

uint64_t hy4_argmax_work_floats(void) {
  return 2ull * kArgBlocks + 1;
}

int hy4_argmax(const float* x, uint32_t dim, float* work, int32_t* out,
               hipStream_t stream) {
  if (dim == 0)
    return 1;
  float* part_v = work;
  int32_t* part_i = reinterpret_cast<int32_t*>(work + kArgBlocks);
  const uint32_t chunk = (dim + kArgBlocks - 1) / kArgBlocks;
  ArgmaxRows<<<kArgBlocks, kBlock, 0, stream>>>(x, nullptr, dim, chunk, part_v,
                                                part_i);
  ArgmaxRows<<<1, kBlock, 0, stream>>>(part_v, part_i, kArgBlocks, kArgBlocks,
                                       part_v + 2 * kArgBlocks, out);
  return Status();
}

uint64_t hy4_argmax_work_floats_batch(uint32_t rows) {
  return 2ull * rows * kArgBlocks;
}

int hy4_argmax_rows(const float* x, uint32_t rows, uint32_t dim, float* work,
                    int32_t* out, hipStream_t stream) {
  if (rows == 0)
    return 0;
  if (dim == 0)
    return 1;
  float* part_v = work;
  int32_t* part_i =
      reinterpret_cast<int32_t*>(work + (uint64_t)rows * kArgBlocks);
  const uint32_t chunk = (dim + kArgBlocks - 1) / kArgBlocks;
  ArgmaxBatchPass1<<<dim3(kArgBlocks, rows), kBlock, 0, stream>>>(
      x, dim, chunk, part_v, part_i);
  ArgmaxBatchPass2<<<rows, kBlock, 0, stream>>>(part_v, part_i, out);
  return Status();
}

int hy4_sample_row(const float* logits, uint32_t dim, const uint32_t* mask,
                   float temperature, double u, void* scratch,
                   struct hy4_sample_result* out, hipStream_t stream) {
  if (!logits || !scratch || !out || dim == 0 ||
      dim > kSampleBlocks * kSampleThreads * kSamplePer)
    return 1;
  auto* s = static_cast<SampleScratch*>(scratch);
  SampleMaxKernel<<<kSampleBlocks, kSampleThreads, 0, stream>>>(logits, mask,
                                                                dim, s);
  if (temperature <= 0.0f) {
    SampleArgmaxFinishKernel<<<1, 1, 0, stream>>>(mask, s);
  } else {
    SampleSumKernel<<<kSampleBlocks, kSampleThreads, 0, stream>>>(
        logits, mask, dim, temperature, s);
    SamplePickKernel<<<1, kSampleThreads, 0, stream>>>(logits, mask, dim,
                                                       temperature, u, s);
  }
  if (hipMemcpyAsync(out, &s->result, sizeof(*out), hipMemcpyDeviceToHost,
                     stream) != hipSuccess ||
      hipStreamSynchronize(stream) != hipSuccess)
    return 1;
  return Status();
}

int hy4_quantize_q8_0_half(const uint16_t* src, uint64_t blocks, void* dst,
                           hipStream_t stream) {
  if (blocks == 0)
    return 0;
  QuantizeQ8Half<<<(blocks + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      reinterpret_cast<const _Float16*>(src), static_cast<BlockQ8_0*>(dst),
      blocks);
  return Status();
}

int hy4_quantize_q8_0_bf16(const uint16_t* src, uint64_t blocks, void* dst,
                           hipStream_t stream) {
  if (blocks == 0)
    return 0;
  QuantizeQ8Bf16<<<(blocks + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
      src, static_cast<BlockQ8_0*>(dst), blocks);
  return Status();
}

}  // extern "C"
