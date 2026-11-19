#include "kernels/rocm/hy4_vision.h"

#include <hip/hip_runtime.h>
#include <math.h>

namespace {

constexpr uint32_t kThreads = 256;

int Status() {
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

uint32_t Blocks(uint64_t count) {
  const uint64_t blocks = (count + kThreads - 1) / kThreads;
  return blocks > 65535u * 16u ? 65535u * 16u : static_cast<uint32_t>(blocks);
}

__device__ float BlockReduce(float v, float* scratch, bool max) {
  for (int o = warpSize / 2; o > 0; o /= 2) {
    const float other = __shfl_xor(v, o);
    v = max ? fmaxf(v, other) : v + other;
  }
  const uint32_t warp = threadIdx.x / warpSize;
  const uint32_t lane = threadIdx.x % warpSize;
  const uint32_t warps = (blockDim.x + warpSize - 1) / warpSize;
  __syncthreads();
  if (lane == 0)
    scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = lane < warps ? scratch[lane] : (max ? -INFINITY : 0.0f);
    for (int o = warpSize / 2; o > 0; o /= 2) {
      const float other = __shfl_xor(v, o);
      v = max ? fmaxf(v, other) : v + other;
    }
    if (lane == 0)
      scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void LayerNorm(const float* x, const float* weight,
                          const float* bias, float* y, uint32_t dim,
                          float eps) {
  __shared__ float scratch[32];
  const float* xr = x + (uint64_t)blockIdx.x * dim;
  float* yr = y + (uint64_t)blockIdx.x * dim;
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
    sum += xr[i];
  const float mean = BlockReduce(sum, scratch, false) / (float)dim;
  float var = 0.0f;
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = xr[i] - mean;
    var += d * d;
  }
  const float scale =
      rsqrtf(BlockReduce(var, scratch, false) / (float)dim + eps);
  for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
    yr[i] = (xr[i] - mean) * scale * weight[i] + bias[i];
}

__global__ void Bias(float* y, const float* bias, uint64_t count, uint32_t dim,
                     int gelu) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    float v = y[i] + bias[i % dim];
    if (gelu)
      v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
    y[i] = v;
  }
}

__global__ void Residual(float* x, const float* y, const float* bias,
                         uint64_t count, uint32_t dim) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    x[i] += y[i] + (bias ? bias[i % dim] : 0.0f);
  }
}

__global__ void MergeGather(const float* x, uint32_t gw, uint32_t dim,
                            float* out, uint64_t count) {
  const uint32_t ow = gw / 2;
  const uint32_t width = dim * 4;
  for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < count;
       k += (uint64_t)gridDim.x * blockDim.x) {
    const uint64_t row = k / width;
    const uint32_t col = (uint32_t)(k % width);
    const uint32_t ic = col / 4;
    const uint32_t kh = (col / 2) % 2;
    const uint32_t kw = col % 2;
    const uint64_t i = row / ow;
    const uint64_t j = row % ow;
    out[k] = x[((2 * i + kh) * gw + 2 * j + kw) * dim + ic];
  }
}

__global__ void Newline(const float* src, const float* newline, float* dst,
                        uint32_t ow, uint32_t dim, uint64_t count) {
  for (uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; k < count;
       k += (uint64_t)gridDim.x * blockDim.x) {
    const uint64_t row = k / dim;
    const uint32_t d = (uint32_t)(k % dim);
    const uint64_t i = row / (ow + 1);
    const uint32_t j = (uint32_t)(row % (ow + 1));
    dst[k] = j == ow ? newline[d] : src[(i * ow + j) * dim + d];
  }
}

// ------------------------------------------------------------- attention
// Fused bidirectional attention on the matrix cores, adapted from
// gemma4-gufo-extension's vision encoder (FusedAttention): the same 72-wide
// heads padded to 80, with the 1/sqrt(72) scale folded into Q and FP32 rows
// out.
constexpr uint32_t kHeads = 16;
constexpr uint32_t kHeadDim = 72;
constexpr uint32_t kHidden = kHeads * kHeadDim;
constexpr uint32_t kPaddedDim = 80;
constexpr uint32_t kDimSteps = kPaddedDim / 16;
constexpr uint32_t kAttentionWaves = 4;
constexpr uint32_t kKeyTile = 64;

using Half16 = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using Float8 = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ _Float16 ToHalf(float v) {
  return static_cast<_Float16>(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// q, k [head][padded][80] and vt [head][80][padded] from FP32 rows; the
// padding is zeroed by the caller.
__global__ void AttnPack(const float* q, const float* k, const float* v,
                         float scale, _Float16* qh, _Float16* kh, _Float16* vt,
                         uint32_t count, uint32_t padded) {
  const uint64_t total = (uint64_t)count * kHidden;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < total;
       i += (uint64_t)gridDim.x * blockDim.x) {
    const uint32_t token = (uint32_t)(i / kHidden);
    const uint32_t head = (uint32_t)(i % kHidden) / kHeadDim;
    const uint32_t d = (uint32_t)(i % kHeadDim);
    const uint64_t row = ((uint64_t)head * padded + token) * kPaddedDim + d;
    qh[row] = ToHalf(q[i] * scale);
    kh[row] = ToHalf(k[i]);
    vt[((uint64_t)head * kPaddedDim + d) * padded + token] = ToHalf(v[i]);
  }
}

__device__ __forceinline__ Float8 Wmma(Half16 a, Half16 b, Float8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

__device__ __forceinline__ Half16 LoadHalf16(const _Float16* p) {
  union {
    Half16 h;
    uint4 u[2];
  } value;
  value.u[0] = *reinterpret_cast<const uint4*>(p);
  value.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return value.h;
}

// Each wave owns 16 queries of one head and walks 64-key tiles with an online
// softmax: binary16 Q K^T on the matrix cores, FP32 softmax state, binary16
// probabilities times V accumulated in FP32. Fragment layout (wave32
// v_wmma_f32_16x16x16_f16): A holds row L%16 and 16 contiguous k, B column
// L%16 and 16 contiguous k; accumulator element i is row 2i + L/16, column
// L%16.
__global__ void __launch_bounds__(32 * kAttentionWaves)
    FusedAttention(const _Float16* q, const _Float16* k, const _Float16* vt,
                   float* out, uint32_t count, uint32_t padded) {
  __shared__ __align__(16) _Float16 p_lds[kAttentionWaves][16][kKeyTile + 8];
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t wave = threadIdx.x / 32;
  const uint32_t head = blockIdx.y;
  const uint32_t q0 = (blockIdx.x * kAttentionWaves + wave) * 16;
  if (q0 >= count)
    return;
  const uint32_t row = lane % 16;
  const uint32_t half = lane / 16;
  const _Float16* qh = q + (uint64_t)head * padded * kPaddedDim;
  const _Float16* kh = k + (uint64_t)head * padded * kPaddedDim;
  const _Float16* vh = vt + (uint64_t)head * kPaddedDim * padded;
  Half16 query[kDimSteps];
#pragma unroll
  for (uint32_t s = 0; s < kDimSteps; ++s) {
    query[s] = LoadHalf16(qh + (uint64_t)(q0 + row) * kPaddedDim + s * 16);
  }
  Float8 acc[kDimSteps] = {};
  float m[8], l[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.0f;
  }
  constexpr uint32_t kBlocks = kKeyTile / 16;
  for (uint32_t key0 = 0; key0 < count; key0 += kKeyTile) {
    Float8 score[kBlocks] = {};
#pragma unroll
    for (uint32_t b = 0; b < kBlocks; ++b) {
#pragma unroll
      for (uint32_t s = 0; s < kDimSteps; ++s) {
        score[b] =
            Wmma(query[s],
                 LoadHalf16(kh + (uint64_t)(key0 + b * 16 + row) * kPaddedDim +
                            s * 16),
                 score[b]);
      }
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      float x[kBlocks];
      float maximum = -INFINITY;
#pragma unroll
      for (uint32_t b = 0; b < kBlocks; ++b) {
        x[b] = key0 + b * 16 + row < count ? score[b][i] : -INFINITY;
        maximum = fmaxf(maximum, x[b]);
      }
#pragma unroll
      for (uint32_t offset = 8; offset > 0; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
      }
      const float next = fmaxf(m[i], maximum);
      const float alpha = __expf(m[i] - next);
      float sum = 0.0f;
#pragma unroll
      for (uint32_t b = 0; b < kBlocks; ++b) {
        const float p = __expf(x[b] - next);
        sum += p;
        p_lds[wave][2 * i + half][b * 16 + row] = static_cast<_Float16>(p);
      }
#pragma unroll
      for (uint32_t offset = 8; offset > 0; offset /= 2) {
        sum += __shfl_xor(sum, offset, 32);
      }
      l[i] = l[i] * alpha + sum;
      m[i] = next;
#pragma unroll
      for (uint32_t c = 0; c < kDimSteps; ++c)
        acc[c][i] *= alpha;
    }
    // A wave's LDS accesses complete in order; the barriers only keep the
    // compiler from moving them across the exchange.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#pragma unroll
    for (uint32_t b = 0; b < kBlocks; ++b) {
      const Half16 probabilities = LoadHalf16(&p_lds[wave][row][b * 16]);
#pragma unroll
      for (uint32_t c = 0; c < kDimSteps; ++c) {
        acc[c] = Wmma(
            probabilities,
            LoadHalf16(vh + (uint64_t)(c * 16 + row) * padded + key0 + b * 16),
            acc[c]);
      }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
  }
#pragma unroll
  for (uint32_t c = 0; c < kDimSteps; ++c) {
    const uint32_t d = c * 16 + row;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t token = q0 + 2 * i + half;
      if (token < count && d < kHeadDim) {
        out[(uint64_t)token * kHidden + head * kHeadDim + d] = acc[c][i] / l[i];
      }
    }
  }
}

__global__ void ToHalfKernel(const float* x, _Float16* y, uint64_t count) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count;
       i += (uint64_t)gridDim.x * blockDim.x) {
    y[i] = ToHalf(x[i]);
  }
}

}  // namespace

uint32_t hy4_vis_attn_padded(uint32_t count) {
  return (count + kKeyTile - 1) / kKeyTile * kKeyTile;
}

int hy4_vis_attention(const float* q, const float* k, const float* v,
                      float scale, uint16_t* qh, uint16_t* kh, uint16_t* vt,
                      float* out, uint32_t count, hipStream_t stream) {
  if (count == 0)
    return 0;
  const uint32_t padded = hy4_vis_attn_padded(count);
  const uint64_t halves = (uint64_t)kHeads * padded * kPaddedDim;
  if (hipMemsetAsync(qh, 0, halves * 2, stream) != hipSuccess ||
      hipMemsetAsync(kh, 0, halves * 2, stream) != hipSuccess ||
      hipMemsetAsync(vt, 0, halves * 2, stream) != hipSuccess) {
    return 1;
  }
  auto* q16 = reinterpret_cast<_Float16*>(qh);
  auto* k16 = reinterpret_cast<_Float16*>(kh);
  auto* v16 = reinterpret_cast<_Float16*>(vt);
  AttnPack<<<Blocks((uint64_t)count * kHidden), kThreads, 0, stream>>>(
      q, k, v, scale, q16, k16, v16, count, padded);
  const uint32_t per_block = 16 * kAttentionWaves;
  FusedAttention<<<dim3((count + per_block - 1) / per_block, kHeads),
                   32 * kAttentionWaves, 0, stream>>>(q16, k16, v16, out, count,
                                                      padded);
  return Status();
}

int hy4_vis_to_half(const float* x, uint16_t* y, uint64_t count,
                    hipStream_t stream) {
  if (count == 0)
    return 0;
  ToHalfKernel<<<Blocks(count), kThreads, 0, stream>>>(
      x, reinterpret_cast<_Float16*>(y), count);
  return Status();
}

int hy4_vis_layer_norm(const float* x, const float* weight, const float* bias,
                       float* y, uint32_t rows, uint32_t dim, float eps,
                       hipStream_t stream) {
  if (rows == 0)
    return 0;
  LayerNorm<<<rows, kThreads, 0, stream>>>(x, weight, bias, y, dim, eps);
  return Status();
}

int hy4_vis_bias(float* y, const float* bias, uint32_t rows, uint32_t dim,
                 int gelu, hipStream_t stream) {
  const uint64_t count = (uint64_t)rows * dim;
  if (count == 0)
    return 0;
  Bias<<<Blocks(count), kThreads, 0, stream>>>(y, bias, count, dim, gelu);
  return Status();
}

int hy4_vis_residual(float* x, const float* y, const float* bias, uint32_t rows,
                     uint32_t dim, hipStream_t stream) {
  const uint64_t count = (uint64_t)rows * dim;
  if (count == 0)
    return 0;
  Residual<<<Blocks(count), kThreads, 0, stream>>>(x, y, bias, count, dim);
  return Status();
}

int hy4_vis_merge_gather(const float* x, uint32_t gh, uint32_t gw, uint32_t dim,
                         float* out, hipStream_t stream) {
  const uint64_t count = (uint64_t)(gh / 2) * (gw / 2) * dim * 4;
  if (count == 0)
    return 0;
  MergeGather<<<Blocks(count), kThreads, 0, stream>>>(x, gw, dim, out, count);
  return Status();
}

int hy4_vis_newline(const float* src, const float* newline, float* dst,
                    uint32_t oh, uint32_t ow, uint32_t dim,
                    hipStream_t stream) {
  const uint64_t count = (uint64_t)oh * (ow + 1) * dim;
  if (count == 0)
    return 0;
  Newline<<<Blocks(count), kThreads, 0, stream>>>(src, newline, dst, ow, dim,
                                                  count);
  return Status();
}
