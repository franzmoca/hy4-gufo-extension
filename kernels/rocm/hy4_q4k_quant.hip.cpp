// Load-time GPU Q4_K quantization for the DFly2 drafter, adapted from Gufo's
// draft_head.hip.cpp (UPSTREAM.md). Built with contraction disabled so every
// operation rounds like ggml's single-precision quantize_row_q4_K_ref.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "hy4_kernels.h"

#pragma clang fp contract(off)

namespace {

constexpr unsigned kThreads = 256;

struct BlockQ8_0 {
  __half d;
  int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34);

struct BlockQ4K {
  __half d;
  __half dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == 144);

__device__ __forceinline__ int NearestInt(float value) {
  return static_cast<int>(__builtin_rintf(value));
}

__device__ __forceinline__ int Q4Code(float iscale, float x, float min_v) {
  return max(0, min(15, NearestInt(iscale * (x - min_v))));
}

// ggml make_qkx2_quants(32, 15, x, w, L, &the_min, Laux, -1, 0.1, 20, false).
__device__ float Qkx2Scale(const float (&x)[32], const float (&w)[32],
                           float* the_min, uint8_t (&L)[32]) {
  float min_v = x[0];
  float max_v = x[0];
  float sum_w = w[0];
  float sum_x = sum_w * x[0];
  for (int i = 1; i < 32; ++i) {
    if (x[i] < min_v)
      min_v = x[i];
    if (x[i] > max_v)
      max_v = x[i];
    sum_w += w[i];
    sum_x += w[i] * x[i];
  }
  if (min_v > 0.0f)
    min_v = 0.0f;
  if (max_v == min_v) {
    for (int i = 0; i < 32; ++i)
      L[i] = 0;
    *the_min = -min_v;
    return 0.0f;
  }
  float iscale = 15.0f / (max_v - min_v);
  float scale = 1.0f / iscale;
  float best_error = 0.0f;
  for (int i = 0; i < 32; ++i) {
    const int l = Q4Code(iscale, x[i], min_v);
    L[i] = static_cast<uint8_t>(l);
    const float diff = scale * static_cast<float>(l) + min_v - x[i];
    best_error += w[i] * (diff * diff);
  }
  for (int step = 0; step <= 20; ++step) {
    iscale =
        (-1.0f + 0.1f * static_cast<float>(step) + 15.0f) / (max_v - min_v);
    float sum_l = 0.0f, sum_l2 = 0.0f, sum_xl = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Q4Code(iscale, x[i], min_v));
      sum_l += w[i] * l;
      sum_l2 += w[i] * l * l;
      sum_xl += w[i] * l * x[i];
    }
    const float det = sum_w * sum_l2 - sum_l * sum_l;
    if (det <= 0.0f)
      continue;
    float this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
    float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
    if (this_min > 0.0f) {
      this_min = 0.0f;
      this_scale = sum_xl / sum_l2;
    }
    float error = 0.0f;
    for (int i = 0; i < 32; ++i) {
      const float l = static_cast<float>(Q4Code(iscale, x[i], min_v));
      const float diff = this_scale * l + this_min - x[i];
      error += w[i] * (diff * diff);
    }
    if (error < best_error) {
      for (int i = 0; i < 32; ++i)
        L[i] = static_cast<uint8_t>(Q4Code(iscale, x[i], min_v));
      best_error = error;
      scale = this_scale;
      min_v = this_min;
    }
  }
  *the_min = -min_v;
  return scale;
}

enum { kSrcHalf = 0, kSrcBf16 = 1, kSrcQ8 = 2 };

template<int kSrc>
__device__ __forceinline__ void LoadSubblock(const void* src, uint64_t subblock,
                                             float (&x)[32]) {
  if constexpr (kSrc == kSrcHalf) {
    const auto* p = static_cast<const __half*>(src) + subblock * 32;
    for (int i = 0; i < 32; ++i)
      x[i] = __half2float(p[i]);
  } else if constexpr (kSrc == kSrcBf16) {
    const auto* p = static_cast<const uint16_t*>(src) + subblock * 32;
    for (int i = 0; i < 32; ++i)
      x[i] = __uint_as_float(static_cast<uint32_t>(p[i]) << 16);
  } else {
    const BlockQ8_0& b = static_cast<const BlockQ8_0*>(src)[subblock];
    const float d = __half2float(b.d);
    for (int i = 0; i < 32; ++i)
      x[i] = static_cast<float>(b.qs[i]) * d;
  }
}

// One thread per 32-value sub-block; eight consecutive threads pack one Q4_K
// super-block of 256 values.
template<int kSrc>
__global__ void QuantizeToQ4KKernel(const void* __restrict__ src,
                                    BlockQ4K* __restrict__ q4k,
                                    uint64_t superblocks) {
  constexpr unsigned kPerBlock = kThreads / 8;
  __shared__ float scales_s[kPerBlock][8];
  __shared__ float mins_s[kPerBlock][8];
  __shared__ uint8_t codes_s[kPerBlock][256];
  const unsigned local = threadIdx.x / 8;
  const unsigned j = threadIdx.x % 8;
  const uint64_t superblock =
      static_cast<uint64_t>(blockIdx.x) * kPerBlock + local;
  const bool active = superblock < superblocks;
  float x[32];
  uint8_t codes[32];  // make_qkx2_quants' L, kept where the scale rounds to 0
  if (active) {
    LoadSubblock<kSrc>(src, superblock * 8 + j, x);
    float sum_x2 = 0.0f;
    for (int i = 0; i < 32; ++i)
      sum_x2 += x[i] * x[i];
    const float av_x = __builtin_sqrtf(sum_x2 / 32.0f);
    float w[32];
    for (int i = 0; i < 32; ++i)
      w[i] = av_x + __builtin_fabsf(x[i]);
    float the_min = 0.0f;
    scales_s[local][j] = Qkx2Scale(x, w, &the_min, codes);
    mins_s[local][j] = the_min;
  }
  __syncthreads();
  if (active) {
    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int k = 0; k < 8; ++k) {
      if (scales_s[local][k] > max_scale)
        max_scale = scales_s[local][k];
      if (mins_s[local][k] > max_min)
        max_min = mins_s[local][k];
    }
    const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    uint8_t packed[12] = {};
    for (int k = 0; k < 8; ++k) {
      const auto ls = static_cast<uint8_t>(
          min(63, max(0, NearestInt(inv_scale * scales_s[local][k]))));
      const auto lm = static_cast<uint8_t>(
          min(63, max(0, NearestInt(inv_min * mins_s[local][k]))));
      if (k < 4) {
        packed[k] = ls;
        packed[k + 4] = lm;
      } else {
        packed[k + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
        packed[k - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
        packed[k] |= static_cast<uint8_t>((lm >> 4) << 6);
      }
    }
    const __half d16 = __float2half_rn(max_scale / 63.0f);
    const __half dmin16 = __float2half_rn(max_min / 63.0f);
    BlockQ4K& out = q4k[superblock];
    if (j == 0) {
      out.d = d16;
      out.dmin = dmin16;
      for (int k = 0; k < 12; ++k)
        out.scales[k] = packed[k];
    }
    const uint8_t sc =
        j < 4 ? packed[j] & 63
              : (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4);
    const uint8_t m = j < 4 ? packed[j + 4] & 63
                            : (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    const float d = __half2float(d16) * static_cast<float>(sc);
    const float dm = __half2float(dmin16) * static_cast<float>(m);
    for (int i = 0; i < 32; ++i) {
      const int l =
          d != 0.0f ? max(0, min(15, NearestInt((x[i] + dm) / d))) : codes[i];
      codes_s[local][32 * j + i] = static_cast<uint8_t>(l);
    }
  }
  __syncthreads();
  if (active) {
    BlockQ4K& out = q4k[superblock];
    for (int i = 0; i < 16; ++i) {
      const int byte = 16 * static_cast<int>(j) + i;
      const int pair = byte / 32;
      const int l = byte % 32;
      out.qs[byte] =
          static_cast<uint8_t>(codes_s[local][64 * pair + l] |
                               (codes_s[local][64 * pair + 32 + l] << 4));
    }
  }
}

template<int kSrc>
int LaunchQ4K(const void* src, uint64_t superblocks, void* dst,
              hipStream_t stream) {
  if (superblocks == 0)
    return 0;
  constexpr uint64_t kPerBlock = kThreads / 8;
  const uint32_t grid =
      static_cast<uint32_t>((superblocks + kPerBlock - 1) / kPerBlock);
  QuantizeToQ4KKernel<kSrc><<<grid, kThreads, 0, stream>>>(
      src, static_cast<BlockQ4K*>(dst), superblocks);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

}  // namespace

extern "C" {

int hy4_quantize_q4_k_half(const uint16_t* src, uint64_t superblocks, void* dst,
                           hipStream_t stream) {
  return LaunchQ4K<kSrcHalf>(src, superblocks, dst, stream);
}

int hy4_quantize_q4_k_bf16(const uint16_t* src, uint64_t superblocks, void* dst,
                           hipStream_t stream) {
  return LaunchQ4K<kSrcBf16>(src, superblocks, dst, stream);
}

int hy4_quantize_q4_k_q8_0(const void* src, uint64_t superblocks, void* dst,
                           hipStream_t stream) {
  return LaunchQ4K<kSrcQ8>(src, superblocks, dst, stream);
}

}  // extern "C"
