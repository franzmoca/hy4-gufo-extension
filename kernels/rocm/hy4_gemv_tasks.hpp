// Device helpers for the decode GEMVs (hy4_gemv.hip.cpp): one lane's unit of
// quantized weights dequantized once and contracted with FP32 activations in
// a fixed FMA order. Copied from ds41-gufo-extension (UPSTREAM.md).
#ifndef HY4_GEMV_TASKS_HPP_
#define HY4_GEMV_TASKS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace hy4_gemv_tasks {

__device__ __forceinline__ float Half(const std::uint8_t* p) {
  return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ float HalfBits(std::uint32_t bits) {
  const auto h = static_cast<unsigned short>(bits & 0xFFFFU);
  return __half2float(*reinterpret_cast<const __half*>(&h));
}

/// ggml get_scale_min_k4 over the 12 scale bytes held as three words
/// (s[0..3], s[4..7], s[8..11]): 6-bit scale and min of 32-value sub-block
/// j, extracted with shifts instead of byte indexing.
__device__ __forceinline__ void ScaleMin(std::uint32_t w0, std::uint32_t w1,
                                         std::uint32_t w2, int j, float* sc,
                                         float* m) {
  const std::uint32_t shift = 8U * static_cast<std::uint32_t>(j & 3);
  const std::uint32_t a = (w0 >> shift) & 0xFFU;
  const std::uint32_t b = (w1 >> shift) & 0xFFU;
  const std::uint32_t c = (w2 >> shift) & 0xFFU;
  const bool low = j < 4;
  *sc = static_cast<float>(low ? (a & 63U) : ((c & 0xFU) | ((a >> 6) << 4)));
  *m = static_cast<float>(low ? (b & 63U) : ((c >> 4) | ((b >> 6) << 4)));
}

/// Converts the four bytes of `v` to float and applies scale * byte + offset.
__device__ __forceinline__ void DequantWord(std::uint32_t v, float scale,
                                            float offset, float* out) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    out[b] = __builtin_fmaf(scale, static_cast<float>((v >> (8 * b)) & 0xFFU),
                            offset);
  }
}

/// Dequantizes 16 values packed per byte (optional high bits) as
/// scale * v + offset, four bytes per word.
__device__ __forceinline__ void Dequant16(const std::uint8_t* q, int shift,
                                          const std::uint8_t* high, int hshift,
                                          int hmask, float scale, float offset,
                                          float* out) {
  std::uint32_t qw[4];
  __builtin_memcpy(qw, q, 16);
  std::uint32_t hw[4] = {};
  if (high != nullptr) {
    __builtin_memcpy(hw, high, 16);
  }
  const std::uint32_t mask = static_cast<std::uint32_t>(hmask) * 0x01010101U;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    std::uint32_t v = (qw[i] >> shift) & 0x0F0F0F0FU;
    if (high != nullptr) {
      v |= ((hw[i] >> hshift) & mask) << 4U;
    }
    DequantWord(v, scale, offset, out + 4 * i);
  }
}

/// acc += sum_i w[i] * x[i] as one FMA chain over n 64-byte-aligned values.
template<int N>
__device__ __forceinline__ float Accumulate(float acc, const float* w,
                                            const float* x) {
#pragma unroll
  for (int i = 0; i < N; i += 4) {
    const float4 f = *reinterpret_cast<const float4*>(x + i);
    acc = __builtin_fmaf(w[i], f.x, acc);
    acc = __builtin_fmaf(w[i + 1], f.y, acc);
    acc = __builtin_fmaf(w[i + 2], f.z, acc);
    acc = __builtin_fmaf(w[i + 3], f.w, acc);
  }
  return acc;
}

// Q4_K: 8 tasks per 256-value super-block. Task t covers qs bytes
// [32 p + 16 h, +16) with p = t / 2, h = t % 2: low nibbles are values
// 64 p + 16 h + i of sub-block 2 p, high nibbles 64 p + 32 + 16 h + i of
// sub-block 2 p + 1. Q5_K adds the fifth bit of value 16 h + i of pair p
// from bit 2 p (low) or 2 p + 1 (high) of qh[16 h + i]. The 16-byte header
// (d, dmin, 12 scale bytes) is one aligned load. w[16 r + i] belongs to
// activation 64 p + 16 h + 32 r + i.
/// The three 16-byte loads of a Q4_K/Q5_K task, so callers can issue them
/// ahead of the decode.
struct RawQ45K {
  uint4 header, q, h;
};

template<bool kFiveBit>
__device__ __forceinline__ RawQ45K LoadQ45K(const std::uint8_t* block, int t) {
  const int p = t >> 1;
  const int h = t & 1;
  const int qs = kFiveBit ? 48 : 16;
  RawQ45K raw{};
  raw.header = *reinterpret_cast<const uint4*>(block);
  raw.q = *reinterpret_cast<const uint4*>(block + qs + 32 * p + 16 * h);
  if constexpr (kFiveBit)
    raw.h = *reinterpret_cast<const uint4*>(block + 16 + 16 * h);
  return raw;
}

template<bool kFiveBit>
__device__ __forceinline__ void DecodeQ45K(const RawQ45K& raw, int t,
                                           float (&w)[32]) {
  const int p = t >> 1;
  const uint4 header = raw.header;
  const float d = HalfBits(header.x);
  const float dmin = HalfBits(header.x >> 16);
  float sc0, m0, sc1, m1;
  ScaleMin(header.y, header.z, header.w, 2 * p, &sc0, &m0);
  ScaleMin(header.y, header.z, header.w, 2 * p + 1, &sc1, &m1);
  const auto* q = reinterpret_cast<const std::uint8_t*>(&raw.q);
  const std::uint8_t* qh =
      kFiveBit ? reinterpret_cast<const std::uint8_t*>(&raw.h) : nullptr;
  Dequant16(q, 0, qh, 2 * p, 1, d * sc0, -(dmin * m0), w);
  Dequant16(q, 4, qh, 2 * p + 1, 1, d * sc1, -(dmin * m1), w + 16);
}

template<bool kFiveBit>
__device__ __forceinline__ void DecodeQ45K(const std::uint8_t* block, int t,
                                           float (&w)[32]) {
  DecodeQ45K<kFiveBit>(LoadQ45K<kFiveBit>(block, t), t, w);
}

/// Activation offset of task t's first 16 values in a Q4_K/Q5_K
/// super-block; the other 16 follow 32 values later.
__device__ __forceinline__ int OffsetQ45K(int t) {
  return 64 * (t >> 1) + 16 * (t & 1);
}

/// acc += task t's 32 decoded Q4_K/Q5_K values w . x (x: the super-block's
/// 256 activations).
__device__ __forceinline__ float DotQ45K(float acc, const float (&w)[32], int t,
                                         const float* x) {
  const float* xv = x + OffsetQ45K(t);
  acc = Accumulate<16>(acc, w, xv);
  return Accumulate<16>(acc, w + 16, xv + 32);
}

/// Q6_K in 8 tasks of 32 values: task t = 2 p + r (n = p / 2, h = p % 2)
/// covers ql bytes [64 n + 16 h, +16) and [64 n + 32 + 16 h, +16) and qh
/// bytes [32 n + 16 h, +16), their low (r = 0) or high (r = 1) nibbles with
/// qh bits 4 r and 4 r + 2, scales sc[8 n + h + 4 r + {0, 2}] and an offset
/// of 32. w[16 g + i] belongs to activation 128 n + 16 h + 64 r + 32 g + i.
/// The loads of a DecodeQ6K32 task, so callers can issue them ahead of the
/// decode: its two ql and one qh vectors, d and its two scales.
struct RawQ6K32 {
  uint4 a, b, h;
  float d;
  std::int8_t s0, s1;
};

__device__ __forceinline__ RawQ6K32 LoadQ6K32(const std::uint8_t* block,
                                              int t) {
  const int p = t >> 1;
  const int r = t & 1;
  const int n = p >> 1;
  const int h = p & 1;
  const auto* sc =
      reinterpret_cast<const std::int8_t*>(block + 192) + 8 * n + h + 4 * r;
  RawQ6K32 raw;
  raw.d = Half(block + 208);
  raw.s0 = sc[0];
  raw.s1 = sc[2];
  raw.a = *reinterpret_cast<const uint4*>(block + 64 * n + 16 * h);
  raw.b = *reinterpret_cast<const uint4*>(block + 64 * n + 32 + 16 * h);
  raw.h = *reinterpret_cast<const uint4*>(block + 128 + 32 * n + 16 * h);
  return raw;
}

__device__ __forceinline__ void DecodeQ6K32(const RawQ6K32& raw, int t,
                                            float (&w)[32]) {
  const int r = t & 1;
  const auto* ql_a = reinterpret_cast<const std::uint8_t*>(&raw.a);
  const auto* ql_b = reinterpret_cast<const std::uint8_t*>(&raw.b);
  const auto* qh = reinterpret_cast<const std::uint8_t*>(&raw.h);
  const float s0 = raw.d * static_cast<float>(raw.s0);
  const float s1 = raw.d * static_cast<float>(raw.s1);
  Dequant16(ql_a, 4 * r, qh, 4 * r, 3, s0, -32.0F * s0, w);
  Dequant16(ql_b, 4 * r, qh, 4 * r + 2, 3, s1, -32.0F * s1, w + 16);
}

__device__ __forceinline__ void DecodeQ6K32(const std::uint8_t* block, int t,
                                            float (&w)[32]) {
  DecodeQ6K32(LoadQ6K32(block, t), t, w);
}

__device__ __forceinline__ int OffsetQ6K32(int t) {
  return 128 * (t >> 2) + 16 * ((t >> 1) & 1) + 64 * (t & 1);
}

/// Q8_0 block (34 bytes, 2-byte aligned inside word-aligned rows): values
/// d * q. `odd` says the block starts two bytes past a word boundary; the
/// nine enclosing words are loaded and the payload selected with byte
/// alignment.
__device__ __forceinline__ void DecodeQ8_0(const std::uint8_t* block, bool odd,
                                           float (&w)[32]) {
  const auto* words =
      reinterpret_cast<const std::uint32_t*>(block - (odd ? 2 : 0));
  std::uint32_t r[9];
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    r[i] = words[i];
  }
  const float d = HalfBits(odd ? r[0] >> 16 : r[0]);
  std::uint32_t qs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    qs[i] = odd ? r[i + 1] : __builtin_amdgcn_alignbyte(r[i + 1], r[i], 2);
  }
  const auto* q = reinterpret_cast<const std::int8_t*>(qs);
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    w[i] = d * static_cast<float>(q[i]);
  }
}

}  // namespace hy4_gemv_tasks

#endif  // HY4_GEMV_TASKS_HPP_
