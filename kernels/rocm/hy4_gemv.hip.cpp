/* Decode projections (1 to 16 tokens) for dense Q8_0, Q4_K and Q6_K
 * weights: weights stay in their GGUF blocks, activations stay FP32, so no
 * pass quantizes its input. Copied from ds41-gufo-extension (UPSTREAM.md);
 * launches take a stream and the caller's split scratch. A lane's unit of
 * work is one 16-byte vector of quantized values; lanes
 * holding the same task of different rows read the same activations, which
 * the hardware broadcasts. A workgroup's eight waves form 8 / S row groups
 * of S waves; the S waves of a group take interleaved super-blocks of its
 * rows and are summed in wave order. */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "hy4_gemv_tasks.hpp"
#include "hy4_kernels.h"

namespace {

using namespace hy4_gemv_tasks;

constexpr int kWave = 32;
constexpr int kWaves = 8;

template<int kType>
struct Format;
template<>
struct Format<12> {  // Q4_K
  static constexpr int kBlockBytes = 144;
  static constexpr int kTasks = 8;
  struct Raw {
    uint4 header, q;
  };
  using SplitRaw = Raw;
  static __device__ __forceinline__ Raw Load(const uint8_t* b, int t) {
    const int p = t >> 1;
    const int h = t & 1;
    return {*reinterpret_cast<const uint4*>(b),
            *reinterpret_cast<const uint4*>(b + 16 + 32 * p + 16 * h)};
  }
  static __device__ __forceinline__ Raw LoadSplit(const uint8_t* b, int t) {
    return Load(b, t);
  }
  static __device__ __forceinline__ void DecodeSplit(const Raw& r, int t,
                                                     float (&w)[32]) {
    Decode(r, t, w);
  }
  static __device__ __forceinline__ void Decode(const Raw& r, int t,
                                                float (&w)[32]) {
    RawQ45K q{};
    q.header = r.header;
    q.q = r.q;
    DecodeQ45K<false>(q, t, w);
  }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t,
                                                float (&w)[32]) {
    Decode(Load(b, t), t, w);
  }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32],
                                              int t, const float* x) {
    return DotQ45K(acc, w, t, x);
  }
};
template<>
struct Format<14> {  // Q6_K
  static constexpr int kBlockBytes = 210;
  static constexpr int kTasks = 8;
  using SplitRaw = RawQ6K32;
  static __device__ __forceinline__ SplitRaw LoadSplit(const uint8_t* b,
                                                       int t) {
    return LoadQ6K32(b, t);
  }
  static __device__ __forceinline__ void DecodeSplit(const SplitRaw& r, int t,
                                                     float (&w)[32]) {
    DecodeQ6K32(r, t, w);
  }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t,
                                                float (&w)[32]) {
    DecodeQ6K32(b, t, w);
  }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32],
                                              int t, const float* x) {
    const float* xv = x + OffsetQ6K32(t);
    acc = Accumulate<16>(acc, w, xv);
    return Accumulate<16>(acc, w + 16, xv + 32);
  }
};
struct RawQ8_0 {
  uint32_t r[9];
};
template<>
struct Format<8> {  // Q8_0, eight 34-byte blocks per 256 values
  static constexpr int kBlockBytes = 272;
  static constexpr int kTasks = 8;
  /* The split kernel's register ring: the nine words around block t, decoded
   * as DecodeQ8_0 does. */
  struct SplitRaw : RawQ8_0 {};
  static __device__ __forceinline__ SplitRaw LoadSplit(const uint8_t* b,
                                                       int t) {
    const auto* words =
        reinterpret_cast<const uint32_t*>(b + 34 * t - (t & 1 ? 2 : 0));
    SplitRaw r;
#pragma unroll
    for (int i = 0; i < 9; ++i)
      r.r[i] = words[i];
    return r;
  }
  static __device__ __forceinline__ void DecodeSplit(const SplitRaw& raw, int t,
                                                     float (&w)[32]) {
    const bool odd = (t & 1) != 0;
    const float d = HalfBits(odd ? raw.r[0] >> 16 : raw.r[0]);
    uint32_t qs[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      qs[i] = odd ? raw.r[i + 1]
                  : __builtin_amdgcn_alignbyte(raw.r[i + 1], raw.r[i], 2);
    const auto* q = reinterpret_cast<const int8_t*>(qs);
#pragma unroll
    for (int i = 0; i < 32; ++i)
      w[i] = d * static_cast<float>(q[i]);
  }
  static __device__ __forceinline__ void Decode(const uint8_t* b, int t,
                                                float (&w)[32]) {
    DecodeQ8_0(b + 34 * t, (t & 1) != 0, w);
  }
  static __device__ __forceinline__ float Dot(float acc, const float (&w)[32],
                                              int t, const float* x) {
    return Accumulate<32>(acc, w, x + 32 * t);
  }
};

/* Row r of token c reads x + c * x_stride + (r / rows_per_x) * k, for the
 * first n <= kCols tokens. Rows from m1 on come from w2 and go to y2 (row
 * r - m1, token stride m - m1); the others to y (token stride y_stride). With
 * kSwiGLU, w_up holds a second matrix of the same shape whose rows are taken
 * in the same pass, and row r stores silu(min(gate, limit)) * clamp(up,
 * +-limit). With add (y's layout), rows stored to y are acc + add; with
 * gate, y[row] is scaled by sigmoid(acc) instead (Gate). Each lane
 * decodes its weights once for all tokens; every token sees the FMA order of
 * a single-token pass. */
/* y scaled by the sigmoid of a gate, as hy4_sigmoid_mul computes it. */
__device__ __forceinline__ float Gate(float y, float g) {
  return y * (1.0f / (1.0f + expf(-g)));
}

template<int kType, int S, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    gemv_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ w_up,
                float limit, const float* __restrict__ x, uint64_t x_stride,
                float* __restrict__ y, uint32_t y_stride, uint32_t m,
                uint32_t k, uint32_t rows_per_x, const uint8_t* __restrict__ w2,
                float* __restrict__ y2, uint32_t m1, uint32_t n,
                const float* __restrict__ add, bool gate) {
  using F = Format<kType>;
  constexpr int kRowsPerWave = kWave / F::kTasks;
  constexpr int kGroups = kWaves / S;
  __shared__ float partial[S > 1 ? kCols : 1][kWaves][kWave];
  __shared__ float partial_up[S > 1 && kSwiGLU ? kCols : 1][kWaves][kWave];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int group = wave / S;
  const int slice = wave % S;
  const int t = lane % F::kTasks;
  const uint32_t row =
      (blockIdx.x * kGroups + group) * kRowsPerWave + lane / F::kTasks;
  const uint32_t blocks_per_row = k / 256;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  const bool second = row >= m1;
  if (row < m) {
    const uint64_t offset =
        (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
    const uint8_t* base = (second ? w2 : w) + offset;
    const float* xr = x + (uint64_t)(row / rows_per_x) * k;
    if constexpr (kCols > 1 && requires { typename F::Raw; }) {
      /* Several tokens make each block's FMAs long; the next block's
       * weights load before this block is decoded. */
      const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
      typename F::Raw next{}, next_up{};
      if (slice < (int)blocks_per_row) {
        next = F::Load(base + (uint64_t)slice * F::kBlockBytes, t);
        if constexpr (kSwiGLU)
          next_up = F::Load(base_up + (uint64_t)slice * F::kBlockBytes, t);
      }
      for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
        const typename F::Raw raw = next, raw_up = next_up;
        if (blk + S < blocks_per_row) {
          next = F::Load(base + (uint64_t)(blk + S) * F::kBlockBytes, t);
          if constexpr (kSwiGLU)
            next_up =
                F::Load(base_up + (uint64_t)(blk + S) * F::kBlockBytes, t);
        }
        float wv[32];
        F::Decode(raw, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (c < (int)n)
            acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
        if constexpr (kSwiGLU) {
          F::Decode(raw_up, t, wv);
#pragma unroll
          for (int c = 0; c < kCols; ++c)
            if (c < (int)n)
              acc_up[c] =
                  F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
        }
      }
    } else
#pragma unroll 2
      for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
        float wv[32];
        F::Decode(base + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          if (kCols == 1 || c < (int)n)
            acc[c] = F::Dot(acc[c], wv, t, xr + c * x_stride + blk * 256);
        if constexpr (kSwiGLU) {
          F::Decode(w_up + offset + (uint64_t)blk * F::kBlockBytes, t, wv);
#pragma unroll
          for (int c = 0; c < kCols; ++c)
            if (kCols == 1 || c < (int)n)
              acc_up[c] =
                  F::Dot(acc_up[c], wv, t, xr + c * x_stride + blk * 256);
        }
      }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int offset = F::kTasks / 2; offset > 0; offset >>= 1) {
      acc[c] += __shfl_xor(acc[c], offset, kWave);
      if constexpr (kSwiGLU)
        acc_up[c] += __shfl_xor(acc_up[c], offset, kWave);
    }
  }
  if constexpr (S > 1) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      partial[c][wave][lane] = acc[c];
      if constexpr (kSwiGLU)
        partial_up[c][wave][lane] = acc_up[c];
    }
    __syncthreads();
    if (slice != 0)
      return;
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
#pragma unroll
      for (int s = 1; s < S; ++s) {
        acc[c] += partial[c][wave + s][lane];
        if constexpr (kSwiGLU)
          acc_up[c] += partial_up[c][wave + s][lane];
      }
    }
  }
  if (t != 0 || row >= m)
    return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (kCols > 1 && c >= (int)n)
      break;
    if constexpr (kSwiGLU) {
      float g = acc[c];
      float u = acc_up[c];
      if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
      }
      y[(uint64_t)c * y_stride + row] = g / (1.0f + expf(-g)) * u;
    } else if (second) {
      y2[(uint64_t)c * (m - m1) + row - m1] = acc[c];
    } else {
      const uint64_t at = (uint64_t)c * y_stride + row;
      y[at] = gate ? Gate(y[at], acc[c]) : add ? acc[c] + add[at] : acc[c];
    }
  }
}

/* Several tokens over Q5_K rows. The lane layout above re-reads each
 * block's activations in every wave, eight tasks per load, and at 8 tokens
 * those vector loads, not the weights, bound it. Here wave w takes task w of
 * 32 rows (one per lane), so a wave's activations are uniform and come
 * through scalar loads; the vector unit loads only weights. Workgroup
 * (tile, slice) runs blocks slice, slice + S, ... in order, as a slice wave
 * above does, reduces the tasks in its xor-tree order
 * ((a0 + a4) + (a2 + a6)) + ((a1 + a5) + (a3 + a7)) and stores the slice
 * result; the tile's last workgroup (atomic counter) adds the S slice
 * results in wave order. With kSwiGLU, w_up's rows run in the same pass
 * (their slice results follow the gate's in part) and the last workgroup
 * stores the SwiGLU as the kernel above does. Every row and token matches
 * the kernel above bit for bit. */
template<int kType, int S, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    split_gemv_kernel(const uint8_t* __restrict__ w,
                      const uint8_t* __restrict__ w_up, float limit,
                      const float* __restrict__ x, uint64_t x_stride,
                      float* __restrict__ y, uint32_t y_stride, uint32_t m,
                      uint32_t k, uint32_t rows_per_x,
                      const uint8_t* __restrict__ w2, float* __restrict__ y2,
                      uint32_t m1, uint32_t n, float* __restrict__ part,
                      uint32_t* __restrict__ counter,
                      const float* __restrict__ add, bool gate) {
  using F = Format<kType>;
  static_assert(F::kTasks == kWaves, "one wave per task");
  constexpr int kMats = kSwiGLU ? 2 : 1;
  __shared__ float red[kWaves / 2][kMats][kCols][kWave];
  const int lane = threadIdx.x % kWave;
  const int t = __builtin_amdgcn_readfirstlane(threadIdx.x / kWave);
  const uint32_t tile = blockIdx.x / S;
  const uint32_t slice = blockIdx.x % S;
  const uint32_t row0 = tile * kWave;
  const uint32_t row = min(row0 + lane, m - 1);
  const uint32_t blocks_per_row = k / 256;
  // m1 is a multiple of 32, so a workgroup's rows come from one matrix.
  const bool second = row0 >= m1;
  const uint64_t offset =
      (uint64_t)(second ? row - m1 : row) * blocks_per_row * F::kBlockBytes;
  const uint8_t* base[kMats];
  base[0] = (second ? w2 : w) + offset;
  if constexpr (kSwiGLU)
    base[kMats - 1] = w_up + offset;
  // rows_per_x is m or a multiple of 32, so a workgroup shares its x row.
  const float* xr = x + (uint64_t)(row0 / rows_per_x) * k;
  float acc[kMats][kCols] = {};
  typename F::SplitRaw next[kMats] = {};
  if (slice < blocks_per_row)
#pragma unroll
    for (int i = 0; i < kMats; ++i)
      next[i] = F::LoadSplit(base[i] + (uint64_t)slice * F::kBlockBytes, t);
  for (uint32_t blk = slice; blk < blocks_per_row; blk += S) {
    typename F::SplitRaw raw[kMats];
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
      raw[i] = next[i];
      if (blk + S < blocks_per_row)
        next[i] =
            F::LoadSplit(base[i] + (uint64_t)(blk + S) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
      float wv[32];
      F::DecodeSplit(raw[i], t, wv);
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        if (c < (int)n)
          acc[i][c] = F::Dot(acc[i][c], wv, t, xr + c * x_stride + blk * 256);
    }
  }
  /* Xor tree over the tasks: partner offset 4, then 2, then 1. */
#pragma unroll
  for (int offset = kWaves / 2; offset > 0; offset >>= 1) {
    if (t >= offset && t < 2 * offset)
#pragma unroll
      for (int i = 0; i < kMats; ++i)
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          red[t - offset][i][c][lane] = acc[i][c];
    __syncthreads();
    if (t < offset)
#pragma unroll
      for (int i = 0; i < kMats; ++i)
#pragma unroll
        for (int c = 0; c < kCols; ++c)
          acc[i][c] += red[t][i][c][lane];
    __syncthreads();
  }
  if (t != 0)
    return;
  const bool valid = row0 + lane < m;  // part rows are exactly m apart
#pragma unroll
  for (int i = 0; i < kMats; ++i)
#pragma unroll
    for (int c = 0; c < kCols; ++c)
      if (c < (int)n && valid)
        part[(((uint64_t)i * S + slice) * kCols + c) * m + row0 + lane] =
            acc[i][c];
  __threadfence();
  uint32_t done = 0;
  if (lane == 0)
    done = atomicAdd(&counter[tile], 1u);
  if (__shfl(done, 0, kWave) != S - 1)
    return;
  __threadfence();
  if (lane == 0)
    counter[tile] = 0;  // ready for the next launch
  if (!valid)
    return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (c >= (int)n)
      break;
    float total[kMats];
#pragma unroll
    for (int i = 0; i < kMats; ++i) {
#pragma unroll
      for (int s = 0; s < S; ++s) {
        const float v = __hip_atomic_load(
            &part[(((uint64_t)i * S + s) * kCols + c) * m + row0 + lane],
            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        total[i] = s ? total[i] + v : v;
      }
    }
    if constexpr (kSwiGLU) {
      float g = total[0];
      float u = total[1];
      if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
      }
      y[(uint64_t)c * y_stride + row0 + lane] = g / (1.0f + expf(-g)) * u;
    } else if (second) {
      y2[(uint64_t)c * (m - m1) + row0 + lane - m1] = total[0];
    } else {
      const uint64_t at = (uint64_t)c * y_stride + row0 + lane;
      y[at] = gate  ? Gate(y[at], total[0])
              : add ? total[0] + add[at]
                    : total[0];
    }
  }
}

/* Routed experts with FP32 activations, adapted from ds41-gufo-extension
 * (UPSTREAM.md). One block of kPlanSlots threads groups `ids[0..n_slots)` by
 * expert: plan[0] is the group count, and group g at plan + 1 + g * kPlanGroup
 * holds {expert_id, count, slot[0..count-1]} in ascending slot order. Every
 * routed kernel sums super-blocks 0..blocks_per_row-1 in order and reduces the
 * 8 tasks in xor-tree order (4, 2, 1), so single-slot, lane-major and
 * task-major launches agree bit for bit. */
constexpr int kPlanSlots = HY4_EXPERT_PLAN_SLOTS;
constexpr int kPlanCols = HY4_GEMV_MAX_TOKENS;
constexpr int kPlanGroup = 2 + kPlanCols;

__global__ void expert_plan_kernel(const int32_t* __restrict__ ids,
                                   uint32_t slots, int32_t* __restrict__ plan) {
  __shared__ int32_t sid[kPlanSlots];
  __shared__ int32_t first_of[kPlanSlots];
  __shared__ int32_t group_of[kPlanSlots];
  const uint32_t s = threadIdx.x;
  if (s < slots)
    sid[s] = ids[s];
  __syncthreads();
  int32_t id = 0, col = 0, count = 0, first = (int32_t)s;
  if (s < slots) {
    id = sid[s];
    for (uint32_t o = 0; o < slots; ++o) {
      const bool same = sid[o] == id;
      count += same;
      if (o < s && same) {
        ++col;
        if ((int32_t)o < first)
          first = (int32_t)o;
      }
    }
    first_of[s] = first;
  }
  __syncthreads();
  if (s < slots && first == (int32_t)s) {
    int32_t group = 0;
    for (uint32_t o = 0; o < s; ++o)
      group += first_of[o] == (int32_t)o;
    group_of[s] = group;
  }
  __syncthreads();
  if (s >= slots)
    return;
  int32_t* g = plan + 1 + group_of[first] * kPlanGroup;
  if (first == (int32_t)s) {
    g[0] = id;
    g[1] = count;
  }
  if (col < kPlanCols)
    g[2 + col] = (int32_t)s;
  if (s == slots - 1) {
    int32_t groups = 0;
    for (uint32_t o = 0; o < slots; ++o)
      groups += first_of[o] == (int32_t)o;
    plan[0] = groups;
  }
}

/* Single-slot routed expert GEMV (n = 1): 32 rows per workgroup (4 rows x 8
 * tasks per wave, no LDS), loading U super-blocks ahead to hide DRAM latency.
 */
template<int kType, bool kSwiGLU, int U>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_single_kernel(const uint8_t* __restrict__ w,
                         const uint8_t* __restrict__ w_up,
                         uint64_t expert_bytes, float limit,
                         const float* __restrict__ x, uint32_t slots_per_x,
                         const int32_t* __restrict__ ids,
                         const float* __restrict__ route_w,
                         float* __restrict__ y, uint32_t m, uint32_t k) {
  using F = Format<kType>;
  constexpr uint32_t kRows = (kWave / F::kTasks) * kWaves;
  const uint32_t slot = blockIdx.y;
  const int32_t expert = ids[slot];
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % F::kTasks;
  const uint32_t row =
      blockIdx.x * kRows + wave * (kWave / F::kTasks) + lane / F::kTasks;
  if (row >= m)
    return;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t offset = (uint64_t)expert * expert_bytes +
                          (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  const float* xr = x + (uint64_t)(slot / slots_per_x) * k;
  float acc = 0.0f;
  float acc_up = 0.0f;
  uint32_t blk = 0;
  for (; blk + U <= blocks_per_row; blk += U) {
    typename F::SplitRaw raw[U];
    typename F::SplitRaw raw_up[kSwiGLU ? U : 1];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      raw[u] = F::LoadSplit(base + (uint64_t)(blk + u) * F::kBlockBytes, t);
      if constexpr (kSwiGLU)
        raw_up[u] =
            F::LoadSplit(base_up + (uint64_t)(blk + u) * F::kBlockBytes, t);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float wv[32];
      F::DecodeSplit(raw[u], t, wv);
      acc = F::Dot(acc, wv, t, xr + (blk + u) * 256);
      if constexpr (kSwiGLU) {
        F::DecodeSplit(raw_up[u], t, wv);
        acc_up = F::Dot(acc_up, wv, t, xr + (blk + u) * 256);
      }
    }
  }
  for (; blk < blocks_per_row; ++blk) {
    float wv[32];
    F::DecodeSplit(F::LoadSplit(base + (uint64_t)blk * F::kBlockBytes, t), t,
                   wv);
    acc = F::Dot(acc, wv, t, xr + blk * 256);
    if constexpr (kSwiGLU) {
      F::DecodeSplit(F::LoadSplit(base_up + (uint64_t)blk * F::kBlockBytes, t),
                     t, wv);
      acc_up = F::Dot(acc_up, wv, t, xr + blk * 256);
    }
  }
#pragma unroll
  for (int o = F::kTasks / 2; o > 0; o >>= 1) {
    acc += __shfl_xor(acc, o, kWave);
    if constexpr (kSwiGLU)
      acc_up += __shfl_xor(acc_up, o, kWave);
  }
  if (t != 0)
    return;
  float v = acc;
  if constexpr (kSwiGLU) {
    float g = acc;
    float u = acc_up;
    if (limit > 0.0f) {
      u = fminf(fmaxf(u, -limit), limit);
      g = fminf(g, limit);
    }
    v = g / (1.0f + expf(-g)) * u;
    if (route_w)
      v *= route_w[slot];
  }
  y[(uint64_t)slot * m + row] = v;
}

/* Lane-major grouped expert GEMV (2..8 tokens): uniform 32-bit activation
 * offsets in SGPRs, 32 rows per workgroup (no LDS), weights decoded once per
 * expert group across its slots. */
template<int kType, bool kSwiGLU, int kCols>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_group_lane_kernel(const uint8_t* __restrict__ w,
                             const uint8_t* __restrict__ w_up,
                             uint64_t expert_bytes, float limit,
                             const float* __restrict__ x, uint32_t slots_per_x,
                             const int32_t* __restrict__ plan,
                             const float* __restrict__ route_w,
                             float* __restrict__ y, uint32_t m, uint32_t k) {
  using F = Format<kType>;
  constexpr uint32_t kRows = (kWave / F::kTasks) * kWaves;
  if ((int32_t)blockIdx.y >= plan[0])
    return;
  const int32_t* g = plan + 1 + blockIdx.y * kPlanGroup;
  const int32_t expert = g[0];
  const int cols = g[1] < kCols ? g[1] : kCols;
  const int lane = threadIdx.x % kWave;
  const int wave = threadIdx.x / kWave;
  const int t = lane % F::kTasks;
  const uint32_t row =
      blockIdx.x * kRows + wave * (kWave / F::kTasks) + lane / F::kTasks;
  if (row >= m)
    return;
  uint32_t x_off[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c)
    x_off[c] = (uint32_t)((c < cols ? g[2 + c] : 0) / slots_per_x) * k;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t offset = (uint64_t)expert * expert_bytes +
                          (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
#pragma unroll 2
  for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
    const float* xb = x + blk * 256;
    float wv[32];
    F::DecodeSplit(F::LoadSplit(base + (uint64_t)blk * F::kBlockBytes, t), t,
                   wv);
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      if (c >= cols)
        break;
      acc[c] = F::Dot(acc[c], wv, t, xb + x_off[c]);
    }
    if constexpr (kSwiGLU) {
      F::DecodeSplit(F::LoadSplit(base_up + (uint64_t)blk * F::kBlockBytes, t),
                     t, wv);
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        if (c >= cols)
          break;
        acc_up[c] = F::Dot(acc_up[c], wv, t, xb + x_off[c]);
      }
    }
  }
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
#pragma unroll
    for (int o = F::kTasks / 2; o > 0; o >>= 1) {
      acc[c] += __shfl_xor(acc[c], o, kWave);
      if constexpr (kSwiGLU)
        acc_up[c] += __shfl_xor(acc_up[c], o, kWave);
    }
  }
  if (t != 0)
    return;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    if (c >= cols)
      break;
    const int32_t slot = g[2 + c];
    float v = acc[c];
    if constexpr (kSwiGLU) {
      float gv = acc[c];
      float uv = acc_up[c];
      if (limit > 0.0f) {
        uv = fminf(fmaxf(uv, -limit), limit);
        gv = fminf(gv, limit);
      }
      v = gv / (1.0f + expf(-gv)) * uv;
      if (route_w)
        v *= route_w[slot];
    }
    y[(uint64_t)slot * m + row] = v;
  }
}

/* Task-major grouped expert GEMV (9..16 tokens): wave t takes task t of 32
 * rows (lane = row), so activations are uniform across the wave and arrive
 * through scalar loads while the vector unit loads only weights. The 8 task
 * waves reduce through LDS in xor-tree order (4, 2, 1), matching the lane
 * kernels bit for bit. */
template<int kType, bool kSwiGLU, int kCols, bool kPrefetch>
__global__ void __launch_bounds__(kWave* kWaves)
    expert_group_task_kernel(const uint8_t* __restrict__ w,
                             const uint8_t* __restrict__ w_up,
                             uint64_t expert_bytes, float limit,
                             const float* __restrict__ x, uint32_t slots_per_x,
                             const int32_t* __restrict__ plan,
                             const float* __restrict__ route_w,
                             float* __restrict__ y, uint32_t m, uint32_t k) {
  using F = Format<kType>;
  static_assert(F::kTasks == kWaves, "one wave per task");
  constexpr uint32_t kRows = kWave;
  constexpr int kMats = kSwiGLU ? 2 : 1;
  constexpr int kRedCols = kCols < 4 ? kCols : 4;
  __shared__ float red[kWaves / 2][kMats][kRedCols][kWave];
  if ((int32_t)blockIdx.y >= plan[0])
    return;
  const int32_t* g = plan + 1 + blockIdx.y * kPlanGroup;
  const int32_t expert = g[0];
  const int cols = g[1] < kCols ? g[1] : kCols;
  const int lane = threadIdx.x % kWave;
  const int t = __builtin_amdgcn_readfirstlane(threadIdx.x / kWave);
  const uint32_t row0 = blockIdx.x * kRows;
  const uint32_t row = min(row0 + lane, m - 1);
  uint32_t x_off[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c)
    x_off[c] = (uint32_t)((c < cols ? g[2 + c] : 0) / slots_per_x) * k;
  const uint32_t blocks_per_row = k / 256;
  const uint64_t offset = (uint64_t)expert * expert_bytes +
                          (uint64_t)row * blocks_per_row * F::kBlockBytes;
  const uint8_t* base = w + offset;
  const uint8_t* base_up = kSwiGLU ? w_up + offset : nullptr;
  float acc[kCols] = {};
  float acc_up[kCols] = {};
  if constexpr (kPrefetch) {
    typename F::SplitRaw next = F::LoadSplit(base, t);
    typename F::SplitRaw next_up{};
    if constexpr (kSwiGLU)
      next_up = F::LoadSplit(base_up, t);
    for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      const typename F::SplitRaw raw = next, raw_up = next_up;
      if (blk + 1 < blocks_per_row) {
        next = F::LoadSplit(base + (uint64_t)(blk + 1) * F::kBlockBytes, t);
        if constexpr (kSwiGLU)
          next_up =
              F::LoadSplit(base_up + (uint64_t)(blk + 1) * F::kBlockBytes, t);
      }
      const float* xb = x + blk * 256;
      float wv[32];
      F::DecodeSplit(raw, t, wv);
      for (int c = 0; c < kCols; ++c) {
        if (c >= cols)
          break;
        acc[c] = F::Dot(acc[c], wv, t, xb + x_off[c]);
      }
      if constexpr (kSwiGLU) {
        F::DecodeSplit(raw_up, t, wv);
        for (int c = 0; c < kCols; ++c) {
          if (c >= cols)
            break;
          acc_up[c] = F::Dot(acc_up[c], wv, t, xb + x_off[c]);
        }
      }
    }
  } else {
    for (uint32_t blk = 0; blk < blocks_per_row; ++blk) {
      const float* xb = x + blk * 256;
      float wv[32];
      F::DecodeSplit(F::LoadSplit(base + (uint64_t)blk * F::kBlockBytes, t), t,
                     wv);
      for (int c = 0; c < kCols; ++c) {
        if (c >= cols)
          break;
        acc[c] = F::Dot(acc[c], wv, t, xb + x_off[c]);
      }
      if constexpr (kSwiGLU) {
        F::DecodeSplit(
            F::LoadSplit(base_up + (uint64_t)blk * F::kBlockBytes, t), t, wv);
        for (int c = 0; c < kCols; ++c) {
          if (c >= cols)
            break;
          acc_up[c] = F::Dot(acc_up[c], wv, t, xb + x_off[c]);
        }
      }
    }
  }
  for (int c0 = 0; c0 < kCols; c0 += kRedCols) {
#pragma unroll
    for (int offset = kWaves / 2; offset > 0; offset >>= 1) {
      if (t >= offset && t < 2 * offset) {
#pragma unroll
        for (int j = 0; j < kRedCols; ++j) {
          if (c0 + j < kCols) {
            red[t - offset][0][j][lane] = acc[c0 + j];
            if constexpr (kSwiGLU)
              red[t - offset][1][j][lane] = acc_up[c0 + j];
          }
        }
      }
      __syncthreads();
      if (t < offset) {
#pragma unroll
        for (int j = 0; j < kRedCols; ++j) {
          if (c0 + j < kCols) {
            acc[c0 + j] += red[t][0][j][lane];
            if constexpr (kSwiGLU)
              acc_up[c0 + j] += red[t][1][j][lane];
          }
        }
      }
      __syncthreads();
    }
  }
  if (t != 0 || row0 + lane >= m)
    return;
  for (int c = 0; c < kCols; ++c) {
    if (c >= cols)
      break;
    const int32_t slot = g[2 + c];
    float v = acc[c];
    if constexpr (kSwiGLU) {
      float gv = acc[c];
      float uv = acc_up[c];
      if (limit > 0.0f) {
        uv = fminf(fmaxf(uv, -limit), limit);
        gv = fminf(gv, limit);
      }
      v = gv / (1.0f + expf(-gv)) * uv;
      if (route_w)
        v *= route_w[slot];
    }
    y[(uint64_t)slot * m + row0 + lane] = v;
  }
}

template<int kType, int kCols>
void launch_experts_lane(const void* w, const void* w_up, uint64_t expert_bytes,
                         float limit, const float* x, uint32_t slots_per_x,
                         const int32_t* plan, const float* route_w, float* y,
                         uint32_t m, uint32_t k, uint32_t n_slots,
                         hipStream_t stream) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * kWaves;
  const dim3 grid((m + kRows - 1) / kRows, n_slots);
  const auto* wg = static_cast<const uint8_t*>(w);
  const auto* wu = static_cast<const uint8_t*>(w_up);
  if (w_up)
    expert_group_lane_kernel<kType, true, kCols>
        <<<grid, kWave * kWaves, 0, stream>>>(wg, wu, expert_bytes, limit, x,
                                              slots_per_x, plan, route_w, y, m,
                                              k);
  else
    expert_group_lane_kernel<kType, false, kCols>
        <<<grid, kWave * kWaves, 0, stream>>>(wg, nullptr, expert_bytes, 0.0f,
                                              x, slots_per_x, plan, nullptr, y,
                                              m, k);
}

template<int kType>
void dispatch_experts(const void* w, const void* w_up, uint64_t expert_bytes,
                      float limit, const float* x, uint32_t slots_per_x,
                      const int32_t* ids, const int32_t* plan,
                      const float* route_w, float* y, uint32_t m, uint32_t k,
                      uint32_t n_slots, uint32_t max_cols, hipStream_t stream) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * kWaves;
  const dim3 grid((m + kRows - 1) / kRows, n_slots);
  const auto* wg = static_cast<const uint8_t*>(w);
  const auto* wu = static_cast<const uint8_t*>(w_up);
  if (max_cols == 1) {
    constexpr int kUnroll = kType == 8 ? 1 : 3;
    if (w_up)
      expert_single_kernel<kType, true, kUnroll>
          <<<grid, kWave * kWaves, 0, stream>>>(wg, wu, expert_bytes, limit, x,
                                                slots_per_x, ids, route_w, y, m,
                                                k);
    else
      expert_single_kernel<kType, false, kUnroll>
          <<<grid, kWave * kWaves, 0, stream>>>(wg, nullptr, expert_bytes, 0.0f,
                                                x, slots_per_x, ids, nullptr, y,
                                                m, k);
    return;
  }
  if (max_cols == 2)
    launch_experts_lane<kType, 2>(w, w_up, expert_bytes, limit, x, slots_per_x,
                                  plan, route_w, y, m, k, n_slots, stream);
  else if (max_cols <= 4)
    launch_experts_lane<kType, 4>(w, w_up, expert_bytes, limit, x, slots_per_x,
                                  plan, route_w, y, m, k, n_slots, stream);
  else if (max_cols <= 6)
    launch_experts_lane<kType, 6>(w, w_up, expert_bytes, limit, x, slots_per_x,
                                  plan, route_w, y, m, k, n_slots, stream);
  else if (max_cols <= 8)
    launch_experts_lane<kType, 8>(w, w_up, expert_bytes, limit, x, slots_per_x,
                                  plan, route_w, y, m, k, n_slots, stream);
  else {
    constexpr bool kPrefetch = kType == 14;
    if (w_up)
      expert_group_task_kernel<kType, true, kPlanCols, kPrefetch>
          <<<grid, kWave * kWaves, 0, stream>>>(wg, wu, expert_bytes, limit, x,
                                                slots_per_x, plan, route_w, y,
                                                m, k);
    else
      expert_group_task_kernel<kType, false, kPlanCols, kPrefetch>
          <<<grid, kWave * kWaves, 0, stream>>>(wg, nullptr, expert_bytes, 0.0f,
                                                x, slots_per_x, plan, nullptr,
                                                y, m, k);
  }
}

/* Slice results and per-tile counters of split_gemv_kernel, in the caller's
 * scratch (hy4_gemv_scratch_bytes): the launches that share one run on one
 * stream, in turn. Without scratch every launch takes the lane layout, which
 * computes the same bits. */
constexpr uint64_t kSplitPartFloats = 16ull * 4 * 65536;
constexpr uint64_t kSplitCounters = kSplitPartFloats / 32;
struct SplitScratch {
  float* part = nullptr;
  uint32_t* counter = nullptr;
};

/* The second matrix of a pair (w2, y2 from row m1), or none (m1 = m). */
struct Second {
  const void* w;
  float* y;
  uint32_t m1;
};

/* Tokens of one launch: x rows x_stride floats apart, y rows y_stride apart;
 * add (NULL or y's layout) is added to y's rows as they are stored, and with
 * gate the rows scale y by their sigmoid instead. */
struct Tokens {
  uint32_t n;
  uint64_t x_stride;
  uint32_t y_stride;
  const float* add = nullptr;
  bool gate = false;
};

template<int kType, int S, int kCols>
void launch(const void* w, const void* w_up, float limit, const float* x,
            float* y, uint32_t m, uint32_t k, uint32_t rows_per_x,
            Second second, Tokens tokens, SplitScratch scratch,
            hipStream_t stream) {
  constexpr uint32_t kRows = (kWave / Format<kType>::kTasks) * (kWaves / S);
  const uint32_t blocks = (m + kRows - 1) / kRows;
  const auto* w2 = static_cast<const uint8_t*>(second.w);
  if constexpr (kCols > 1) {
    /* Plain Q4_K rows from 2 tokens, plain and paired Q6_K/Q8_0 rows from 4,
     * the fused SwiGLU from 6; below those the lane layout is as fast or
     * faster. */
    const bool split = w_up ? kCols >= 6 : kCols >= (kType == 12 ? 2 : 4);
    if (split && second.m1 % kWave == 0 &&
        (rows_per_x == m || rows_per_x % kWave == 0) &&
        (uint64_t)m * S * kCols * (w_up ? 2 : 1) <= kSplitPartFloats) {
      if (scratch.part) {
        const auto* wu = static_cast<const uint8_t*>(w_up);
        const auto* wg = static_cast<const uint8_t*>(w);
        const uint32_t grid = (m + kWave - 1) / kWave * S;
        if (w_up)
          split_gemv_kernel<kType, S, true, kCols>
              <<<grid, kWave * kWaves, 0, stream>>>(
                  wg, wu, limit, x, tokens.x_stride, y, tokens.y_stride, m, k,
                  rows_per_x, w2, second.y, second.m1, tokens.n, scratch.part,
                  scratch.counter, tokens.add, tokens.gate);
        else
          split_gemv_kernel<kType, S, false, kCols>
              <<<grid, kWave * kWaves, 0, stream>>>(
                  wg, nullptr, 0.0f, x, tokens.x_stride, y, tokens.y_stride, m,
                  k, rows_per_x, w2, second.y, second.m1, tokens.n,
                  scratch.part, scratch.counter, tokens.add, tokens.gate);
        return;
      }
    }
  }
  if (w_up)
    gemv_kernel<kType, S, true, kCols><<<blocks, kWave * kWaves, 0, stream>>>(
        static_cast<const uint8_t*>(w), static_cast<const uint8_t*>(w_up),
        limit, x, tokens.x_stride, y, tokens.y_stride, m, k, rows_per_x, w2,
        second.y, second.m1, tokens.n, tokens.add, tokens.gate);
  else
    gemv_kernel<kType, S, false, kCols><<<blocks, kWave * kWaves, 0, stream>>>(
        static_cast<const uint8_t*>(w), nullptr, 0.0f, x, tokens.x_stride, y,
        tokens.y_stride, m, k, rows_per_x, w2, second.y, second.m1, tokens.n,
        tokens.add, tokens.gate);
}

template<int kType, int kCols>
bool dispatch_slices(const void* w, const void* w_up, float limit,
                     const float* x, float* y, uint32_t m, uint32_t k,
                     uint32_t rows_per_x, int slices, Second second,
                     Tokens tokens, SplitScratch scratch, hipStream_t stream) {
  switch (slices) {
    case 2:
      launch<kType, 2, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second,
                              tokens, scratch, stream);
      return true;
    case 4:
      launch<kType, 4, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second,
                              tokens, scratch, stream);
      return true;
    case 8:
      launch<kType, 8, kCols>(w, w_up, limit, x, y, m, k, rows_per_x, second,
                              tokens, scratch, stream);
      return true;
    default:
      return false;
  }
}

template<int kType>
int dispatch(const void* w, const void* w_up, float limit, const float* x,
             float* y, uint32_t m, uint32_t k, uint32_t rows_per_x, int slices,
             Second second, Tokens tokens, SplitScratch scratch,
             hipStream_t stream) {
  bool ok = false;
  if (tokens.n == 1)
    ok = dispatch_slices<kType, 1>(w, w_up, limit, x, y, m, k, rows_per_x,
                                   slices, second, tokens, scratch, stream);
  else if (tokens.n == 2)
    ok = dispatch_slices<kType, 2>(w, w_up, limit, x, y, m, k, rows_per_x,
                                   slices, second, tokens, scratch, stream);
  else if (tokens.n <= 4)
    ok = dispatch_slices<kType, 4>(w, w_up, limit, x, y, m, k, rows_per_x,
                                   slices, second, tokens, scratch, stream);
  else if (tokens.n <= 6)
    ok = dispatch_slices<kType, 6>(w, w_up, limit, x, y, m, k, rows_per_x,
                                   slices, second, tokens, scratch, stream);
  else if (tokens.n <= 8)
    ok = dispatch_slices<kType, 8>(w, w_up, limit, x, y, m, k, rows_per_x,
                                   slices, second, tokens, scratch, stream);
  else if (tokens.n <= 16)
    ok = dispatch_slices<kType, 16>(w, w_up, limit, x, y, m, k, rows_per_x,
                                    slices, second, tokens, scratch, stream);
  if (!ok)
    return 1;
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "hy4: gemv launch failed: %s\n", hipGetErrorString(err));
  return 1;
}

}  // namespace

namespace {

int run(int type, const void* w, const void* w_up, float limit, const float* x,
        float* y, uint32_t m, uint32_t k, Second second, Tokens tokens,
        void* scratch, hipStream_t stream) {
  if (k % 256 != 0 || m == 0 || tokens.n == 0 || tokens.n > HY4_GEMV_MAX_TOKENS)
    return 1;
  if (!second.w)
    second.m1 = m;
  SplitScratch split;
  if (scratch) {
    split.part = static_cast<float*>(scratch);
    split.counter = reinterpret_cast<uint32_t*>(split.part + kSplitPartFloats);
  }
  // Slice count depends only on (k, m) so single-token and multi-token passes
  // sum identically: 4 slices when k <= 1024 (4 super-blocks/row, avoiding
  // empty slices on wq_b), 4/2 slices on vocab-sized heads (m > 32768 / 65536)
  // so split_gemv_kernel stays within the 16 MiB scratch up to 16 tokens, and
  // 8 slices elsewhere.
  const int slices = k <= 1024 ? 4 : m > 65536 ? 2 : m > 32768 ? 4 : 8;
  switch (type) {
    case HY4_TYPE_Q8_0:
      return dispatch<8>(w, w_up, limit, x, y, m, k, m, slices, second, tokens,
                         split, stream);
    case HY4_TYPE_Q4_K:
      return dispatch<12>(w, w_up, limit, x, y, m, k, m, slices, second, tokens,
                          split, stream);
    case HY4_TYPE_Q6_K:
      return dispatch<14>(w, w_up, limit, x, y, m, k, m, slices, second, tokens,
                          split, stream);
    default:
      return 1;
  }
}

}  // namespace

extern "C" uint64_t hy4_gemv_scratch_bytes(void) {
  return kSplitPartFloats * sizeof(float) + kSplitCounters * sizeof(uint32_t);
}

extern "C" int hy4_gemv(int type, const void* w, const float* x, float* y,
                        uint32_t m, uint32_t k, uint32_t n, void* scratch,
                        hipStream_t stream) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, Second{nullptr, nullptr, 0},
             Tokens{n, k, m}, scratch, stream);
}

extern "C" int hy4_gemv_add(int type, const void* w, const float* x,
                            const float* add, float* y, uint32_t m, uint32_t k,
                            uint32_t n, void* scratch, hipStream_t stream) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, Second{nullptr, nullptr, 0},
             Tokens{n, k, m, add}, scratch, stream);
}

extern "C" int hy4_gemv_gate(int type, const void* w, const float* x, float* y,
                             uint32_t m, uint32_t k, uint32_t n, void* scratch,
                             hipStream_t stream) {
  return run(type, w, nullptr, 0.0f, x, y, m, k, Second{nullptr, nullptr, 0},
             Tokens{n, k, m, nullptr, true}, scratch, stream);
}

extern "C" uint64_t hy4_expert_plan_ints(void) {
  return 1ull + (uint64_t)kPlanSlots * kPlanGroup;
}

extern "C" int hy4_expert_plan(const int32_t* ids, uint32_t n_slots,
                               int32_t* plan, hipStream_t stream) {
  if (!ids || !plan || n_slots == 0 || n_slots > kPlanSlots)
    return 1;
  expert_plan_kernel<<<1, kPlanSlots, 0, stream>>>(ids, n_slots, plan);
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "hy4: expert plan launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}

extern "C" int hy4_gemv_experts(int type, const void* w, const void* w_up,
                                uint64_t expert_bytes, const float* x,
                                uint32_t slots_per_x, const int32_t* ids,
                                const int32_t* plan, const float* route_w,
                                float limit, float* y, uint32_t m, uint32_t k,
                                uint32_t n_slots, uint32_t max_cols,
                                hipStream_t stream) {
  if (k % 256 != 0 || m == 0 || n_slots == 0 || slots_per_x == 0 ||
      max_cols == 0 || max_cols > HY4_GEMV_MAX_TOKENS ||
      (max_cols == 1 && !ids) ||
      (max_cols > 1 && (!plan || n_slots > kPlanSlots)))
    return 1;
  switch (type) {
    case HY4_TYPE_Q8_0:
      dispatch_experts<8>(w, w_up, expert_bytes, limit, x, slots_per_x, ids,
                          plan, route_w, y, m, k, n_slots, max_cols, stream);
      break;
    case HY4_TYPE_Q4_K:
      dispatch_experts<12>(w, w_up, expert_bytes, limit, x, slots_per_x, ids,
                           plan, route_w, y, m, k, n_slots, max_cols, stream);
      break;
    case HY4_TYPE_Q6_K:
      dispatch_experts<14>(w, w_up, expert_bytes, limit, x, slots_per_x, ids,
                           plan, route_w, y, m, k, n_slots, max_cols, stream);
      break;
    default:
      return 1;
  }
  const hipError_t err = hipGetLastError();
  if (err == hipSuccess)
    return 0;
  fprintf(stderr, "hy4: expert gemv launch failed: %s\n",
          hipGetErrorString(err));
  return 1;
}

extern "C" int hy4_gemv_pair(int type, const void* w1, float* y1, uint32_t m1,
                             const void* w2, float* y2, uint32_t m2,
                             const float* x, uint32_t k, uint32_t n,
                             void* scratch, hipStream_t stream) {
  if (!w2 || m2 == 0)
    return 1;
  return run(type, w1, nullptr, 0.0f, x, y1, m1 + m2, k, Second{w2, y2, m1},
             Tokens{n, k, m1}, scratch, stream);
}

extern "C" int hy4_gemv_swiglu(int type, const void* gate, const void* up,
                               const float* x, float* y, uint32_t m, uint32_t k,
                               float limit, uint32_t n, void* scratch,
                               hipStream_t stream) {
  if (!up)
    return 1;
  return run(type, gate, up, limit, x, y, m, k, Second{nullptr, nullptr, 0},
             Tokens{n, k, m}, scratch, stream);
}
