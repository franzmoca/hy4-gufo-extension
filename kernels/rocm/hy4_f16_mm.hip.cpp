// LLVM generates gfx11 code with real-true16 by default and moves f16 values
// through register halves, which slows these kernels (DS41 measured 2.5-13%).
// The attribute covers every function defined from here on; the HIP math
// wrappers are included before this line and would not inline into it, so
// these kernels call the builtins directly.
#if defined(__HIP_DEVICE_COMPILE__)
#pragma clang attribute push(__attribute__((target("no-real-true16"))), \
                             apply_to = function)
#endif

/* Batched Q4_K, Q6_K and Q8_0 projections on the FP16 matrix cores, dense and
 * routed. Adapted from the DeepSeek V4.1 Flash extension's dense FP16 kernel
 * (ds41_dense_f16.hip.cpp, MIT). The activations are rounded to FP16 as they
 * are staged (11-bit mantissa, against the 8-bit block quantization of MMQ's
 * Q8_1 inputs), or arrive as FP16, and each weight decodes to FP16 as
 * (d sc) q - dmin m (Q6_K: (d sc) (q - 32), Q8_0: d q), so the K loop has no
 * per-block scale arithmetic beside the matrix-core products; the products
 * accumulate in FP32.
 *
 * Each wave holds 16 weight rows as the B operand (lane = weight row) against
 * up to eight 16-token tiles. A K16 step covers two 8-value groups of a
 * 32-value sub-block; each half-wave loads and decodes one of them (half h:
 * group 2 e + h of step e) and takes the other from the opposite half with
 * permlanex16, so half 1 holds every step with its groups swapped and reads
 * its activation fragment swapped to match. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "kernels/rocm/hy4_kernels.h"

namespace {

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

constexpr int kThreads = 512;  // 16 waves
constexpr int kStageK = 64;    // activation values per LDS stage
constexpr int kPitch =
    kStageK + 8;  // halves; token rows land on distinct banks
constexpr uint32_t kHalfMagic = 0x64646464u;

__device__ __forceinline__ uint32_t scale_byte(const u32x4& head, int i) {
  const uint32_t word = i < 4   ? head.y
                        : i < 8 ? head.z
                                : head.w;  // scales start at byte 4
  return (word >> (8 * (i % 4))) & 0xffu;
}

/* (scale, bias) of K-quant sub-block j as a half2: weight = scale q - bias. */
__device__ __forceinline__ uint32_t k_affine(const u32x4& head, int j) {
  uint32_t sc, mn;
  if (j < 4) {
    sc = scale_byte(head, j) & 63;
    mn = scale_byte(head, j + 4) & 63;
  } else {
    sc =
        (scale_byte(head, j + 4) & 0xf) | ((scale_byte(head, j - 4) >> 6) << 4);
    mn = (scale_byte(head, j + 4) >> 4) | ((scale_byte(head, j) >> 6) << 4);
  }
  const float d = static_cast<float>(
      __builtin_bit_cast(_Float16, static_cast<uint16_t>(head.x & 0xffffu)));
  const float dmin = static_cast<float>(
      __builtin_bit_cast(_Float16, static_cast<uint16_t>(head.x >> 16)));
  return __builtin_bit_cast(
      uint32_t, half2_t{static_cast<_Float16>(d * static_cast<float>(sc)),
                        static_cast<_Float16>(-dmin * static_cast<float>(mn))});
}

/* Eight unsigned values (one byte each, < 1024) as four half2 1024 + v. */
__device__ __forceinline__ void magic8(uint32_t b0, uint32_t b1, uint32_t* p) {
  p[0] = __builtin_amdgcn_perm(b0, kHalfMagic, 0x01050004u);
  p[1] = __builtin_amdgcn_perm(b0, kHalfMagic, 0x03070206u);
  p[2] = __builtin_amdgcn_perm(b1, kHalfMagic, 0x01050004u);
  p[3] = __builtin_amdgcn_perm(b1, kHalfMagic, 0x03070206u);
}

/* Q4_K (144 bytes: d, dmin, scales[12], qs[128]). A half-wave keeps the
 * header and, for value offsets 8 h .. 8 h + 7 and 16 + 8 h .. 16 + 8 h + 7
 * of each sub-block pair, the nibble bytes (low nibble: sub-block 2 p, high:
 * 2 p + 1). */
struct Q4Lane {
  u32x4 head;
  uint2 qs[4][2];  // [pair][part]
};

struct Q4K {
  static constexpr int kBlockBytes = 144;
  static constexpr bool kWide = false;
  using Lane = Q4Lane;
  static __device__ __forceinline__ Lane load(const uint8_t* b, int h) {
    Lane q;
    q.head = *reinterpret_cast<const u32x4*>(b);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair)
#pragma unroll
      for (int part = 0; part < 2; ++part)
        q.qs[pair][part] = *reinterpret_cast<const uint2*>(b + 16 + 32 * pair +
                                                           16 * part + 8 * h);
    return q;
  }
  static __device__ __forceinline__ uint32_t affine(const Lane& q, int j) {
    return k_affine(q.head, j);
  }
  static __device__ __forceinline__ void decode8(const Lane& q, int j, int part,
                                                 uint32_t affine,
                                                 uint32_t* out) {
    const uint2 low = q.qs[j / 2][part];
    const uint32_t b0 = (low.x >> (4 * (j % 2))) & 0x0f0f0f0fu;
    const uint32_t b1 = (low.y >> (4 * (j % 2))) & 0x0f0f0f0fu;
    const half2_t s = __builtin_bit_cast(half2_t, affine);
    const half2_t scale = {s.x, s.x}, bias = {s.y, s.y};
    const half2_t magic = {static_cast<_Float16>(-1024.0f),
                           static_cast<_Float16>(-1024.0f)};
    uint32_t p[4];
    magic8(b0, b1, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __builtin_bit_cast(
          uint32_t, (__builtin_bit_cast(half2_t, p[i]) + magic) * scale + bias);
  }
};

/* Q6_K (210 bytes: ql[128], qh[64], scales[16], d). For value offsets
 * 8 h .. 8 h + 7 and 16 + 8 h .. 16 + 8 h + 7 of each 32-value sub-block
 * j = 4 n + g: the ql bytes of both ql vectors of half n (low nibbles for
 * g < 2, high for g >= 2) and the qh bytes of half n (bits 2 g), with the
 * 16 signed scales and d. Blocks are 210 bytes, so these loads are only
 * 2-byte aligned. */
struct Q6Lane {
  uint2 ql[2][2][2];  // [n][g & 1][part]
  uint2 qh[2][2];     // [n][part]
  u32x4 sc;
  uint32_t d;
};

struct Q6K {
  static constexpr int kBlockBytes = 210;
  static constexpr bool kWide = true;
  using Lane = Q6Lane;
  static __device__ __forceinline__ Lane load(const uint8_t* b, int h) {
    Lane q;
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
      for (int part = 0; part < 2; ++part) {
#pragma unroll
        for (int a = 0; a < 2; ++a)
          q.ql[n][a][part] = *reinterpret_cast<const uint2*>(
              b + 64 * n + 32 * a + 16 * part + 8 * h);
        q.qh[n][part] = *reinterpret_cast<const uint2*>(b + 128 + 32 * n +
                                                        16 * part + 8 * h);
      }
    q.sc = *reinterpret_cast<const u32x4*>(b + 192);
    q.d = *reinterpret_cast<const uint16_t*>(b + 208);
    return q;
  }
  /* The scales of parts 0 and 1 of sub-block j (scales 8 n + 2 g and the
   * next one) as a half2. */
  static __device__ __forceinline__ uint32_t affine(const Lane& q, int j) {
    const int i = 8 * (j >> 2) + 2 * (j & 3);
    const uint32_t word = q.sc[i / 4] >> (8 * (i % 4));
    const float d = static_cast<float>(
        __builtin_bit_cast(_Float16, static_cast<uint16_t>(q.d)));
    return __builtin_bit_cast(
        uint32_t,
        half2_t{
            static_cast<_Float16>(
                d * static_cast<float>(static_cast<int8_t>(word & 0xffu))),
            static_cast<_Float16>(d * static_cast<float>(static_cast<int8_t>(
                                          (word >> 8) & 0xffu)))});
  }
  static __device__ __forceinline__ void decode8(const Lane& q, int j, int part,
                                                 uint32_t affine,
                                                 uint32_t* out) {
    const int n = j >> 2, g = j & 3;
    const uint2 low = q.ql[n][g & 1][part];
    const uint2 high = q.qh[n][part];
    const int shift = 4 * (g >> 1);
    const uint32_t b0 = ((low.x >> shift) & 0x0f0f0f0fu) |
                        (((high.x >> (2 * g)) & 0x03030303u) << 4);
    const uint32_t b1 = ((low.y >> shift) & 0x0f0f0f0fu) |
                        (((high.y >> (2 * g)) & 0x03030303u) << 4);
    const half2_t s = __builtin_bit_cast(half2_t, affine);
    const _Float16 sc = part ? s.y : s.x;
    const half2_t scale = {sc, sc};
    /* 1024 + q - 1056 = q - 32, exact in FP16. */
    const half2_t magic = {static_cast<_Float16>(-1056.0f),
                           static_cast<_Float16>(-1056.0f)};
    uint32_t p[4];
    magic8(b0, b1, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __builtin_bit_cast(
          uint32_t, (__builtin_bit_cast(half2_t, p[i]) + magic) * scale);
  }
};

/* Eight Q8_0 blocks (d, qs[32]) as one 256-value block: sub-block j is
 * Q8_0 block j. A half-wave keeps the value bytes 8 h .. 8 h + 7 and
 * 16 + 8 h .. 16 + 8 h + 7 of each, and the eight scales two per word.
 * Blocks are 34 bytes, so these loads are only 2-byte aligned. */
struct Q8Lane {
  uint2 qs[8][2];  // [j][part]
  uint32_t d[4];
};

struct Q8 {
  static constexpr int kBlockBytes = 8 * 34;
  static constexpr bool kWide = true;
  using Lane = Q8Lane;
  static __device__ __forceinline__ Lane load(const uint8_t* b, int h) {
    Lane q;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int part = 0; part < 2; ++part)
        q.qs[j][part] =
            *reinterpret_cast<const uint2*>(b + 34 * j + 2 + 16 * part + 8 * h);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      q.d[i] = *reinterpret_cast<const uint16_t*>(b + 68 * i) |
               (uint32_t)*reinterpret_cast<const uint16_t*>(b + 68 * i + 34)
                   << 16;
    return q;
  }
  /* d of sub-block j in both halves. */
  static __device__ __forceinline__ uint32_t affine(const Lane& q, int j) {
    const uint32_t d = (q.d[j / 2] >> (16 * (j % 2))) & 0xffffu;
    return d | d << 16;
  }
  static __device__ __forceinline__ void decode8(const Lane& q, int j, int part,
                                                 uint32_t affine,
                                                 uint32_t* out) {
    /* Flipping the sign bit maps q to q + 128 in 0 .. 255; 1024 + q + 128 -
     * 1152 = q, exact in FP16. */
    const uint32_t b0 = q.qs[j][part].x ^ 0x80808080u,
                   b1 = q.qs[j][part].y ^ 0x80808080u;
    const half2_t scale = __builtin_bit_cast(half2_t, affine);
    const half2_t magic = {static_cast<_Float16>(-1152.0f),
                           static_cast<_Float16>(-1152.0f)};
    uint32_t p[4];
    magic8(b0, b1, p);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      out[i] = __builtin_bit_cast(
          uint32_t, (__builtin_bit_cast(half2_t, p[i]) + magic) * scale);
  }
};

/* Lane L's word from lane L ^ 16. With fetch-inactive and bound-control set
 * the instruction does not keep its destination's old value (every lane is
 * active here), so the compiler copies no register for it. */
__device__ __forceinline__ uint32_t other_half(uint32_t v) {
  return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, true,
                                      true);
}

/* A barrier that waits only for LDS traffic: __syncthreads also waits for
 * every outstanding global load, which would wait for the weight and
 * activation prefetches at once. */
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ float finite_or_zero(float v) {
  return __builtin_isfinite(v) ? v : 0.0f;
}

/* The shared K loop: rows of `wr` (this lane's weight row) against `kTiles`
 * 16-token tiles of activations staged through LDS. Thread `tid` stages
 * kPer values of token row `st` at offset kPer sh of each stage from xs
 * (null: the row is past the batch and stays unstaged). */
template<typename F, int kTiles, typename X>
struct KLoop {
  static constexpr int kTok = 16 * kTiles;
  static constexpr int kPer = kStageK * kTok / kThreads;
  static constexpr int kVec = kPer * sizeof(X) / 16;  // 16-byte loads per stage
  /* Wide formats hold more weight bytes per lane; a shorter ring keeps the
   * register count down. */
  static constexpr int kAhead = (sizeof(X) == 2 ? 4 : 2) / (F::kWide ? 2 : 1);

  static __device__ __forceinline__ void run(_Float16* act, const uint8_t* wr,
                                             const X* xs, uint32_t k, int live,
                                             float8 (&acc)[kTiles]) {
    const int tid = threadIdx.x, lane = tid % 32, h = lane / 16;
    const uint32_t st = tid % kTok, sh = tid / kTok;
    _Float16* as0 = act + st * kPitch + kPer * sh;
    _Float16* as1 = act + kTok * kPitch + st * kPitch + kPer * sh;
    u32x4 ring[kAhead][kVec];
    const auto fetch = [&](u32x4(&slot)[kVec], uint32_t k0) {
      if (xs) {
#pragma unroll
        for (int c = 0; c < kVec; ++c)
          slot[c] = reinterpret_cast<const u32x4*>(xs + k0)[c];
      }
    };
    int g = 0;
    const auto store = [&](const u32x4(&slot)[kVec]) {
      if (xs) {
        _Float16* as = (g & 1) ? as1 : as0;
#pragma unroll
        for (int c = 0; c < kPer / 8; ++c) {
          if constexpr (sizeof(X) == 2) {
            *reinterpret_cast<u32x4*>(as + 8 * c) = slot[c];
          } else {
            const float4 a = __builtin_bit_cast(float4, slot[2 * c]),
                         b = __builtin_bit_cast(float4, slot[2 * c + 1]);
            *reinterpret_cast<half8*>(as + 8 * c) = half8{
                (_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
          }
        }
      }
    };

    const uint32_t blocks = k / 256;
    typename F::Lane next = F::load(wr, h);
#pragma unroll
    for (int a = 0; a < kAhead; ++a)
      if (a * kStageK < k)
        fetch(ring[a], a * kStageK);
    for (uint32_t sb = 0; sb < blocks; ++sb) {
      const typename F::Lane q = next;
      if (sb + 1 < blocks)
        next = F::load(wr + (uint64_t)(sb + 1) * F::kBlockBytes, h);
#pragma unroll
      for (int stage = 0; stage < 4; ++stage) {
        const uint32_t k0 = sb * 256 + stage * kStageK;
        store(ring[stage % kAhead]);
        /* Stage g stores into buffer g & 1 while stage g-1 still reads the
         * other one: one barrier per stage instead of two. */
        lds_barrier();
        if (k0 + kAhead * kStageK < k)
          fetch(ring[stage % kAhead], k0 + kAhead * kStageK);
        const _Float16* cur = act + (g & 1) * kTok * kPitch;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int j = stage * 2 + jj;
          const uint32_t affine = F::affine(q, j);
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            uint32_t words[8];
            F::decode8(q, j, e, affine, words);
#pragma unroll
            for (int i = 0; i < 4; ++i)
              words[4 + i] = other_half(words[i]);
            const half16 b = __builtin_bit_cast(half16, words);
            const int kk = jj * 32 + e * 16;
            /* Half h reads values 8 h .. 8 h + 7 of its token row and takes
             * the other eight from the opposite half. */
            const auto frag = [&](int t) {
              const uint4 own = *reinterpret_cast<const uint4*>(
                  cur + (t * 16 + lane % 16) * kPitch + kk + 8 * h);
              uint32_t v[8] = {own.x, own.y, own.z, own.w};
#pragma unroll
              for (int i = 0; i < 4; ++i)
                v[4 + i] = other_half(v[i]);
              return __builtin_bit_cast(half16, v);
            };
#pragma unroll
            for (int t = 0; t < kTiles; ++t)
              if (t < live)
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(b, frag(t),
                                                                    acc[t]);
          }
        }
        ++g;
      }
    }
  }
};

/* ---- Dense: a workgroup takes 256 weight rows and 128 tokens. ---- */

constexpr int kRows = 256;
constexpr int kTile = 128;

/* out[t, z m + r] = sum_c W_z[r, c] x[t, z k + c] for group z; x rows are
 * groups * k values of X, out rows groups * m of Y. Row blocks past the first
 * matrix's row blocks take a second [m2][k] matrix w2 of one group on the same
 * rows of x, into out2. Rows past m (m a multiple of 16) are computed from the
 * last row and not stored. */
template<typename F, bool kFull, typename X, typename Y>
__global__ __launch_bounds__(kThreads) void dense_f16_kernel(
    const uint8_t* __restrict__ w, const X* __restrict__ x, Y* __restrict__ out,
    uint32_t m, uint32_t n, uint32_t k, uint32_t groups, uint32_t tile0,
    const uint8_t* __restrict__ w2, Y* __restrict__ out2, uint32_t m2,
    uint32_t row_blocks, uint32_t tiles, uint32_t chunk_rows,
    uint32_t chunk_tiles) {
  __shared__ __attribute__((aligned(16))) _Float16 act[2 * kTile * kPitch];
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, h = lane / 16;
  const uint64_t row_bytes = (uint64_t)k / 256 * F::kBlockBytes;
  /* Workgroups run in index order: group, then chunks of chunk_tiles token
   * tiles, chunks of chunk_rows row blocks, tiles, row blocks. */
  const uint32_t per_group = row_blocks * tiles;
  const uint32_t z = blockIdx.x / per_group;
  uint32_t index = blockIdx.x - z * per_group;
  const uint32_t tile_chunk = index / (row_blocks * chunk_tiles);
  index -= tile_chunk * row_blocks * chunk_tiles;
  const uint32_t tiles_here =
      min(chunk_tiles, tiles - tile_chunk * chunk_tiles);
  const uint32_t row_chunk = index / (chunk_rows * tiles_here);
  index -= row_chunk * chunk_rows * tiles_here;
  const uint32_t rows_here =
      min(chunk_rows, row_blocks - row_chunk * chunk_rows);
  const uint32_t tile = tile_chunk * chunk_tiles + index / rows_here;
  uint32_t block = row_chunk * chunk_rows + index % rows_here;
  const uint32_t blocks1 = (m + kRows - 1) / kRows;
  if (block >= blocks1) {
    block -= blocks1;
    w = w2;
    out = out2;
    m = m2;
  }
  const uint32_t row0 = block * kRows + wave * 16;
  const uint32_t row = min(row0 + lane % 16, m - 1);
  const uint8_t* wr = w + ((uint64_t)z * m + row) * row_bytes;
  const uint32_t t0 = (tile0 + tile) * kTile;
  /* Live 16-token tiles; a runtime count even for full tiles, since a
   * constant one makes the compiler hoist every fragment load and spill. */
  const uint32_t rows16 = (n - t0 + 15) / 16;
  const int live = (int)(rows16 < 8 ? rows16 : 8);
  const uint64_t x_stride = (uint64_t)groups * k;
  using L = KLoop<F, 8, X>;
  const uint32_t st = tid % kTile, sh = tid / kTile;
  const X* xs = kFull || t0 + st < n
                    ? x + (t0 + st) * x_stride + (uint64_t)z * k + L::kPer * sh
                    : nullptr;

  float8 acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t)
    acc[t] = float8{};
  L::run(act, wr, xs, k, live, acc);

  /* Accumulator element e of a lane is (row 2 e + lane / 16, token lane % 16).
   */
  if (row0 >= m)
    return;
  const uint64_t out_stride = (uint64_t)groups * m;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    if (t >= live)
      continue;
    const uint32_t token = t0 + t * 16 + lane % 16;
    if (!kFull && token >= n)
      continue;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      out[token * out_stride + (uint64_t)z * m + row0 + 2 * e + h] =
          (Y)finite_or_zero(acc[t][e]);
  }
}

struct DenseArgs {
  const uint8_t* w;
  const void* x;
  void* out;
  uint32_t m, n, k, groups;
  const uint8_t* w2;
  void* out2;
  uint32_t m2;
};

struct DenseChunks {
  uint32_t rows, tiles;
};

/* gfx1151 reads its 32 MiB last-level cache about four times faster than
 * DRAM. A batch sweeps the weights once per token tile, so a matrix larger
 * than the cache is taken in row chunks that stay cached across the tiles,
 * each with the activations it rereads (DS41's measured rule). The order does
 * not change any result. */
DenseChunks dense_chunks(uint64_t block_bytes, uint64_t tile_bytes,
                         uint32_t blocks, uint32_t tiles) {
  constexpr uint64_t kCache = 32ull << 20, kRowChunk = 5ull << 20,
                     kTileChunk = 12ull << 20;
  const uint64_t weights = block_bytes * blocks;
  const auto fit = [](uint64_t budget, uint64_t unit, uint32_t count) {
    return (uint32_t)std::min<uint64_t>(count,
                                        std::max<uint64_t>(1, budget / unit));
  };
  if (weights <= kCache)
    return {blocks, 1};
  if (tile_bytes * tiles <= kCache / 4 && block_bytes * 8 <= kRowChunk)
    return {fit(kRowChunk, block_bytes, blocks), tiles};
  if (weights >= 2 * kCache)
    return {fit(kRowChunk, block_bytes, blocks),
            fit(kTileChunk, tile_bytes, tiles)};
  return {blocks, 1};
}

template<typename F, typename X, typename Y>
int dense_launch(const DenseArgs& a, hipStream_t stream) {
  const auto* xt = static_cast<const X*>(a.x);
  auto* yt = static_cast<Y*>(a.out);
  auto* y2 = static_cast<Y*>(a.out2);
  const uint32_t full = a.n / kTile, blocks = (a.m + kRows - 1) / kRows +
                                              (a.m2 + kRows - 1) / kRows;
  if (full) {
    const uint64_t row_bytes = (uint64_t)a.k / 256 * F::kBlockBytes;
    const DenseChunks c = dense_chunks(
        kRows * row_bytes, (uint64_t)kTile * a.k * sizeof(X), blocks, full);
    dense_f16_kernel<F, true, X, Y>
        <<<dim3(blocks * full * a.groups), kThreads, 0, stream>>>(
            a.w, xt, yt, a.m, a.n, a.k, a.groups, 0, a.w2, y2, a.m2, blocks,
            full, std::min(c.rows, blocks), std::min(c.tiles, full));
  }
  if (a.n % kTile)
    dense_f16_kernel<F, false, X, Y>
        <<<dim3(blocks * a.groups), kThreads, 0, stream>>>(
            a.w, xt, yt, a.m, a.n, a.k, a.groups, full, a.w2, y2, a.m2, blocks,
            1, blocks, 1);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

template<typename F>
int dense_io(const DenseArgs& a, uint32_t io, hipStream_t stream) {
  switch (io) {
    case 0:
      return dense_launch<F, float, float>(a, stream);
    case HY4_F16_X_HALF:
      return dense_launch<F, _Float16, float>(a, stream);
    case HY4_F16_OUT_HALF:
      return dense_launch<F, float, _Float16>(a, stream);
    default:
      return dense_launch<F, _Float16, _Float16>(a, stream);
  }
}

int dense_run(int type, const DenseArgs& a, uint32_t io, hipStream_t stream) {
  if (a.m == 0 || a.m % 16 != 0 || a.m2 % 16 != 0 || a.k % 256 != 0 ||
      a.n == 0 || a.groups == 0 || (a.m2 && a.groups != 1) ||
      (io & ~(HY4_F16_X_HALF | HY4_F16_OUT_HALF)))
    return 1;
  switch (type) {
    case HY4_TYPE_Q8_0:
      return dense_io<Q8>(a, io, stream);
    case HY4_TYPE_Q4_K:
      return dense_io<Q4K>(a, io, stream);
    case HY4_TYPE_Q6_K:
      return dense_io<Q6K>(a, io, stream);
    default:
      return 1;
  }
}

/* ---- Routed experts: a workgroup takes one 128-slot tile of one expert. ----
 *
 * `rows` holds the slots grouped by expert (expert e owns rows[bounds[e] ..
 * bounds[e + 1])), and each tile entry (expert | index << 16) names 128 of
 * them (16-slot tiles past the bucket are skipped, so a wide tile pads no
 * more than a narrow one, and with four tiles the compiler hoists every
 * fragment load and spills). Slot s reads x row s / x_div. Without a pair the
 * 16 waves take 256 weight rows and store y[s, r] (FP32). With a pair, waves
 * 0-7 take 128 gate rows and waves 8-15 the same up rows; the up waves pass
 * their sums through LDS and the gate waves store y[s, r] = route_w[s]
 * silu(min(g, limit)) clamp(u, +-limit) as FP16 (no clamp for limit <= 0). */
constexpr int kSlots = 128;

template<typename F, typename X, bool kPair>
__global__ __launch_bounds__(kThreads) void routed_f16_kernel(
    const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
    uint64_t expert_bytes, const X* __restrict__ x, uint32_t x_div,
    const int32_t* __restrict__ rows, const int32_t* __restrict__ bounds,
    const int32_t* __restrict__ tiles, const float* __restrict__ route_w,
    float limit, void* __restrict__ y, uint32_t m, uint32_t k) {
  /* The activation stages (2 x 128 x kPitch halves) and, after the K loop,
   * half the up sums (8 waves x 4 tiles x 8 x 32 floats) share one buffer. */
  constexpr int kActBytes = 2 * kSlots * kPitch * 2,
                kUpBytes = 8 * 4 * 8 * 32 * 4;
  __shared__ __attribute__((aligned(16))) uint8_t
      smem[kPair ? (kActBytes > kUpBytes ? kActBytes : kUpBytes) : kActBytes];
  __shared__ int32_t slot_of[kSlots];
  const int tile = tiles[blockIdx.y];
  if (tile < 0)
    return;
  const int expert = tile & 0xffff;
  const int begin = bounds[expert] + (tile >> 16) * kSlots;
  const int count = min(kSlots, bounds[expert + 1] - begin);
  const int live = (count + 15) / 16;
  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32, h = lane / 16;
  if (tid < kSlots)
    slot_of[tid] = tid < count ? rows[begin + tid] : -1;
  __syncthreads();

  const int mat = kPair ? wave / 8 : 0;
  const uint32_t row0 =
      kPair ? blockIdx.x * 128 + (wave % 8) * 16 : blockIdx.x * 256 + wave * 16;
  const uint32_t row = min(row0 + lane % 16, m - 1);
  const uint64_t row_bytes = (uint64_t)k / 256 * F::kBlockBytes;
  const uint8_t* wr =
      (mat ? w1 : w0) + (uint64_t)expert * expert_bytes + row * row_bytes;

  using L = KLoop<F, 8, X>;
  const uint32_t st = tid % kSlots, sh = tid / kSlots;
  const int32_t s_slot = slot_of[st];
  const X* xs = s_slot >= 0
                    ? x + (uint64_t)(s_slot / (int)x_div) * k + L::kPer * sh
                    : nullptr;

  float8 acc[8];
#pragma unroll
  for (int t = 0; t < 8; ++t)
    acc[t] = float8{};
  L::run(reinterpret_cast<_Float16*>(smem), wr, xs, k, live, acc);

  if constexpr (kPair) {
    float* up = reinterpret_cast<float*>(smem);
    /* Tiles 0-3, then 4-7: each round's up sums fill the buffer. */
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      __syncthreads();  // the activation stages or the previous round are read
      if (mat) {
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
          for (int e = 0; e < 8; ++e)
            up[(((wave % 8) * 4 + t) * 8 + e) * 32 + lane] =
                acc[4 * half + t][e];
      }
      __syncthreads();
      if (mat || row0 >= m)
        continue;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        if (4 * half + t >= live)
          continue;
        const int32_t slot = slot_of[(4 * half + t) * 16 + lane % 16];
        if (slot < 0)
          continue;
        const float rw = route_w[slot];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float g = finite_or_zero(acc[4 * half + t][e]);
          float u =
              finite_or_zero(up[(((wave % 8) * 4 + t) * 8 + e) * 32 + lane]);
          if (limit > 0.0f) {
            g = __builtin_fminf(g, limit);
            u = __builtin_fminf(__builtin_fmaxf(u, -limit), limit);
          }
          const float v =
              g / (1.0f + __builtin_amdgcn_exp2f(0x1.715476p+0f * -g)) * u * rw;
          static_cast<_Float16*>(y)[(uint64_t)slot * m + row0 + 2 * e + h] =
              static_cast<_Float16>(v);
        }
      }
    }
  } else {
    if (row0 >= m)
      return;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
      if (t >= live)
        continue;
      const int32_t slot = slot_of[t * 16 + lane % 16];
      if (slot < 0)
        continue;
#pragma unroll
      for (int e = 0; e < 8; ++e)
        static_cast<float*>(y)[(uint64_t)slot * m + row0 + 2 * e + h] =
            finite_or_zero(acc[t][e]);
    }
  }
}

template<typename F>
int routed_launch(const void* w, const void* w_up, uint64_t expert_bytes,
                  const void* x, uint32_t x_div, const int32_t* rows,
                  const int32_t* bounds, const int32_t* tiles, uint32_t n_tiles,
                  const float* route_w, float limit, void* y, uint32_t m,
                  uint32_t k, hipStream_t stream) {
  const auto* a = static_cast<const uint8_t*>(w);
  const auto* b = static_cast<const uint8_t*>(w_up);
  if (w_up)
    routed_f16_kernel<F, float, true>
        <<<dim3((m + 127) / 128, n_tiles), kThreads, 0, stream>>>(
            a, b, expert_bytes, static_cast<const float*>(x), x_div, rows,
            bounds, tiles, route_w, limit, y, m, k);
  else
    routed_f16_kernel<F, _Float16, false>
        <<<dim3((m + 255) / 256, n_tiles), kThreads, 0, stream>>>(
            a, nullptr, expert_bytes, static_cast<const _Float16*>(x), x_div,
            rows, bounds, tiles, nullptr, 0.0f, y, m, k);
  return hipGetLastError() == hipSuccess ? 0 : 1;
}

}  // namespace

int hy4_f16_dense(int type, const void* w, const void* x, void* out, uint32_t m,
                  uint32_t n, uint32_t k, uint32_t groups, uint32_t io,
                  hipStream_t stream) {
  return dense_run(type,
                   {static_cast<const uint8_t*>(w), x, out, m, n, k, groups,
                    nullptr, nullptr, 0},
                   io, stream);
}

int hy4_f16_dense_pair(int type, const void* w1, void* out1, uint32_t m1,
                       const void* w2, void* out2, uint32_t m2, const void* x,
                       uint32_t n, uint32_t k, uint32_t io,
                       hipStream_t stream) {
  return dense_run(type,
                   {static_cast<const uint8_t*>(w1), x, out1, m1, n, k, 1,
                    static_cast<const uint8_t*>(w2), out2, m2},
                   io, stream);
}

int hy4_f16_routed(int type, const void* w, const void* w_up,
                   uint64_t expert_bytes, const void* x, uint32_t x_div,
                   const int32_t* rows, const int32_t* bounds,
                   const int32_t* tiles, uint32_t n_tiles, const float* route_w,
                   float limit, void* y, uint32_t m, uint32_t k,
                   hipStream_t stream) {
  if (k % 256 != 0 || m == 0 || m % 16 != 0 || n_tiles == 0 || x_div == 0)
    return 1;
  switch (type) {
    case HY4_TYPE_Q8_0:
      return routed_launch<Q8>(w, w_up, expert_bytes, x, x_div, rows, bounds,
                               tiles, n_tiles, route_w, limit, y, m, k, stream);
    case HY4_TYPE_Q4_K:
      return routed_launch<Q4K>(w, w_up, expert_bytes, x, x_div, rows, bounds,
                                tiles, n_tiles, route_w, limit, y, m, k,
                                stream);
    case HY4_TYPE_Q6_K:
      return routed_launch<Q6K>(w, w_up, expert_bytes, x, x_div, rows, bounds,
                                tiles, n_tiles, route_w, limit, y, m, k,
                                stream);
    default:
      return 1;
  }
}

uint32_t hy4_f16_routed_plan(const int32_t* ids, uint32_t slots,
                             uint32_t n_experts, int32_t* rows, int32_t* bounds,
                             int32_t* tiles) {
  for (uint32_t e = 0; e <= n_experts; ++e)
    bounds[e] = 0;
  for (uint32_t s = 0; s < slots; ++s)
    if (ids[s] >= 0 && (uint32_t)ids[s] < n_experts)
      ++bounds[ids[s] + 1];
  for (uint32_t e = 0; e < n_experts; ++e)
    bounds[e + 1] += bounds[e];
  uint32_t n_tiles = 0;
  for (uint32_t e = 0; e < n_experts; ++e)
    for (int32_t t = 0; t * kSlots < bounds[e + 1] - bounds[e]; ++t)
      tiles[n_tiles++] = (int32_t)(e | (uint32_t)t << 16);
  /* Fill each bucket in slot order, using the next bucket's start as a cursor.
   */
  for (uint32_t s = 0; s < slots; ++s)
    if (ids[s] >= 0 && (uint32_t)ids[s] < n_experts)
      rows[bounds[ids[s]]++] = (int32_t)s;
  for (uint32_t e = n_experts; e > 0; --e)
    bounds[e] = bounds[e - 1];
  bounds[0] = 0;
  return n_tiles;
}

#if defined(__HIP_DEVICE_COMPILE__)
#pragma clang attribute pop
#endif
