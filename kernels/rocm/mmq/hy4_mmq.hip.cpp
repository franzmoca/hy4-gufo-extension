// SPDX-License-Identifier: MIT
// hy4_mmq.hip.cpp - host wrapper around llama.cpp's vendored mul_mat_q and
// mul_mat_vec_q kernels.
//
// Implements the public hy4_mmq_* entry points for the weight types Hy4's
// GGUFs carry (Q8_0, Q4_K, Q6_K) and explicitly instantiates
// mul_mat_q_case<T> for each of them.
//
#include "hy4_mmq.h"

#include "common.hip.hpp"
#include "mmq.hip.hpp"
#include "quantize.hip.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// ----------------------------------------------------------------------------
// Init
// ----------------------------------------------------------------------------

struct hy4_hip_persistent_scratch {
    void *ptr = nullptr;
    size_t bytes = 0;
    std::mutex mutex;
};

static hy4_hip_persistent_scratch g_hip_dense_q81_scratch;

static void *hy4_hip_persistent_scratch_reserve_locked(
        hy4_hip_persistent_scratch *scratch,
        size_t bytes,
        size_t granularity,
        hipStream_t stream,
        const char *tag) {
    if (!scratch || bytes == 0 || stream != (hipStream_t)0) return nullptr;
    if (scratch->ptr && scratch->bytes >= bytes) return scratch->ptr;
    if (granularity == 0 || bytes > SIZE_MAX - (granularity - 1u)) {
        return nullptr;
    }

    const size_t reserve =
        ((bytes + granularity - 1u) / granularity) * granularity;
    hipError_t err = hipStreamSynchronize(stream);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: scratch resize synchronization failed: %s\n",
                tag, hipGetErrorString(err));
        return nullptr;
    }

    void *next = nullptr;
    err = hipMalloc(&next, reserve);
    if (err != hipSuccess) {
        fprintf(stderr, "%s: persistent scratch allocation of %zu bytes failed: %s\n",
                tag, reserve, hipGetErrorString(err));
        (void)hipGetLastError();
        return nullptr;
    }
    if (scratch->ptr) (void)hipFree(scratch->ptr);
    scratch->ptr = next;
    scratch->bytes = reserve;
    return scratch->ptr;
}

extern "C" int hy4_mmq_init(int device) {
    if (device < 0) {
        fprintf(stderr, "hy4_mmq_init: invalid device %d\n", device);
        return -1;
    }
    hy4_ggml_hip_set_device(device);
    // Trigger lazy population of the device-info singleton.
    const auto & info = hy4_ggml_hip_info();
    if (info.device_count == 0) {
        fprintf(stderr, "hy4_mmq_init: no HIP devices found\n");
        return -1;
    }
    if (device >= info.device_count) {
        fprintf(stderr, "hy4_mmq_init: device %d out of range (have %d)\n",
                device, info.device_count);
        return -1;
    }

    return 0;
}

// ----------------------------------------------------------------------------
// Dense matmul implementation, shared across the quant types.
//
// Computes  out[col, row] = sum_k W[row, k] * X[k, col]   with W in the
// type-specific block layout and X / out in F32 (X K-innermost row-major,
// out column-major out[col*M + row]).
//
// Mirrors upstream mmq.cu:154-159 (the no-ids branch) but builds mmq_args
// from plain pointers + shape ints instead of ggml_tensor introspection.
// ----------------------------------------------------------------------------

// Per-device singleton context. Owns the pool for stream-K fixup scratch.
// Phase 4 will make this per-stream as well; for now a single context per
// device is sufficient for the dense path.
namespace {

__global__ static void hy4_mmq_sanitize_f32_kernel(float *p, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = p[i];
    if (!isfinite(v)) p[i] = 0.0f;
}

static void hy4_mmq_sanitize_f32(float *p, uint64_t n, hipStream_t stream) {
    if (!p || n == 0) return;
    hy4_mmq_sanitize_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256, 0, stream>>>(p, n);
}

hy4_ggml_hip_context * get_ctx_for_device(int device) {
    static hy4_ggml_hip_context * cached[HY4_GGML_HIP_MAX_DEVICES] = {};
    if (device < 0 || device >= HY4_GGML_HIP_MAX_DEVICES) return nullptr;
    if (!cached[device]) {
        cached[device] = new hy4_ggml_hip_context(device);
    }
    return cached[device];
}

template <ggml_type type>
int hy4_mmq_dense_impl(
        const char  * tag,
        const void  * W,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        hipStream_t  stream) {

    if (!W || !X_f32 || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (K <= 0 || M <= 0 || N <= 0) {
        fprintf(stderr, "%s: bad shape M=%d N=%d K=%d\n", tag, M, N, K);
        return -1;
    }
    if (K % 256 != 0) {
        // mmq requires K to be a multiple of the largest super-block size
        // it sees during the inner tile loop, which is QK_K=256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = hy4_ggml_hip_get_device();
    const int cc  = hy4_ggml_hip_info().devices[dev].cc;

    hy4_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    /* Order the pool's hipMallocAsync/hipFreeAsync on the SAME
     * stream the kernels below launch on.  The pool defaults to
     * hipStreamPerThread; with kernels on the legacy stream the RAII free is
     * ordered on an EMPTY stream, so the driver can recycle/remap the scratch
     * while the in-flight quantize/GEMM still reads it -> intermittent illegal
     * access under shape churn (the batched-draft early-step crash).  The vec
     * impls already do this (graph-capture fix); the batched impls were missed.
     */
    hy4_pool_set_stream(stream);

    // 1. Quantize F32 activations into MMQ's Q8_1 layout.
    const int64_t ne00         = K;
    const int64_t ne10_padded  = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11         = N;
    const size_t y_block_size = sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = 4 * QK8_1;
    const size_t nbytes_src1_q8_1 =
        ne11 * ne10_padded * y_block_size / y_values_per_block +
        get_mmq_x_max_host(cc) * sizeof(block_q8_1_mmq);

    hy4_ggml_hip_pool_alloc<char> src1_q8_1_pool;
    char *src1_q8_1 = nullptr;
    std::unique_lock<std::mutex> dense_scratch_lock;
    if (stream == (hipStream_t)0) {
        dense_scratch_lock =
            std::unique_lock<std::mutex>(g_hip_dense_q81_scratch.mutex);
        src1_q8_1 = (char *)hy4_hip_persistent_scratch_reserve_locked(
            &g_hip_dense_q81_scratch,
            nbytes_src1_q8_1,
            64u * 1024u * 1024u,
            stream,
            tag);
    }
    if (!src1_q8_1) {
        src1_q8_1 = src1_q8_1_pool.alloc(
            ctx->pool(), nbytes_src1_q8_1);
    }

    // S1.1a fix: the mmq Y (activation) buffer is over-allocated for the kernel's
    // tail-tile reads (the +mmq_x_max blocks above), and ne11 columns may not fill
    // the final column tile -- but hy4_quantize_mmq_q8_1_hip only writes the ne11 valid
    // columns.  The mmq kernel (mmq.hip.hpp:3528) unconditionally loads the full column
    // tile, reading the never-written tail.  Pool allocs reuse stale device memory,
    // so that tail is non-deterministic: any allocator/stream perturbation (e.g. an
    // MTP draft's hipMalloc) changes it and flips a near-threshold argmax in the
    // batched forward (confirmed by compute-sanitizer --tool initcheck on a PRO6000
    // / sm_120: 4-byte uninitialized __global__ read in mul_mat_q_process_tile).
    // The tail's dot-products are masked out by write_back, so only their
    // non-determinism matters; zero the buffer so the tail is a deterministic zero
    // (a zero q8_1 block contributes 0 to the dot product).

    hy4_quantize_mmq_q8_1_hip(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1,
        type, /*ne00=*/K, /*s11=*/(int64_t)K, /*s12=*/0, /*s13=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/1, /*ne3=*/1,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, hipGetErrorString(err));
        return -2;
    }

    // 2. Build mmq_args. stride_row_x is in WEIGHT BLOCKS per row, which
    //    is K / blck_size(type). Q8_0 has block size 32; Q4_K and Q6_K are
    //    K-quants with block size 256.
    const int64_t blck   = ggml_blck_size(type);
    const int64_t s01    = (int64_t)K / blck;
    const int64_t s1     = (int64_t)M;
    const int64_t s12    = ne11 * ne10_padded * y_block_size /
                           (y_values_per_block * sizeof(int));
    const int64_t s13    = s12;

    const bool use_stream_k =
        (HY4_GGML_HIP_CC_IS_NVIDIA(cc) && hy4_ggml_hip_highest_compiled_arch(cc) >= HY4_GGML_HIP_CC_VOLTA) ||
        HY4_GGML_HIP_CC_IS_CDNA(cc);

    mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*ncols_x=*/ne00,    /*nrows_x=*/(int64_t)M,    /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01,/*ncols_y=*/ne11,          /*nrows_dst=*/s1,
        /*nchannels_x=*/1, /*nchannels_y=*/1,
        /*stride_channel_x=*/(int64_t)M * s01, /*stride_channel_y=*/s12, /*stride_channel_dst=*/(int64_t)M,
        /*nsamples_x=*/1,    /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13, /*stride_sample_dst=*/0,
        /*use_stream_k=*/use_stream_k,
        /*ncols_max=*/ne11,
    };
    /* Tiled launches sanitize at the store (see mul_mat_q_case). */
    const bool inline_sanitize = !use_stream_k;
    args.sanitize_output = inline_sanitize;

    mul_mat_q_case<type>(*ctx, args, stream);

    err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: mul_mat_q_case launch failed: %s\n", tag, hipGetErrorString(err));
        return -3;
    }
    if (!inline_sanitize) {
        hy4_mmq_sanitize_f32(out_f32, (uint64_t)s1 * (uint64_t)N, stream);
    }
    return 0;
}

}

// ----------------------------------------------------------------------------
// MMVQ entry points.
//
// mmvq is upstream's matrix-vector matmul family, optimised for the
// n_tokens <= MMVQ_MAX_BATCH_SIZE=8 regime. Unlike mmq it consumes the
// CANONICAL block_q8_1 layout (via hy4_quantize_row_q8_1_hip), not the
// interleaved block_q8_1_mmq that hy4_quantize_mmq_q8_1_hip produces.
//
// The single-W _moe_vec entries cover:
//   - the down matmul at decode (treating [n_tokens=1, n_expert_used=6]
//     as [n_tokens=6, n_expert_used=1])
//   - dense attention projections at decode (n_tokens=1, no MoE)
//   - any small-batch path where mmvq's per-token grid wins over mmq's
//     tile-based approach
// ----------------------------------------------------------------------------

#include "mmvq.hip.hpp"

namespace {

template <ggml_type type>
int hy4_mmq_moe_vec_impl(
        const char    * tag,
        const void    * W,
        const float   * X_f32,
        const int32_t * ids,
        float         * out_f32,
        int             M,
        int             K,
        int             n_tokens,
        int             n_experts,
        int             n_expert_used,
        hipStream_t    stream) {

    if (!W || !X_f32 || !ids || !out_f32) {
        fprintf(stderr, "%s: null pointer\n", tag);
        return -1;
    }
    if (M <= 0 || K <= 0 || n_tokens <= 0 || n_experts <= 0 || n_expert_used <= 0) {
        fprintf(stderr, "%s: bad shape M=%d K=%d ntok=%d nexp=%d nused=%d\n",
                tag, M, K, n_tokens, n_experts, n_expert_used);
        return -1;
    }
    if (K % 256 != 0) {
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }
    if (n_expert_used > n_experts) {
        fprintf(stderr, "%s: n_expert_used=%d > n_experts=%d\n", tag, n_expert_used, n_experts);
        return -1;
    }
    // mmvq's per-arch batch cap. ncols_dst as computed below is
    // max(n_tokens, n_expert_used) depending on which dim we route into.
    // We follow upstream's convention: ne_y = n_tokens, ne_dst = n_expert_used.
    // So ncols_dst = n_tokens and nchannels_dst = n_expert_used.
    // FD Inc2a: n_tokens beyond the per-launch column cap no longer rejects;
    // the launch loop below splits the column dim into capped chunks.

    const int dev = hy4_ggml_hip_get_device();
    hy4_ggml_hip_context * ctx = get_ctx_for_device(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get HIP context for device %d\n", tag, dev);
        return -1;
    }

    // Route the pool's hipMallocAsync / hipFreeAsync through the same
    // stream the caller uses for kernel launches so allocations do not
    // invalidate graph capture.
    hy4_pool_set_stream(stream);

    // 1. Quantize X into CANONICAL Q8_1 (NOT the MMQ-interleaved variant).
    //    Layout: [ne13=1, ne12=n_tokens, ne11=1, ne10_padded blocks].
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const size_t nbytes_q8_1 =
        (size_t)n_tokens * ne10_padded * sizeof(block_q8_1) / QK8_1;
    hy4_ggml_hip_pool_alloc<char> src1_q8_1_pool;
    src1_q8_1_pool.alloc(ctx->pool(), nbytes_q8_1);
    char* src1_q8_1_ptr = src1_q8_1_pool.get();

    // s11 = stride between rows of an src1 channel in source-float units.
    //       Logical src1 [K, ne11=1, ne12=n_tokens, ne13=1] - K innermost.
    // s12 = stride between channels = K * ne11 = K.
    // s13 = stride between samples = K * ne11 * ne12 = K * n_tokens.
    hy4_quantize_row_q8_1_hip(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1_ptr,
        type, /*ne00=*/K,
        /*s11=*/(int64_t)K, /*s12=*/(int64_t)K, /*s13=*/(int64_t)K * n_tokens,
        /*ne0=*/ne10_padded, /*ne1=*/1, /*ne2=*/n_tokens, /*ne3=*/1,
        stream);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "%s: hy4_quantize_row_q8_1_hip failed: %s\n",
                tag, hipGetErrorString(err));
        return -2;
    }

    // 2. mmvq stride setup. Mirror upstream's hy4_ggml_hip_mul_mat_vec_q
    //    dispatch (mmvq.cu:1101-1136).
    //
    //    For MoE (ids != nullptr): per the dispatch math at line 1121-1130,
    //      ncols_dst          = ne2  = n_tokens
    //      nchannels_y        = ne11 = 1
    //      nchannels_dst      = ne1  = n_expert_used
    //      stride_col_y       = s12  = ne11 * (ne10_padded / QK8_1)
    //      stride_col_dst     = s2   = n_expert_used * M (token stride in dst)
    //      stride_channel_y   = s11  = ne10_padded / QK8_1
    //      stride_channel_dst = s1   = M (channel/slot stride in dst)
    //      ids_stride         = stride between rows of ids[] tensor
    //
    //    FD Inc2a stride fix: stride_col_dst was previously M, same as the
    //    channel stride.  That was invisible while every caller degenerated
    //    one dim (gate/up at n_tokens=1: col index always 0; down at
    //    n_expert_used=1: channel index always 0, and 1 * M == M keeps it
    //    bit-identical here).  At n_tokens >= 2 with n_expert_used > 1 the
    //    multi-token MoE kernel writes dst[chan*s1 + col*s2 + row], and
    //    equal strides collide (token=0,slot=1) with (token=1,slot=0).
    //    s2 = n_expert_used * M yields the row-major
    //    [token * n_expert_used + slot, M] layout the swiglu consumer
    //    expects.
    const int64_t blck      = ggml_blck_size(type);
    const int64_t s01_row   = (int64_t)K / blck;            // weight row stride in blocks
    const int64_t s02_chan  = (int64_t)M * s01_row;         // expert-stack stride
    const int64_t s11_y     = ne10_padded / QK8_1;          // src1 channel stride in blocks
    const int64_t s12_y     = (int64_t)1 * s11_y;           // ne11 * s11
    const int64_t s1_dst    = (int64_t)M;                   // dst channel (slot) stride
    const int64_t s2_dst    = (int64_t)n_expert_used * M;   // dst col (token) stride

    // ids_stride: stride between rows of the ids tensor in int32 elements.
    // Caller passes ids[t * n_expert_used + s], so stride between tokens
    // is n_expert_used.
    const int ids_stride = n_expert_used;

    hy4_ggml_hip_mm_fusion_args_device fusion = {};

    /* mmvq stores every (token, slot) output, zeroing non-finite values. */

    // FD Inc2a: one mmvq launch serves at most col_cap columns -- the moe
    // kernel runs one warp per column (block.y = ncols_dst) under
    // __launch_bounds__ baked per COMPILED arch + type
    // (get_mmvq_mmid_max_batch_for_device).  The runtime device cc can
    // exceed the compiled arch (CUDA_ARCH= builds run default-arch PTX on
    // newer GPUs), so the host cap MUST be looked up at the compiled arch:
    // asking the runtime cc says 8 where the compiled bounds say 7 (e.g.
    // Q2_K builds at turing_plus -> 7*warp_size threads) and the launch
    // dies with cudaErrorInvalidValue.  Wider batches run as
    // ceil(n_tokens / col_cap) launches; every per-column stride (vy, ids,
    // dst) is uniform, so a chunk is plain pointer offsets.  The single
    // quantize above already covers all columns.
    const int cc      = hy4_ggml_hip_info().devices[dev].cc;
    const int col_cap = hy4_get_mmvq_mmid_max_batch(type, hy4_ggml_hip_highest_compiled_arch(cc));

    for (int c0 = 0; c0 < n_tokens; c0 += col_cap) {
        const int ncols = (n_tokens - c0 < col_cap) ? (n_tokens - c0) : col_cap;
        mul_mat_vec_q_switch_type(
            /*vx=*/W, /*type_x=*/type,
            /*vy=*/(const void *)(src1_q8_1_ptr + (size_t)c0 * s12_y * sizeof(block_q8_1)),
            /*ids=*/ids + (size_t)c0 * ids_stride, /*fusion=*/fusion,
            /*dst=*/out_f32 + (int64_t)c0 * s2_dst,
            /*ncols_x=*/K, /*nrows_x=*/M, /*ncols_dst=*/ncols,
            /*stride_row_x=*/(int)s01_row,
            /*stride_col_y=*/(int)s12_y,
            /*stride_col_dst=*/(int)s2_dst,
            /*nchannels_x=*/n_experts,
            /*nchannels_y=*/1,
            /*nchannels_dst=*/n_expert_used,
            /*stride_channel_x=*/(int)s02_chan,
            /*stride_channel_y=*/(int)s11_y,
            /*stride_channel_dst=*/(int)s1_dst,
            /*nsamples_x=*/1, /*nsamples_dst=*/1,
            /*stride_sample_x=*/0, /*stride_sample_y=*/0, /*stride_sample_dst=*/0,
            /*ids_stride=*/ids_stride, stream);

        err = hipGetLastError();
        if (err != hipSuccess) {
            fprintf(stderr, "%s: mul_mat_vec_q_switch_type launch failed: %s (cols %d..%d cap %d)\n",
                    tag, hipGetErrorString(err), c0, c0 + ncols - 1, col_cap);
            return -3;
        }
    }

    return 0;
}

} // anonymous namespace

// Weight types chosen per tensor at load time. mmvq dispatches on the type at
// run time; the template parameter only selects the activation quantization
// layout, which is the canonical Q8_1 for every type here.
#define HY4_MMQ_TYPED_CASES(X) \
  X(GGML_TYPE_Q8_0)             \
  X(GGML_TYPE_Q4_K)             \
  X(GGML_TYPE_Q6_K)

extern "C" int hy4_mmq_moe_vec_typed(int type, const void* W, const float* X,
                                      const int32_t* ids, float* out, int M,
                                      int K, int n_tokens, int n_experts,
                                      int n_expert_used, hipStream_t stream) {
  switch ((ggml_type)type) {
#define HY4_CASE(T)                                                                  \
  case T:                                                                             \
    return hy4_mmq_moe_vec_impl<T>("hy4_mmq_moe_vec_typed", W, X, ids, out, M, K, \
                                    n_tokens, n_experts, n_expert_used, stream);
    HY4_MMQ_TYPED_CASES(HY4_CASE)
#undef HY4_CASE
    default:
      fprintf(stderr, "hy4_mmq_moe_vec_typed: unsupported type %d\n", type);
      return -1;
  }
}


// hy4: routed experts of a multi-token pass in one launch, the slots ordered
// by expert (hy4_mmvq_moe_shared). X's n_rows rows are quantized to Q8_1 in one launch,
// row by row as the one-token calls quantize them.
static int hy4_mmq_moe_shared(const char* tag, int type, const void* W, const float* X,
                              const int32_t* ids, float* out, int M, int K, int n_rows,
                              int n_slots, int n_experts, bool per_token, hipStream_t stream) {
  if (!W || !X || !ids || !out || M <= 0 || K <= 0 || n_rows <= 0 || n_slots <= 0 ||
      n_experts <= 0 || K % 256 != 0) {
    fprintf(stderr, "%s: bad arguments\n", tag);
    return -1;
  }
  const int dev = hy4_ggml_hip_get_device();
  hy4_ggml_hip_context* ctx = get_ctx_for_device(dev);
  if (!ctx) return -1;
  hy4_pool_set_stream(stream);
  const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
  hy4_ggml_hip_pool_alloc<char> q8(ctx->pool(),
                                   (size_t)n_rows * ne10_padded * sizeof(block_q8_1) / QK8_1);
  hy4_quantize_row_q8_1_hip(X, nullptr, (void*)q8.get(), (ggml_type)type, K, K, K,
                            (int64_t)K * n_rows, ne10_padded, 1, n_rows, 1, stream);
  const int64_t s01_row = (int64_t)K / ggml_blck_size((ggml_type)type);
  hy4_ggml_hip_pool_alloc<uint32_t> order(ctx->pool(), (size_t)n_slots);
  if (!hy4_mmvq_moe_shared(W, (ggml_type)type, q8.get(), ids, order.get(), out, K, M, (int)s01_row,
                           (int)(ne10_padded / QK8_1), (int)(M * s01_row), n_slots,
                           per_token ? n_slots / n_rows : 1, per_token, stream)) {
    fprintf(stderr, "%s: unsupported type %d\n", tag, type);
    return -1;
  }
  return hipGetLastError() == hipSuccess ? 0 : -3;
}

extern "C" int hy4_mmq_moe_vec_tokens_typed(int type, const void* W, const float* X,
                                             const int32_t* ids, float* out, int M, int K,
                                             int n_tokens, int n_experts, int n_expert_used,
                                             hipStream_t stream) {
  return hy4_mmq_moe_shared("hy4_mmq_moe_vec_tokens_typed", type, W, X, ids, out, M, K, n_tokens,
                            n_tokens * n_expert_used, n_experts, true, stream);
}

extern "C" int hy4_mmq_moe_vec_slots_typed(int type, const void* W, const float* X,
                                            const int32_t* ids, float* out, int M, int K,
                                            int n_slots, int n_experts, hipStream_t stream) {
  return hy4_mmq_moe_shared("hy4_mmq_moe_vec_slots_typed", type, W, X, ids, out, M, K, n_slots,
                            n_slots, n_experts, false, stream);
}

// Batched GEMMs on the mmq tiles of each weight type.
extern "C" int hy4_mmq_dense_typed(int type, const void* W, const float* X,
                                    float* out, int M, int N, int K,
                                    hipStream_t stream) {
  switch ((ggml_type)type) {
#define HY4_CASE(T) \
  case T:            \
    return hy4_mmq_dense_impl<T>("hy4_mmq_dense_typed", W, X, out, M, N, K, stream);
    HY4_MMQ_TYPED_CASES(HY4_CASE)
#undef HY4_CASE
    default:
      fprintf(stderr, "hy4_mmq_dense_typed: unsupported type %d\n", type);
      return -1;
  }
}

// Explicit instantiations, one per weight type the public API exposes. Each
// drags in the load_tiles_<type> + vec_dot_<type>_* device functions from
// mmq.hip.hpp.
template void mul_mat_q_case<GGML_TYPE_Q8_0>(
    hy4_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q4_K>(
    hy4_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
template void mul_mat_q_case<GGML_TYPE_Q6_K>(
    hy4_ggml_hip_context & ctx, const mmq_args & args, hipStream_t stream);
