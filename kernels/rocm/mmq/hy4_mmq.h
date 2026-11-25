// SPDX-License-Identifier: MIT
// Raw-pointer C ABI for the HIP quantized matrix kernels.
// Functions return zero on success and nonzero on failure. Device pointers
// remain caller-owned; every operation uses the supplied HIP stream.

#pragma once

#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One-time init. Sets the current HIP device and triggers lazy population
// of the device-info singleton. Safe to call repeatedly.
//
//   device: HIP device ordinal (0 for the primary GPU).
// Returns 0 on success.
int hy4_mmq_init(int device);

// Batched GEMM (mmq) over a weight type chosen at run time (ggml type code:
// Q8_0, Q4_K or Q6_K), with FP32 activations and outputs and K a multiple of
// 256: out [N, M] = X [N, K] @ W^T.
int hy4_mmq_dense_typed(int type, const void* W, const float* X_f32,
                        float* out_f32, int M, int N, int K,
                        hipStream_t stream);

// Set the thread-local stream that the internal pool uses for
// hipMallocAsync / hipFreeAsync.  Defaults to hipStreamPerThread.
// Use the graph capture stream so pool
// allocations land on the captured stream and don't invalidate capture.
// Pass NULL to reset to hipStreamPerThread.
void hy4_pool_set_stream(hipStream_t stream);

// MoE matvecs (mmvq, N <= 8) over a weight type chosen at run time (Q8_0,
// Q4_K or Q6_K), with FP32 activations and outputs and K a multiple of 256.
// ids [n_tokens, n_expert_used] select the experts of W,
// out [token * n_expert_used + slot, M].
int hy4_mmq_moe_vec_typed(int type, const void* W, const float* X_f32,
                          const int32_t* ids, float* out_f32, int M, int K,
                          int n_tokens, int n_experts, int n_expert_used,
                          hipStream_t stream);
// The same results as n_tokens one-token hy4_mmq_moe_vec_typed calls (token
// t's slots ids[t * n_expert_used ..], out rows t * n_expert_used + j), in
// one launch with the slots of an expert run back to back, so that repeated
// experts read their rows from the caches. Q8_0, Q4_K, Q6_K;
// n_tokens * n_expert_used <= 1024.
int hy4_mmq_moe_vec_tokens_typed(int type, const void* W, const float* X_f32,
                                 const int32_t* ids, float* out_f32, int M,
                                 int K, int n_tokens, int n_experts,
                                 int n_expert_used, hipStream_t stream);
// The same results as hy4_mmq_moe_vec_typed(..., n_slots, n_experts, 1) for
// n_slots > 1 (slot s reads X row s), ordered by expert as above.
int hy4_mmq_moe_vec_slots_typed(int type, const void* W, const float* X_f32,
                                const int32_t* ids, float* out_f32, int M,
                                int K, int n_slots, int n_experts,
                                hipStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif
