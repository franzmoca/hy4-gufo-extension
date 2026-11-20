#pragma once
#include "common.hip.hpp"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int hy4_get_mmvq_mmid_max_batch(ggml_type type, int cc);

// ds4: direct switch over the per-type templated mmvq dispatch.  Was
// `static` upstream; we promote it so hy4_mmq.cu can call it without
// constructing ggml_tensor structs.  Args match the original signature
// exactly.
void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const hy4_ggml_hip_mm_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, hipStream_t stream);

// hy4: routed expert matvecs of a multi-token pass in one launch, bitwise
// equal to the one-token launches, the slots ordered by expert so that
// repeated experts read their rows from the caches (see mmvq.hip.cpp).
// per_token: slot s reads y row s / slots_per_row as mul_mat_vec_q<type, 1>
// would for that token; else slot s reads y row s as
// mul_mat_vec_q_moe<type, 2>. dst rows are nrows_x floats per slot; order is
// n_slots (<= 1024) words of scratch. Returns false for an unsupported type.
bool hy4_mmvq_moe_shared(
        const void * vx, ggml_type type, const void * vy, const int32_t * ids, uint32_t * order, float * dst,
        int ncols_x, int nrows_x, int stride_row_x, int stride_col_y, int stride_channel_x,
        int n_slots, int slots_per_row, bool per_token, hipStream_t stream);
