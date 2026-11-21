#ifndef HY4FLASH_KERNELS_ROCM_HY4_VISION_H_
#define HY4FLASH_KERNELS_ROCM_HY4_VISION_H_

/* Kernels of Hy4 Flash's vision tower and merger (runtime/hy4_vision.cpp).
 * FP32, row-major [rows][dim]. Each returns 0, or 1 on a launch error. */

#include <hip/hip_runtime_api.h>
#include <stdint.h>

/* y = (x - mean) / sqrt(var + eps) * weight + bias, per row. */
int hy4_vis_layer_norm(const float* x, const float* weight, const float* bias,
                       float* y, uint32_t rows, uint32_t dim, float eps,
                       hipStream_t stream);

/* hy4_vis_layer_norm into binary16 bits, saturated to the finite range. */
int hy4_vis_layer_norm_half(const float* x, const float* weight,
                            const float* bias, uint16_t* y, uint32_t rows,
                            uint32_t dim, float eps, hipStream_t stream);

/* out = binary16 bits of GELU (erf) of y, saturated to the finite range. */
int hy4_vis_gelu_half(const float* y, uint16_t* out, uint64_t count,
                      hipStream_t stream);

/* y = y + bias (per column), then GELU (erf) when `gelu`. */
int hy4_vis_bias(float* y, const float* bias, uint32_t rows, uint32_t dim,
                 int gelu, hipStream_t stream);

/* The 2x2 stride-2 convolution's input columns: x is the patch grid
 * [gh][gw][dim]; out row (i * gw/2 + j) holds, at ic * 4 + kh * 2 + kw, patch
 * (2i + kh, 2j + kw)'s channel ic (the GGUF kernel's [oc][ic][kh][kw] order).
 */
int hy4_vis_merge_gather(const float* x, uint32_t gh, uint32_t gw, uint32_t dim,
                         float* out, hipStream_t stream);

/* dst [oh][ow + 1][dim]: the rows of src [oh][ow][dim], each merged row
 * followed by `newline`. */
int hy4_vis_newline(const float* src, const float* newline, float* dst,
                    uint32_t oh, uint32_t ow, uint32_t dim, hipStream_t stream);

/* y = binary16 bits of x, saturated to the finite range. */
int hy4_vis_to_half(const float* x, uint16_t* y, uint64_t count,
                    hipStream_t stream);

/* Keys per padded attention operand: count rounded up to whole key tiles. */
uint32_t hy4_vis_attn_padded(uint32_t count);

/* Bidirectional attention of 16 heads of 72 over `count` FP32 rows q, k, v
 * [count][1152] into out [count][1152], softmax(scale q k^T) v, on the matrix
 * cores in binary16 with FP32 accumulation and softmax; out holds binary16
 * bits, saturated to the finite range. qh, kh and vt are scratch of
 * 16 * 80 * hy4_vis_attn_padded(count) binary16 values each. */
int hy4_vis_attention(const float* q, const float* k, const float* v,
                      float scale, uint16_t* qh, uint16_t* kh, uint16_t* vt,
                      uint16_t* out, uint32_t count, hipStream_t stream);

#endif /* HY4FLASH_KERNELS_ROCM_HY4_VISION_H_ */
