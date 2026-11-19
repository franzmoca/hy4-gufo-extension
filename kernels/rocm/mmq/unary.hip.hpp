#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void hy4_ggml_hip_op_abs(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_sgn(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_neg(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_step(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_gelu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_silu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_silu_back(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_gelu_erf(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_gelu_quick(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_tanh(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_relu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_sigmoid(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_hardsigmoid(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_exp(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_hardswish(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_leaky_relu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_sqr(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_sqrt(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_sin(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_cos(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_log(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_expm1(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_softplus(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_elu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_floor(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_ceil(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_round(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_trunc(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_reglu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_geglu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_swiglu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_swiglu_oai(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_geglu_erf(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_geglu_quick(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_xielu(hy4_ggml_hip_context & ctx, ggml_tensor * dst);

void hy4_ggml_hip_op_unary_mul(hy4_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void hy4_ggml_hip_op_relu_sqr(hy4_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float hy4_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float hy4_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float hy4_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
