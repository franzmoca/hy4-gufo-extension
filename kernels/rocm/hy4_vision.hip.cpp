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

__global__ void Softmax(float* scores, uint32_t cols) {
  __shared__ float scratch[32];
  float* row = scores + (uint64_t)blockIdx.x * cols;
  float m = -INFINITY;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x)
    m = fmaxf(m, row[i]);
  m = BlockReduce(m, scratch, true);
  float sum = 0.0f;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    const float e = expf(row[i] - m);
    row[i] = e;
    sum += e;
  }
  const float inv = 1.0f / BlockReduce(sum, scratch, false);
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x)
    row[i] *= inv;
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

}  // namespace

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

int hy4_vis_softmax(float* scores, uint32_t rows, uint32_t cols,
                    hipStream_t stream) {
  if (rows == 0)
    return 0;
  Softmax<<<rows, kThreads, 0, stream>>>(scores, cols);
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
