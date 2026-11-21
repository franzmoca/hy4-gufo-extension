// The algorithm search is adapted from Gufo's EmbeddingGemma 2 encoder
// (src/models/embedding_gemma2/hip/blas.cpp); see UPSTREAM.md.

#include "runtime/hy4_vision_gemm.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <string>
#include <vector>

#include "kernels/rocm/hy4_vision.h"

namespace gufo::models::hy4_flash {
namespace {

constexpr int kCandidates = 32;
constexpr int kTimedRuns = 3;
constexpr std::size_t kWorkspaceBytes = std::size_t{32} << 20;

bool Fail(std::string* error, const std::string& message) {
  if (error)
    *error = message;
  return false;
}

// The column-major problem hipBLASLt sees: y^T[out][rows] = W x^T.
struct Layouts {
  hipblasLtMatrixLayout_t w = nullptr, x = nullptr, y = nullptr;
  Layouts(std::uint32_t rows, std::uint32_t out, std::uint32_t in) {
    if (hipblasLtMatrixLayoutCreate(&w, HIP_R_16F, in, out, in) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&x, HIP_R_16F, in, rows, in) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&y, HIP_R_32F, out, rows, out) !=
            HIPBLAS_STATUS_SUCCESS) {
      y = nullptr;
    }
  }
  ~Layouts() {
    for (auto layout : {w, x, y}) {
      if (layout)
        (void)hipblasLtMatrixLayoutDestroy(layout);
    }
  }
  Layouts(const Layouts&) = delete;
  Layouts& operator=(const Layouts&) = delete;
  [[nodiscard]] bool ok() const { return y != nullptr; }
};

}  // namespace

struct VisionGemm::Plan {
  hipblasLtMatmulDesc_t op = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  bool usable = false;

  ~Plan() {
    if (op)
      (void)hipblasLtMatmulDescDestroy(op);
  }
};

std::unique_ptr<VisionGemm> VisionGemm::Create(hipStream_t stream,
                                               std::string* error) {
  std::unique_ptr<VisionGemm> g(new VisionGemm());
  g->stream_ = stream;
  hipblasHandle_t blas = nullptr;
  if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS) {
    Fail(error, "hy4 vision: hipblasCreate failed");
    return nullptr;
  }
  g->blas_ = blas;
  (void)hipblasSetStream(blas, stream);
  hipblasLtHandle_t lt = nullptr;
  if (hipblasLtCreate(&lt) == HIPBLAS_STATUS_SUCCESS &&
      hipMalloc(&g->workspace_, kWorkspaceBytes) == hipSuccess) {
    g->lt_ = lt;
  } else {
    if (lt)
      (void)hipblasLtDestroy(lt);
    g->workspace_ = nullptr;
    (void)hipGetLastError();
  }
  return g;
}

VisionGemm::~VisionGemm() {
  plans_.clear();
  if (lt_)
    (void)hipblasLtDestroy(static_cast<hipblasLtHandle_t>(lt_));
  if (workspace_)
    (void)hipFree(workspace_);
  if (blas_)
    (void)hipblasDestroy(static_cast<hipblasHandle_t>(blas_));
}

std::size_t VisionGemm::TunedShapes() const {
  std::size_t count = 0;
  for (const auto& [key, plan] : plans_)
    count += plan->usable ? 1 : 0;
  return count;
}

bool VisionGemm::Tune(const std::uint16_t* w, std::uint32_t out,
                      std::uint32_t in, const float* bias, bool with_c,
                      std::uint32_t rows, std::string* error) {
  const Key key{out, in, bias != nullptr, with_c};
  if (!lt_ || plans_.contains(key))
    return true;
  auto& plan = plans_[key];
  plan = std::make_unique<Plan>();
  const auto lt = static_cast<hipblasLtHandle_t>(lt_);
  const hipblasOperation_t transpose = HIPBLAS_OP_T;
  const hipblasLtEpilogue_t epilogue =
      bias ? HIPBLASLT_EPILOGUE_BIAS : HIPBLASLT_EPILOGUE_DEFAULT;
  const hipDataType bias_type = HIP_R_32F;
  Layouts layouts(rows, out, in);
  if (hipblasLtMatmulDescCreate(&plan->op, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(plan->op, HIPBLASLT_MATMUL_DESC_TRANSA,
                                      &transpose, sizeof transpose) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(plan->op, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                      &epilogue, sizeof epilogue) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(
          plan->op, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof bias_type) != HIPBLAS_STATUS_SUCCESS ||
      (bias && hipblasLtMatmulDescSetAttribute(
                   plan->op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                   sizeof bias) != HIPBLAS_STATUS_SUCCESS) ||
      !layouts.ok()) {
    return true;  // hipBLAS serves this shape
  }
  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS)
    return true;
  const std::uint64_t workspace = kWorkspaceBytes;
  (void)hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace,
      sizeof workspace);
  std::vector<hipblasLtMatmulHeuristicResult_t> results(kCandidates);
  int count = 0;
  const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
      lt, plan->op, layouts.w, layouts.x, layouts.y, layouts.y, preference,
      kCandidates, results.data(), &count);
  (void)hipblasLtMatmulPreferenceDestroy(preference);
  if (status != HIPBLAS_STATUS_SUCCESS || count <= 0)
    return true;

  // Scratch operands; timing does not depend on their values.
  void* x = nullptr;
  void* y = nullptr;
  hipEvent_t begin = nullptr, end = nullptr;
  const auto release = [&] {
    if (x)
      (void)hipFree(x);
    if (y)
      (void)hipFree(y);
    if (begin)
      (void)hipEventDestroy(begin);
    if (end)
      (void)hipEventDestroy(end);
  };
  if (hipMalloc(&x, std::size_t{rows} * in * 2) != hipSuccess ||
      hipMalloc(&y, std::size_t{rows} * out * 4) != hipSuccess ||
      hipMemsetAsync(x, 0, std::size_t{rows} * in * 2, stream_) != hipSuccess ||
      hipMemsetAsync(y, 0, std::size_t{rows} * out * 4, stream_) !=
          hipSuccess ||
      hipEventCreate(&begin) != hipSuccess ||
      hipEventCreate(&end) != hipSuccess) {
    release();
    return Fail(error, "hy4 vision: GEMM tuning allocation failed");
  }
  const float one = 1.0f;
  const float beta = with_c ? 1.0f : 0.0f;
  float best = -1.0f;
  for (int i = 0; i < count; ++i) {
    if (results[i].state != HIPBLAS_STATUS_SUCCESS ||
        results[i].workspaceSize > kWorkspaceBytes) {
      continue;
    }
    const auto launch = [&] {
      return hipblasLtMatmul(lt, plan->op, &one, w, layouts.w, x, layouts.x,
                             &beta, y, layouts.y, y, layouts.y,
                             &results[i].algo, workspace_, kWorkspaceBytes,
                             stream_) == HIPBLAS_STATUS_SUCCESS;
    };
    // The first launch loads the kernel and is not timed.
    bool launched = launch() && hipEventRecord(begin, stream_) == hipSuccess;
    for (int run = 0; launched && run < kTimedRuns; ++run)
      launched = launch();
    float ms = -1.0f;
    if (launched && hipEventRecord(end, stream_) == hipSuccess &&
        hipEventSynchronize(end) == hipSuccess) {
      (void)hipEventElapsedTime(&ms, begin, end);
    }
    if (ms < 0.0f) {
      (void)hipGetLastError();
      continue;
    }
    if (best < 0.0f || ms < best) {
      best = ms;
      plan->algo = results[i].algo;
      plan->usable = true;
    }
  }
  const hipError_t synced = hipStreamSynchronize(stream_);
  release();
  if (synced != hipSuccess)
    return Fail(error, "hy4 vision: GEMM tuning failed");
  return true;
}

bool VisionGemm::Run(const std::uint16_t* w, const std::uint16_t* x, float* y,
                     const float* bias, const float* c, std::uint32_t rows,
                     std::uint32_t out, std::uint32_t in, std::string* error) {
  const auto found = plans_.find(Key{out, in, bias != nullptr, c != nullptr});
  if (found == plans_.end() || !found->second->usable)
    return Fallback(w, x, y, bias, c, rows, out, in, error);
  Plan& plan = *found->second;
  const auto lt = static_cast<hipblasLtHandle_t>(lt_);
  if (bias && hipblasLtMatmulDescSetAttribute(
                  plan.op, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                  sizeof bias) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error, "hy4 vision: GEMM bias setup failed");
  }
  Layouts layouts(rows, out, in);
  if (!layouts.ok())
    return Fail(error, "hy4 vision: GEMM layout setup failed");
  const float one = 1.0f;
  const float beta = c ? 1.0f : 0.0f;
  hipblasLtMatmulAlgo_t algo = plan.algo;
  std::size_t workspace = 0;
  if (hipblaslt_ext::matmulIsAlgoSupported(
          lt, plan.op, &one, layouts.w, layouts.x, &beta, layouts.y, layouts.y,
          algo, workspace) != HIPBLAS_STATUS_SUCCESS ||
      workspace > kWorkspaceBytes) {
    return Fallback(w, x, y, bias, c, rows, out, in, error);
  }
  if (hipblasLtMatmul(lt, plan.op, &one, w, layouts.w, x, layouts.x, &beta,
                      c ? c : y, layouts.y, y, layouts.y, &algo, workspace_,
                      kWorkspaceBytes, stream_) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error, "hy4 vision: GEMM failed");
  }
  return true;
}

bool VisionGemm::Fallback(const std::uint16_t* w, const std::uint16_t* x,
                          float* y, const float* bias, const float* c,
                          std::uint32_t rows, std::uint32_t out,
                          std::uint32_t in, std::string* error) {
  if (c && c != y &&
      hipMemcpyAsync(y, c, std::size_t{rows} * out * sizeof(float),
                     hipMemcpyDeviceToDevice, stream_) != hipSuccess) {
    return Fail(error, "hy4 vision: GEMM copy failed");
  }
  const float one = 1.0f;
  const float beta = c ? 1.0f : 0.0f;
  if (hipblasGemmEx(static_cast<hipblasHandle_t>(blas_), HIPBLAS_OP_T,
                    HIPBLAS_OP_N, static_cast<int>(out), static_cast<int>(rows),
                    static_cast<int>(in), &one, w, HIP_R_16F,
                    static_cast<int>(in), x, HIP_R_16F, static_cast<int>(in),
                    &beta, y, HIP_R_32F, static_cast<int>(out),
                    HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error, "hy4 vision: GEMM failed");
  }
  if (bias && hy4_vis_bias(y, bias, rows, out, 0, stream_) != 0)
    return Fail(error, "hy4 vision: bias failed");
  return true;
}

}  // namespace gufo::models::hy4_flash
