#ifndef HY4FLASH_RUNTIME_HY4_VISION_GEMM_HPP_
#define HY4FLASH_RUNTIME_HY4_VISION_GEMM_HPP_

// The vision tower's GEMMs on hipBLASLt (docs/EXPERIMENTS.md, E61).

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>

namespace gufo::models::hy4_flash {

/// y[rows][out] = x[rows][in] W[out][in]^T + bias, plus c[rows][out] when
/// given (c may be y): binary16 operands, FP32 accumulation and result.
///
/// rocBLAS has no matrix-core kernel for binary16 operands with an FP32
/// result on gfx1151 and runs a scalar one at about 6 TFLOPS. hipBLASLt's
/// first heuristic choice is that kernel too, but a later one reaches 28 to 39
/// TFLOPS from 1,024 to 16,384 rows on the tower's shapes, the same one at
/// every row count. So each shape's candidates are timed once, at load, and
/// the fastest serves every row count it supports; hipBLAS serves the rest.
/// Not thread-safe: the encoder serializes its calls.
class VisionGemm {
public:
  /// Null with `error` set when hipBLAS cannot start; hipBLASLt is optional.
  [[nodiscard]] static std::unique_ptr<VisionGemm> Create(hipStream_t stream,
                                                          std::string* error);
  ~VisionGemm();
  VisionGemm(const VisionGemm&) = delete;
  VisionGemm& operator=(const VisionGemm&) = delete;

  /// Picks the algorithm for W's shape and whether calls pass a bias and c,
  /// timing the candidates on `rows` rows of scratch input. Returns false
  /// only for device errors; without a usable algorithm calls use hipBLAS.
  [[nodiscard]] bool Tune(const std::uint16_t* w, std::uint32_t out,
                          std::uint32_t in, const float* bias, bool with_c,
                          std::uint32_t rows, std::string* error);
  [[nodiscard]] bool Run(const std::uint16_t* w, const std::uint16_t* x,
                         float* y, const float* bias, const float* c,
                         std::uint32_t rows, std::uint32_t out,
                         std::uint32_t in, std::string* error);

  /// Shapes that run on hipBLASLt, of those tuned.
  [[nodiscard]] std::size_t TunedShapes() const;

private:
  struct Plan;
  using Key = std::tuple<std::uint32_t, std::uint32_t, bool, bool>;
  VisionGemm() = default;
  bool Fallback(const std::uint16_t* w, const std::uint16_t* x, float* y,
                const float* bias, const float* c, std::uint32_t rows,
                std::uint32_t out, std::uint32_t in, std::string* error);

  hipStream_t stream_ = nullptr;
  void* blas_ = nullptr;  // hipblasHandle_t
  void* lt_ = nullptr;    // hipblasLtHandle_t
  void* workspace_ = nullptr;
  std::map<Key, std::unique_ptr<Plan>> plans_;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_VISION_GEMM_HPP_
