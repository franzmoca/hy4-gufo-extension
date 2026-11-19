#ifndef HY4FLASH_RUNTIME_HY4_VISION_HPP_
#define HY4FLASH_RUNTIME_HY4_VISION_HPP_

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gufo::models::hy4_flash {

/// Device rows [rows][width] of FP32 language-model input embeddings.
class DeviceRows {
public:
  DeviceRows(float* data, std::uint32_t rows, std::uint32_t width)
      : data_(data), rows_(rows), width_(width) {}
  ~DeviceRows();
  DeviceRows(const DeviceRows&) = delete;
  DeviceRows& operator=(const DeviceRows&) = delete;
  [[nodiscard]] const float* data() const { return data_; }
  [[nodiscard]] std::uint32_t rows() const { return rows_; }
  [[nodiscard]] std::uint32_t width() const { return width_; }

private:
  float* data_;
  std::uint32_t rows_;
  std::uint32_t width_;
};

/// Hy4 Flash's vision tower and merger from the `mmproj` BF16 GGUF, in FP32
/// on the GPU (docs/PLAN.md, decision 19): an image whose sides are
/// multiples of kFactor becomes (h / kFactor) * (w / kFactor + 1) rows.
class VisionEncoder {
public:
  static constexpr std::uint32_t kPatch = 16;
  static constexpr std::uint32_t kMerge = 2;
  static constexpr std::uint32_t kFactor = kPatch * kMerge;

  ~VisionEncoder();
  VisionEncoder(const VisionEncoder&) = delete;
  VisionEncoder& operator=(const VisionEncoder&) = delete;

  /// `out_width` is the language model's hidden size; work runs on `stream`.
  [[nodiscard]] static std::unique_ptr<VisionEncoder> Load(
      const std::string& path, std::uint32_t out_width, hipStream_t stream,
      std::string* error);
  /// SHA-256 (hex) of the GGUF file, computed once.
  [[nodiscard]] const std::string& Identity() const;
  [[nodiscard]] std::uint64_t ResidentBytes() const { return resident_bytes_; }

  /// Encodes RGB8 pixels [height][width][3]; both sides multiples of
  /// kFactor. Calls are serialized by the caller.
  [[nodiscard]] std::shared_ptr<const DeviceRows> Encode(
      const std::uint8_t* rgb, std::uint32_t width, std::uint32_t height,
      std::string* error);

private:
  struct Layer;
  VisionEncoder() = default;
  bool Linear(const std::uint16_t* w, const float* x, float* y,
              std::uint32_t rows, std::uint32_t out, std::uint32_t in,
              std::string* error);

  std::string path_;
  mutable std::once_flag identity_once_;
  mutable std::string identity_;
  std::uint32_t out_width_ = 0;
  hipStream_t stream_ = nullptr;
  void* blas_ = nullptr;  // hipblasHandle_t
  std::vector<Layer> layers_;
  // Matrices are binary16 bits.
  const std::uint16_t* patch_w_ = nullptr;  // [hidden][3 * kPatch * kPatch]
  const float* patch_b_ = nullptr;
  std::vector<float> positions_;  // [grid][grid][hidden], host
  std::uint32_t grid_ = 0;        // learned position grid side
  const float* merge_norm_ = nullptr;
  const std::uint16_t* merge_w_ = nullptr;  // [4 hidden][4 hidden]
  const float* merge_b_ = nullptr;
  const std::uint16_t* proj_w_ = nullptr;  // [out_width][4 hidden]
  const float* proj_b_ = nullptr;
  const float* newline_ = nullptr;  // [out_width]
  std::uint16_t* xh_ = nullptr;     // Encode's binary16 GEMM input
  std::vector<void*> allocations_;
  std::uint64_t resident_bytes_ = 0;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_VISION_HPP_
