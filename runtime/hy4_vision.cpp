// Hy4 Flash vision tower and merger (docs/PLAN.md, decision 19), after
// Transformers' HYV4VLVisionModel.
#include "runtime/hy4_vision.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/sha256.hpp>
#include <string_view>
#include <type_traits>

#include "kernels/rocm/hy4_kernels.h"
#include "kernels/rocm/hy4_vision.h"

namespace gufo::models::hy4_flash {
namespace {

constexpr std::uint32_t kHidden = 1152;
constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kHeadDim = kHidden / kHeads;
constexpr std::uint32_t kFfn = 4304;
constexpr std::uint32_t kLayers = 27;
constexpr std::uint32_t kPatchValues =
    3 * VisionEncoder::kPatch * VisionEncoder::kPatch;
constexpr float kNormEps = 1e-5f;

// Binary16 bits of v, rounded to nearest even (the weights are well inside
// the binary16 range).
std::uint16_t ToHalfBits(float v) {
  std::uint32_t x;
  std::memcpy(&x, &v, sizeof x);
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  const std::int32_t exp =
      static_cast<std::int32_t>((x >> 23) & 0xff) - 127 + 15;
  std::uint32_t mant = x & 0x7fffffu;
  if (exp >= 31)
    return static_cast<std::uint16_t>(sign | 0x7bffu);
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<std::uint16_t>(sign);
    mant |= 0x800000u;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exp);
    std::uint32_t half = mant >> shift;
    const std::uint32_t rest = mant & ((1u << shift) - 1);
    const std::uint32_t mid = 1u << (shift - 1);
    if (rest > mid || (rest == mid && (half & 1)))
      ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
  const std::uint32_t rest = mant & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1)))
    ++half;
  if (half >= 0x7c00u)
    half = 0x7bffu;
  return static_cast<std::uint16_t>(sign | half);
}

bool Fail(std::string* error, std::string message) {
  if (error)
    *error = std::move(message);
  return false;
}

std::string HipError(std::string_view what, hipError_t status) {
  return "hy4 vision: " + std::string(what) + ": " + hipGetErrorString(status);
}

#define HY4V_CHECK(expr)                                       \
  do {                                                         \
    if ((expr) != 0)                                           \
      return Fail(error, "hy4 vision: kernel failed: " #expr); \
  } while (0)

// A device buffer freed on scope exit.
struct Buffer {
  float* data = nullptr;
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  bool Allocate(std::uint64_t floats, std::string* error) {
    const hipError_t status = hipMalloc(&data, floats * sizeof(float));
    if (status != hipSuccess) {
      data = nullptr;
      return Fail(error, HipError("scratch allocation", status));
    }
    return true;
  }
};

}  // namespace

// Matrices are binary16 bits (BF16 weights converted on upload); norms and
// biases FP32.
struct VisionEncoder::Layer {
  const float *ln1_w, *ln1_b, *q_b, *k_b, *v_b, *o_b, *ln2_w, *ln2_b, *up_b,
      *down_b;
  const std::uint16_t *q_w, *k_w, *v_w, *o_w, *up_w, *down_w;
};

DeviceRows::~DeviceRows() {
  if (data_)
    (void)hipFree(data_);
}

VisionEncoder::~VisionEncoder() {
  gemm_.reset();
  for (void* p : allocations_)
    (void)hipFree(p);
}

std::unique_ptr<VisionEncoder> VisionEncoder::Load(const std::string& path,
                                                   std::uint32_t out_width,
                                                   hipStream_t stream,
                                                   std::string* error) {
  auto reader = core::GgufReader::OpenFile(path, error);
  if (!reader)
    return nullptr;
  const core::GgufReader& r = *reader;
  const auto type = r.GetMetadataString("clip.projector_type");
  if (!type || *type != "hy_v4_flash") {
    Fail(error,
         "hy4 vision: " + path +
             " is not a Hy4 Flash mmproj (clip.projector_type hy_v4_flash)");
    return nullptr;
  }
  if (r.GetMetadataUint32("clip.vision.embedding_length").value_or(0) !=
          kHidden ||
      r.GetMetadataUint32("clip.vision.block_count").value_or(0) != kLayers ||
      r.GetMetadataUint32("clip.vision.attention.head_count").value_or(0) !=
          kHeads ||
      r.GetMetadataUint32("clip.vision.feed_forward_length").value_or(0) !=
          kFfn ||
      r.GetMetadataUint32("clip.vision.patch_size").value_or(0) != kPatch ||
      r.GetMetadataUint32("clip.vision.projection_dim").value_or(0) !=
          out_width) {
    Fail(error, "hy4 vision: unexpected mmproj shape");
    return nullptr;
  }

  std::unique_ptr<VisionEncoder> e(new VisionEncoder());
  e->path_ = path;
  e->out_width_ = out_width;
  e->stream_ = stream;
  e->gemm_ = VisionGemm::Create(stream, error);
  if (!e->gemm_)
    return nullptr;

  // Every weight is uploaded as FP32.
  std::vector<float> host;
  const auto to_f32 = [&](const core::GgufTensorInfo& t) -> bool {
    const std::uint64_t count = t.ElementCount();
    host.resize(count);
    if (t.type == core::GgmlType::kF32) {
      std::memcpy(host.data(), t.data, count * sizeof(float));
    } else if (t.type == core::GgmlType::kBF16) {
      const auto* src = static_cast<const std::uint16_t*>(t.data);
      for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint32_t bits = std::uint32_t{src[i]} << 16;
        std::memcpy(&host[i], &bits, sizeof bits);
      }
    } else {
      return Fail(error, "hy4 vision: " + std::string(t.name) +
                             " is neither F32 nor BF16");
    }
    return true;
  };
  std::vector<std::uint16_t> halves;
  const auto get = [&](const std::string& name, std::uint64_t count,
                       auto** out) -> bool {
    using T = std::remove_const_t<
        std::remove_pointer_t<std::remove_pointer_t<decltype(out)>>>;
    constexpr bool kHalf = std::is_same_v<T, std::uint16_t>;
    const auto* t = r.FindTensor(name);
    if (!t)
      return Fail(error, "hy4 vision: missing tensor " + name);
    if (t->ElementCount() != count)
      return Fail(error, "hy4 vision: unexpected size of " + name);
    if (!to_f32(*t))
      return false;
    const void* source = host.data();
    std::uint64_t bytes = count * sizeof(float);
    if constexpr (kHalf) {
      halves.resize(count);
      for (std::uint64_t i = 0; i < count; ++i)
        halves[i] = ToHalfBits(host[i]);
      source = halves.data();
      bytes = count * sizeof(std::uint16_t);
    }
    void* dev = nullptr;
    const hipError_t status = hipMalloc(&dev, bytes);
    if (status != hipSuccess)
      return Fail(error, HipError("weight allocation", status));
    e->allocations_.push_back(dev);
    if (hipMemcpy(dev, source, bytes, hipMemcpyHostToDevice) != hipSuccess) {
      return Fail(error, "hy4 vision: weight upload failed");
    }
    e->resident_bytes_ += bytes;
    *out = static_cast<const T*>(dev);
    return true;
  };

  bool ok =
      get("v.patch_embd.weight", std::uint64_t{kHidden} * kPatchValues,
          &e->patch_w_) &&
      get("v.patch_embd.bias", kHidden, &e->patch_b_) &&
      get("mm.pre_norm.weight", kHidden, &e->merge_norm_) &&
      get("mm.0.weight", std::uint64_t{4 * kHidden} * 4 * kHidden,
          &e->merge_w_) &&
      get("mm.0.bias", 4 * kHidden, &e->merge_b_) &&
      get("mm.2.weight", std::uint64_t{out_width} * 4 * kHidden, &e->proj_w_) &&
      get("mm.2.bias", out_width, &e->proj_b_) &&
      get("v.image_newline", out_width, &e->newline_);
  e->layers_.resize(kLayers);
  for (std::uint32_t i = 0; ok && i < kLayers; ++i) {
    Layer& l = e->layers_[i];
    const std::string p = "v.blk." + std::to_string(i) + ".";
    const std::uint64_t sq = std::uint64_t{kHidden} * kHidden;
    const std::uint64_t ff = std::uint64_t{kHidden} * kFfn;
    ok = get(p + "ln1.weight", kHidden, &l.ln1_w) &&
         get(p + "ln1.bias", kHidden, &l.ln1_b) &&
         get(p + "attn_q.weight", sq, &l.q_w) &&
         get(p + "attn_q.bias", kHidden, &l.q_b) &&
         get(p + "attn_k.weight", sq, &l.k_w) &&
         get(p + "attn_k.bias", kHidden, &l.k_b) &&
         get(p + "attn_v.weight", sq, &l.v_w) &&
         get(p + "attn_v.bias", kHidden, &l.v_b) &&
         get(p + "attn_out.weight", sq, &l.o_w) &&
         get(p + "attn_out.bias", kHidden, &l.o_b) &&
         get(p + "ln2.weight", kHidden, &l.ln2_w) &&
         get(p + "ln2.bias", kHidden, &l.ln2_b) &&
         get(p + "ffn_up.weight", ff, &l.up_w) &&
         get(p + "ffn_up.bias", kFfn, &l.up_b) &&
         get(p + "ffn_down.weight", ff, &l.down_w) &&
         get(p + "ffn_down.bias", kHidden, &l.down_b);
  }
  if (!ok)
    return nullptr;

  // The learned position grid stays on the host: each image interpolates it.
  const auto* positions = r.FindTensor("v.position_embd.weight");
  if (!positions || positions->dimensions.size() != 2 ||
      positions->dimensions[0] != kHidden || !to_f32(*positions)) {
    Fail(error, "hy4 vision: missing or unexpected v.position_embd.weight");
    return nullptr;
  }
  const auto cells = positions->dimensions[1];
  e->grid_ = static_cast<std::uint32_t>(
      std::lround(std::sqrt(static_cast<double>(cells))));
  if (std::uint64_t{e->grid_} * e->grid_ != cells) {
    Fail(error, "hy4 vision: the position grid is not square");
    return nullptr;
  }
  e->positions_ = std::move(host);

  // One algorithm per GEMM shape, timed on 1,024 rows (VisionGemm).
  constexpr std::uint32_t kTuneRows = 1024;
  const Layer& l = e->layers_[0];
  VisionGemm& g = *e->gemm_;
  if (!g.Tune(e->patch_w_, kHidden, kPatchValues, e->patch_b_, true, kTuneRows,
              error) ||
      !g.Tune(l.q_w, kHidden, kHidden, l.q_b, false, kTuneRows, error) ||
      !g.Tune(l.o_w, kHidden, kHidden, l.o_b, true, kTuneRows, error) ||
      !g.Tune(l.up_w, kFfn, kHidden, l.up_b, false, kTuneRows, error) ||
      !g.Tune(l.down_w, kHidden, kFfn, l.down_b, true, kTuneRows, error) ||
      !g.Tune(e->merge_w_, 4 * kHidden, 4 * kHidden, e->merge_b_, false,
              kTuneRows / 4, error) ||
      !g.Tune(e->proj_w_, out_width, 4 * kHidden, e->proj_b_, false,
              kTuneRows / 4, error)) {
    return nullptr;
  }
  return e;
}

const std::string& VisionEncoder::Identity() const {
  std::call_once(identity_once_,
                 [&] { identity_ = crypto::Sha256FileHex(path_); });
  return identity_;
}

std::shared_ptr<const DeviceRows> VisionEncoder::Encode(const std::uint8_t* rgb,
                                                        std::uint32_t width,
                                                        std::uint32_t height,
                                                        std::string* error) {
  if (width == 0 || height == 0 || width % kFactor != 0 ||
      height % kFactor != 0) {
    Fail(error, "hy4 vision: image sides must be positive multiples of 32");
    return nullptr;
  }
  const std::uint32_t gh = height / kPatch;
  const std::uint32_t gw = width / kPatch;
  const std::uint32_t n = gh * gw;
  const std::uint32_t oh = gh / kMerge;
  const std::uint32_t ow = gw / kMerge;
  const std::uint32_t merged = oh * ow;
  hipStream_t st = stream_;

  // Patches in raster order, each (channel, y, x), rescaled and normalized
  // with mean and std 0.5.
  std::vector<float> patches(std::size_t{n} * kPatchValues);
  for (std::uint32_t py = 0; py < gh; ++py) {
    for (std::uint32_t px = 0; px < gw; ++px) {
      float* out = patches.data() + (std::size_t{py} * gw + px) * kPatchValues;
      for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y = 0; y < kPatch; ++y) {
          for (std::uint32_t x = 0; x < kPatch; ++x) {
            const std::size_t at =
                ((std::size_t{py} * kPatch + y) * width + px * kPatch + x) * 3 +
                c;
            const float v = static_cast<float>(rgb[at]) * 0.00392156862745098f;
            out[(c * kPatch + y) * kPatch + x] = (v - 0.5f) / 0.5f;
          }
        }
      }
    }
  }
  // The position grid, interpolated bilinearly as torch's interpolate with
  // scale_factor (side + 0.1) / grid and align_corners false.
  std::vector<float> pos(std::size_t{n} * kHidden);
  struct Tap {
    std::uint32_t i0, i1;
    float l0, l1;
  };
  const auto taps = [&](std::uint32_t side) {
    std::vector<Tap> result(side);
    const float scale =
        static_cast<float>(1.0 / ((static_cast<double>(side) + 0.1) / grid_));
    for (std::uint32_t d = 0; d < side; ++d) {
      const float src =
          std::max(scale * (static_cast<float>(d) + 0.5f) - 0.5f, 0.0f);
      const auto i0 = static_cast<std::uint32_t>(src);
      const std::uint32_t i1 = i0 + (i0 < grid_ - 1 ? 1 : 0);
      const float l1 = src - static_cast<float>(i0);
      result[d] = {i0, i1, 1.0f - l1, l1};
    }
    return result;
  };
  const auto ty = taps(gh);
  const auto tx = taps(gw);
  for (std::uint32_t y = 0; y < gh; ++y) {
    for (std::uint32_t x = 0; x < gw; ++x) {
      const float* p00 = positions_.data() +
                         (std::size_t{ty[y].i0} * grid_ + tx[x].i0) * kHidden;
      const float* p01 = positions_.data() +
                         (std::size_t{ty[y].i0} * grid_ + tx[x].i1) * kHidden;
      const float* p10 = positions_.data() +
                         (std::size_t{ty[y].i1} * grid_ + tx[x].i0) * kHidden;
      const float* p11 = positions_.data() +
                         (std::size_t{ty[y].i1} * grid_ + tx[x].i1) * kHidden;
      float* out = pos.data() + (std::size_t{y} * gw + x) * kHidden;
      for (std::uint32_t d = 0; d < kHidden; ++d) {
        out[d] = ty[y].l0 * (tx[x].l0 * p00[d] + tx[x].l1 * p01[d]) +
                 ty[y].l1 * (tx[x].l0 * p10[d] + tx[x].l1 * p11[d]);
      }
    }
  }

  // Binary16 attention operands, in floats.
  const std::uint64_t packed =
      std::uint64_t{kHeads} * 80 * hy4_vis_attn_padded(n) / 2;
  Buffer in, x, t, q, k, v, ffn, qh, kh, vt, m1, m2, out, xh;
  const std::uint64_t rows = std::uint64_t{n} * kHidden;
  if (!in.Allocate(std::uint64_t{n} * kPatchValues, error) ||
      !x.Allocate(rows, error) || !t.Allocate(rows, error) ||
      !q.Allocate(rows, error) || !k.Allocate(rows, error) ||
      !v.Allocate(rows, error) ||
      !ffn.Allocate(std::uint64_t{n} * kFfn, error) ||
      !qh.Allocate(packed, error) || !kh.Allocate(packed, error) ||
      !vt.Allocate(packed, error) ||
      !xh.Allocate((std::uint64_t{n} * kFfn + 1) / 2, error) ||
      !m1.Allocate(std::uint64_t{merged} * 4 * kHidden, error) ||
      !m2.Allocate(std::uint64_t{merged} * out_width_, error) ||
      !out.Allocate(std::uint64_t{oh} * (ow + 1) * out_width_, error)) {
    return nullptr;
  }
  // Every GEMM reads its input from h as binary16, written by the step before
  // it: rounded where an FP32 copy was converted before.
  auto* h = reinterpret_cast<std::uint16_t*>(xh.data);
  VisionGemm& g = *gemm_;
  const auto run = [&]() -> bool {
    if (hipMemcpyAsync(in.data, patches.data(), patches.size() * sizeof(float),
                       hipMemcpyHostToDevice, st) != hipSuccess ||
        hipMemcpyAsync(t.data, pos.data(), pos.size() * sizeof(float),
                       hipMemcpyHostToDevice, st) != hipSuccess) {
      return Fail(error, "hy4 vision: upload failed");
    }
    // x = patches W^T + b + positions.
    HY4V_CHECK(
        hy4_vis_to_half(in.data, h, std::uint64_t{n} * kPatchValues, st));
    if (!g.Run(patch_w_, h, x.data, patch_b_, t.data, n, kHidden, kPatchValues,
               error)) {
      return false;
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    for (const Layer& l : layers_) {
      HY4V_CHECK(hy4_vis_layer_norm_half(x.data, l.ln1_w, l.ln1_b, h, n,
                                         kHidden, kNormEps, st));
      if (!g.Run(l.q_w, h, q.data, l.q_b, nullptr, n, kHidden, kHidden,
                 error) ||
          !g.Run(l.k_w, h, k.data, l.k_b, nullptr, n, kHidden, kHidden,
                 error) ||
          !g.Run(l.v_w, h, v.data, l.v_b, nullptr, n, kHidden, kHidden,
                 error)) {
        return false;
      }
      HY4V_CHECK(hy4_vis_attention(
          q.data, k.data, v.data, scale, reinterpret_cast<uint16_t*>(qh.data),
          reinterpret_cast<uint16_t*>(kh.data),
          reinterpret_cast<uint16_t*>(vt.data), h, n, st));
      if (!g.Run(l.o_w, h, x.data, l.o_b, x.data, n, kHidden, kHidden, error))
        return false;

      HY4V_CHECK(hy4_vis_layer_norm_half(x.data, l.ln2_w, l.ln2_b, h, n,
                                         kHidden, kNormEps, st));
      if (!g.Run(l.up_w, h, ffn.data, l.up_b, nullptr, n, kFfn, kHidden,
                 error)) {
        return false;
      }
      HY4V_CHECK(hy4_vis_gelu_half(ffn.data, h, std::uint64_t{n} * kFfn, st));
      if (!g.Run(l.down_w, h, x.data, l.down_b, x.data, n, kHidden, kFfn,
                 error)) {
        return false;
      }
    }

    // Merger: RMSNorm, the 2x2 convolution, GELU, the 1x1 convolution, and a
    // newline after every merged row.
    HY4V_CHECK(hy4_rms_norm(x.data, merge_norm_, t.data, n, kHidden, kHidden,
                            kNormEps, st));
    HY4V_CHECK(hy4_vis_merge_gather(t.data, gh, gw, kHidden, ffn.data, st));
    HY4V_CHECK(hy4_vis_to_half(ffn.data, h, std::uint64_t{n} * kHidden, st));
    if (!g.Run(merge_w_, h, m1.data, merge_b_, nullptr, merged, 4 * kHidden,
               4 * kHidden, error)) {
      return false;
    }
    HY4V_CHECK(
        hy4_vis_gelu_half(m1.data, h, std::uint64_t{merged} * 4 * kHidden, st));
    if (!g.Run(proj_w_, h, m2.data, proj_b_, nullptr, merged, out_width_,
               4 * kHidden, error)) {
      return false;
    }
    HY4V_CHECK(
        hy4_vis_newline(m2.data, newline_, out.data, oh, ow, out_width_, st));
    const hipError_t status = hipStreamSynchronize(st);
    if (status != hipSuccess)
      return Fail(error, HipError("encode", status));
    return true;
  };
  if (!run()) {
    (void)hipStreamSynchronize(st);
    return nullptr;
  }
  auto result =
      std::make_shared<DeviceRows>(out.data, oh * (ow + 1), out_width_);
  out.data = nullptr;
  return result;
}

}  // namespace gufo::models::hy4_flash
