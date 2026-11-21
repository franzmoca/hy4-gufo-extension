// The vision tower kernels (kernels/rocm/hy4_vision.h) against double
// precision host references: norms, GELU, the merger's gather and newline
// layout, binary16 conversion, the fused attention over counts that are
// and are not whole key tiles, and the tower's GEMMs (runtime/hy4_vision_gemm)
// on their shapes and with their bias and accumulation, at row counts other
// than the one their algorithm was timed on.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "kernels/rocm/hy4_vision.h"
#include "runtime/hy4_vision_gemm.hpp"

namespace {

int g_failures = 0;
std::mt19937 g_rng(1152);

std::vector<float> Random(std::size_t count, float lo = -1.0f,
                          float hi = 1.0f) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(count);
  for (float& x : v)
    x = dist(g_rng);
  return v;
}

template<typename T>
T* Device(const std::vector<T>& host) {
  void* p = nullptr;
  if (hipMalloc(&p, std::max<std::size_t>(host.size(), 1) * sizeof(T)) !=
      hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    std::exit(1);
  }
  if (!host.empty())
    (void)hipMemcpy(p, host.data(), host.size() * sizeof(T),
                    hipMemcpyHostToDevice);
  return static_cast<T*>(p);
}

template<typename T>
std::vector<T> Host(const T* device, std::size_t count) {
  std::vector<T> host(count);
  (void)hipDeviceSynchronize();
  (void)hipMemcpy(host.data(), device, count * sizeof(T),
                  hipMemcpyDeviceToHost);
  return host;
}

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

// Max |got - want| relative to the largest |want|.
void Near(const std::vector<float>& got, const std::vector<double>& want,
          double tol, const std::string& what) {
  double scale = 1e-3, worst = 0.0;
  std::size_t at = 0;
  for (double w : want)
    scale = std::max(scale, std::fabs(w));
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = std::fabs(got[i] - want[i]);
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  if (got.size() != want.size() || !(worst / scale <= tol)) {
    std::fprintf(stderr, "FAIL %s: error %.3g at %zu (got %.7g, want %.7g)\n",
                 what.c_str(), worst / scale, at, got[at], want[at]);
    ++g_failures;
  }
}

double Gelu(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double HalfValue(uint16_t h) {
  const int exp = (h >> 10) & 0x1f;
  const int mant = h & 0x3ff;
  double v;
  if (exp == 31)
    v = mant ? NAN : INFINITY;
  else
    v = exp == 0 ? std::ldexp(mant, -24) : std::ldexp(1024 + mant, exp - 25);
  return h & 0x8000 ? -v : v;
}

std::vector<float> FromHalf(const uint16_t* device, std::size_t count) {
  std::vector<float> out;
  for (const uint16_t h : Host(device, count))
    out.push_back((float)HalfValue(h));
  return out;
}

void TestRows() {
  const uint32_t rows = 7, dim = 1152;
  const auto x = Random(rows * dim, -4, 6);
  const auto w = Random(dim, 0.5f, 1.5f);
  const auto b = Random(dim);
  float* y = Device(std::vector<float>(rows * dim));
  Expect(hy4_vis_layer_norm(Device(x), Device(w), Device(b), y, rows, dim,
                            1e-5f, nullptr) == 0,
         "layer_norm status");
  std::vector<double> want(rows * dim);
  for (uint32_t r = 0; r < rows; ++r) {
    double mean = 0, var = 0;
    for (uint32_t c = 0; c < dim; ++c)
      mean += x[r * dim + c];
    mean /= dim;
    for (uint32_t c = 0; c < dim; ++c)
      var += (x[r * dim + c] - mean) * (x[r * dim + c] - mean);
    var /= dim;
    for (uint32_t c = 0; c < dim; ++c)
      want[r * dim + c] =
          (x[r * dim + c] - mean) / std::sqrt(var + 1e-5) * w[c] + b[c];
  }
  Near(Host(y, rows * dim), want, 2e-6, "layer_norm");
  // Into binary16: rounding, at most 2^-11 of the largest value.
  uint16_t* yh = Device(std::vector<uint16_t>(rows * dim));
  Expect(hy4_vis_layer_norm_half(Device(x), Device(w), Device(b), yh, rows, dim,
                                 1e-5f, nullptr) == 0,
         "layer_norm_half status");
  Near(FromHalf(yh, rows * dim), want, 5e-4, "layer_norm_half");

  uint16_t* gh = Device(std::vector<uint16_t>(rows * dim));
  Expect(hy4_vis_gelu_half(Device(x), gh, rows * dim, nullptr) == 0,
         "gelu_half status");
  std::vector<double> want_gelu(rows * dim);
  for (uint32_t i = 0; i < rows * dim; ++i)
    want_gelu[i] = Gelu(x[i]);
  Near(FromHalf(gh, rows * dim), want_gelu, 5e-4, "gelu_half");

  for (int gelu : {0, 1}) {
    float* z = Device(x);
    Expect(hy4_vis_bias(z, Device(b), rows, dim, gelu, nullptr) == 0,
           "bias status");
    std::vector<double> want_bias(rows * dim);
    for (uint32_t i = 0; i < rows * dim; ++i) {
      const double v = (double)x[i] + b[i % dim];
      want_bias[i] = gelu ? Gelu(v) : v;
    }
    Near(Host(z, rows * dim), want_bias, 1e-6, gelu ? "bias gelu" : "bias");
  }
}

void TestMerger() {
  const uint32_t gh = 6, gw = 8, dim = 40, oh = gh / 2, ow = gw / 2;
  const auto x = Random(gh * gw * dim);
  float* gathered = Device(std::vector<float>(oh * ow * 4 * dim));
  Expect(hy4_vis_merge_gather(Device(x), gh, gw, dim, gathered, nullptr) == 0,
         "merge_gather status");
  std::vector<double> want(oh * ow * 4 * dim);
  for (uint32_t i = 0; i < oh; ++i)
    for (uint32_t j = 0; j < ow; ++j)
      for (uint32_t ic = 0; ic < dim; ++ic)
        for (uint32_t kh = 0; kh < 2; ++kh)
          for (uint32_t kw = 0; kw < 2; ++kw)
            want[(i * ow + j) * 4 * dim + ic * 4 + kh * 2 + kw] =
                x[((2 * i + kh) * gw + 2 * j + kw) * dim + ic];
  Near(Host(gathered, want.size()), want, 0, "merge_gather");

  const auto src = Random(oh * ow * dim);
  const auto newline = Random(dim);
  float* dst = Device(std::vector<float>(oh * (ow + 1) * dim));
  Expect(hy4_vis_newline(Device(src), Device(newline), dst, oh, ow, dim,
                         nullptr) == 0,
         "newline status");
  std::vector<double> want_nl(oh * (ow + 1) * dim);
  for (uint32_t i = 0; i < oh; ++i) {
    for (uint32_t j = 0; j < ow; ++j)
      for (uint32_t c = 0; c < dim; ++c)
        want_nl[(i * (ow + 1) + j) * dim + c] = src[(i * ow + j) * dim + c];
    for (uint32_t c = 0; c < dim; ++c)
      want_nl[(i * (ow + 1) + ow) * dim + c] = newline[c];
  }
  Near(Host(dst, want_nl.size()), want_nl, 0, "newline");
}

void TestToHalf() {
  std::vector<float> x{0.0f,     -0.0f,    1.0f,  -2.5f, 1.0f / 3.0f,
                       65504.0f, 70000.0f, -1e9f, 6e-5f, 3e-8f};
  const auto more = Random(1000, -100, 100);
  x.insert(x.end(), more.begin(), more.end());
  uint16_t* y = Device(std::vector<uint16_t>(x.size()));
  Expect(hy4_vis_to_half(Device(x), y, x.size(), nullptr) == 0,
         "to_half status");
  const auto got = Host(y, x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double want = std::clamp((double)x[i], -65504.0, 65504.0);
    const double v = HalfValue(got[i]);
    // Round to nearest: within half an ulp (2^-11 relative, 2^-25 absolute
    // for subnormals).
    if (!(std::fabs(v - want) <=
          std::max(std::fabs(want) * std::ldexp(1.0, -11),
                   std::ldexp(1.0, -25)))) {
      Expect(false, "to_half of " + std::to_string(x[i]) + " gave " +
                        std::to_string(v));
      return;
    }
  }
}

void TestAttention() {
  constexpr uint32_t kHeads = 16, kHeadDim = 72, kHidden = kHeads * kHeadDim;
  const float scale = 1.0f / std::sqrt((float)kHeadDim);
  Expect(hy4_vis_attn_padded(1) == 64 && hy4_vis_attn_padded(64) == 64 &&
             hy4_vis_attn_padded(65) == 128,
         "attn_padded");
  for (uint32_t count : {1u, 48u, 64u, 200u}) {
    const auto q = Random(count * kHidden, -2, 2);
    const auto k = Random(count * kHidden, -2, 2);
    const auto v = Random(count * kHidden, -2, 2);
    const std::size_t halves =
        (std::size_t)kHeads * 80 * hy4_vis_attn_padded(count);
    // Scratch starts dirty: the kernel must clear its own padding.
    std::vector<uint16_t> dirty(halves, 0x3c00);
    uint16_t* out = Device(std::vector<uint16_t>(count * kHidden));
    Expect(hy4_vis_attention(Device(q), Device(k), Device(v), scale,
                             Device(dirty), Device(dirty), Device(dirty), out,
                             count, nullptr) == 0,
           "attention status");
    std::vector<double> want(count * kHidden, 0.0);
    std::vector<double> s(count);
    for (uint32_t h = 0; h < kHeads; ++h)
      for (uint32_t i = 0; i < count; ++i) {
        double mx = -INFINITY, den = 0;
        for (uint32_t j = 0; j < count; ++j) {
          double d = 0;
          for (uint32_t c = 0; c < kHeadDim; ++c)
            d += (double)q[i * kHidden + h * kHeadDim + c] *
                 k[j * kHidden + h * kHeadDim + c];
          s[j] = d * scale;
          mx = std::max(mx, s[j]);
        }
        for (uint32_t j = 0; j < count; ++j)
          den += (s[j] = std::exp(s[j] - mx));
        for (uint32_t j = 0; j < count; ++j)
          for (uint32_t c = 0; c < kHeadDim; ++c)
            want[i * kHidden + h * kHeadDim + c] +=
                s[j] / den * v[j * kHidden + h * kHeadDim + c];
      }
    // Binary16 operands and probabilities: about 1e-3 of the output range.
    Near(FromHalf(out, count * kHidden), want, 4e-3,
         "attention count " + std::to_string(count));
  }
}

void TestGemm() {
  struct Shape {
    uint32_t out, in;
    bool bias, accumulate;
  };
  // patch embedding, q/k/v, attention output, MLP up and down, merger.
  const Shape shapes[] = {{1152, 768, true, true},  {1152, 1152, true, false},
                          {1152, 1152, true, true}, {4304, 1152, true, false},
                          {1152, 4304, true, true}, {4608, 4608, true, false},
                          {3072, 4608, true, false}};
  std::string error;
  auto gemm = gufo::models::hy4_flash::VisionGemm::Create(nullptr, &error);
  Expect(gemm != nullptr, "VisionGemm::Create: " + error);
  if (!gemm)
    return;
  for (const Shape& s : shapes) {
    // Binary16 operands; values chosen so the products are exact in FP32.
    std::vector<uint16_t> w(std::size_t{s.out} * s.in);
    std::vector<double> wd(w.size());
    for (std::size_t i = 0; i < w.size(); ++i) {
      const _Float16 h = (_Float16)Random(1, -0.05f, 0.05f)[0];
      std::memcpy(&w[i], &h, 2);
      wd[i] = (double)h;
    }
    const auto bias = Random(s.out, -1, 1);
    uint16_t* w_dev = Device(w);
    const float* bias_dev = Device(bias);
    Expect(gemm->Tune(w_dev, s.out, s.in, bias_dev, s.accumulate, 1024, &error),
           "tune: " + error);
    for (const uint32_t rows : {48u, 440u, 1024u, 3000u}) {
      std::vector<uint16_t> x(std::size_t{rows} * s.in);
      std::vector<double> xd(x.size());
      for (std::size_t i = 0; i < x.size(); ++i) {
        const _Float16 h = (_Float16)Random(1, -1, 1)[0];
        std::memcpy(&x[i], &h, 2);
        xd[i] = (double)h;
      }
      const auto c = Random(std::size_t{rows} * s.out, -4, 4);
      float* y = Device(s.accumulate ? c : std::vector<float>(c.size()));
      Expect(gemm->Run(w_dev, Device(x), y, bias_dev,
                       s.accumulate ? y : nullptr, rows, s.out, s.in, &error),
             "gemm: " + error);
      // Every seventh row and the last: all tile positions, at a seventh of
      // the host reference's cost.
      const auto all = Host(y, c.size());
      std::vector<float> got;
      std::vector<double> want;
      std::vector<uint32_t> checked;
      for (uint32_t r = 0; r < rows; r += 7)
        checked.push_back(r);
      if (checked.back() != rows - 1)
        checked.push_back(rows - 1);
      for (const uint32_t r : checked) {
        for (uint32_t o = 0; o < s.out; ++o) {
          double sum =
              bias[o] + (s.accumulate ? c[std::size_t{r} * s.out + o] : 0.0);
          for (uint32_t k = 0; k < s.in; ++k)
            sum +=
                xd[std::size_t{r} * s.in + k] * wd[std::size_t{o} * s.in + k];
          want.push_back(sum);
          got.push_back(all[std::size_t{r} * s.out + o]);
        }
      }
      // FP32 accumulation: about 1e-6 of the range. Binary16 rounding of the
      // result or of c would show as 2^-11.
      Near(got, want, 2e-5,
           "gemm " + std::to_string(s.out) + "x" + std::to_string(s.in) +
               (s.accumulate ? " +c" : "") + " rows " + std::to_string(rows));
    }
  }
  // The fallback is exact too, so only report which path ran.
  std::printf("vision GEMM: %zu of %zu shapes on hipBLASLt\n",
              gemm->TunedShapes(), std::size(shapes));
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "SKIP: no HIP device\n");
    return 77;
  }
  TestRows();
  TestMerger();
  TestToHalf();
  TestAttention();
  TestGemm();
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 vision kernels: all checks passed\n");
  return 0;
}
