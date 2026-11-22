// Every entry of kernels/rocm/hy4_kernels.h against a double precision host
// reference of its documented contract, on small random inputs: ring
// wrap-around, windows, selections, sinks, ties and the M-RoPE sections.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "kernels/rocm/hy4_kernels.h"
#include "kernels/rocm/mmq/hy4_mmq.h"

namespace {

int g_failures = 0;
std::mt19937 g_rng(20261119);

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

// Max |got - want| relative to max(|want|, 1e-3) over the arrays; infinities
// must match exactly.
void Near(const std::vector<float>& got, const std::vector<double>& want,
          double tol, const std::string& what) {
  if (got.size() != want.size()) {
    Expect(false, what + ": size");
    return;
  }
  double scale = 1e-3, worst = 0.0;
  std::size_t at = 0;
  for (double w : want) {
    if (std::isfinite(w))
      scale = std::max(scale, std::fabs(w));
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isinf(want[i]) || std::isinf(got[i])) {
      if (got[i] != want[i]) {
        Expect(false, what + ": infinity at " + std::to_string(i));
        return;
      }
      continue;
    }
    const double d = std::fabs(got[i] - want[i]);
    if (!(d <= worst)) {
      worst = d;
      at = i;
    }
  }
  const double rel = worst / scale;
  if (!(rel <= tol)) {
    std::fprintf(stderr, "FAIL %s: error %.3g at %zu (got %.7g, want %.7g)\n",
                 what.c_str(), rel, at, got[at], want[at]);
    ++g_failures;
  }
}

double Sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// Binary16 bits of v (round to nearest even; up to the normal range's top,
// which the test values keep to) and back.
uint16_t HalfBits(float v) {
  uint32_t x;
  std::memcpy(&x, &v, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  const uint32_t mant = x & 0x7fffffu;
  if (exp <= 0)  // subnormal or zero: a multiple of 2^-24
    return (uint16_t)(sign |
                      (uint32_t)std::nearbyint(std::fabs(v) * 16777216.0f));
  uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rest = mant & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1)))
    ++half;
  return (uint16_t)(sign | half);
}
double HalfValue(uint16_t h) {
  const int exp = (h >> 10) & 0x1f;
  const int mant = h & 0x3ff;
  const double v =
      exp == 0 ? std::ldexp(mant, -24) : std::ldexp(1024 + mant, exp - 25);
  return h & 0x8000 ? -v : v;
}

#pragma pack(push, 1)
struct BlockQ8_0 {
  uint16_t d;
  int8_t qs[32];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34);

std::vector<BlockQ8_0> RandomQ8_0(std::size_t blocks) {
  std::uniform_int_distribution<int> q(-127, 127), s(1, 60);
  std::vector<BlockQ8_0> v(blocks);
  for (auto& b : v) {
    b.d = HalfBits((float)s(g_rng) / 1024.0f);
    for (auto& x : b.qs)
      x = (int8_t)q(g_rng);
  }
  return v;
}
double Q8Value(const std::vector<BlockQ8_0>& v, std::size_t i) {
  return HalfValue(v[i / 32].d) * v[i / 32].qs[i % 32];
}

double E4M3(uint8_t v) {
  const int exp = (v >> 3) & 0xF, man = v & 7;
  const double r =
      exp == 0 ? std::ldexp(man, -9) : std::ldexp(1.0 + man / 8.0, exp - 7);
  return v & 0x80 ? -r : r;
}

std::vector<double> RmsRow(const float* x, uint32_t dim, double eps) {
  double ss = 0;
  for (uint32_t i = 0; i < dim; ++i)
    ss += (double)x[i] * x[i];
  const double s = 1.0 / std::sqrt(ss / dim + eps);
  std::vector<double> out(dim);
  for (uint32_t i = 0; i < dim; ++i)
    out[i] = x[i] * s;
  return out;
}

void TestQ8_0() {
  const uint32_t rows = 9, dim = 96, n = 4;
  const auto table = RandomQ8_0(rows * dim / 32);
  const std::vector<int32_t> tokens{3, 0, 8, 3};
  auto* dt = Device(table);
  auto* dtok = Device(tokens);
  float* out = Device(std::vector<float>(n * dim));
  Expect(hy4_embed_q8_0(dt, dtok, n, dim, out, nullptr) == 0,
         "embed_q8_0 status");
  std::vector<double> want(n * dim);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < dim; ++i)
      want[t * dim + i] = Q8Value(table, (std::size_t)tokens[t] * dim + i);
  Near(Host(out, n * dim), want, 0, "embed_q8_0");

  float* all = Device(std::vector<float>(rows * dim));
  Expect(hy4_dequant_q8_0(dt, rows * dim / 32, all, nullptr) == 0,
         "dequant_q8_0 status");
  std::vector<double> full(rows * dim);
  for (std::size_t i = 0; i < full.size(); ++i)
    full[i] = Q8Value(table, i);
  Near(Host(all, rows * dim), full, 0, "dequant_q8_0");
}

void TestRmsNorm() {
  const uint32_t rows = 5, dim = 200, stride = 256;
  const auto x = Random(rows * stride, -3, 3);
  const auto w = Random(dim, 0.5f, 1.5f);
  auto* dx = Device(x);
  auto* dw = Device(w);
  for (bool weighted : {true, false}) {
    float* out = Device(x);  // untouched padding stays as x
    Expect(hy4_rms_norm(dx, weighted ? dw : nullptr, out, rows, dim, stride,
                        1e-6f, nullptr) == 0,
           "rms_norm status");
    std::vector<double> want(x.begin(), x.end());
    for (uint32_t r = 0; r < rows; ++r) {
      const auto n = RmsRow(&x[r * stride], dim, 1e-6);
      for (uint32_t i = 0; i < dim; ++i)
        want[r * stride + i] = n[i] * (weighted ? w[i] : 1.0);
    }
    Near(Host(out, rows * stride), want, 2e-6,
         weighted ? "rms_norm" : "rms_norm unweighted");
    // The binary16 copy: the FP32 outputs rounded once, bit for bit.
    float* out2 = Device(x);
    auto* half = Device(std::vector<uint16_t>(rows * stride));
    Expect(hy4_rms_norm_half(dx, weighted ? dw : nullptr, out2, half, rows, dim,
                             stride, 1e-6f, nullptr) == 0,
           "rms_norm_half status");
    const auto f = Host(out2, rows * stride);
    const auto hb = Host(half, rows * stride);
    Expect(f == Host(out, rows * stride), "rms_norm_half FP32 output");
    bool same = true;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t i = 0; i < dim; ++i)
        same &= hb[r * stride + i] == HalfBits(f[r * stride + i]);
    Expect(same, "rms_norm_half rounds the FP32 output");
  }
}

void TestHyperConnections() {
  const uint32_t n = 11, hc = 4, dim = 96, flat = hc * dim;
  const float eps = 1e-6f, hc_eps = 1e-6f, magnitude = 2.0f;
  const auto x = Random(n * dim);
  float* dh = Device(std::vector<float>(n * flat));
  Expect(hy4_hc_init(Device(x), dh, n, hc, dim, nullptr) == 0,
         "hc_init status");
  std::vector<double> init(n * flat);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < hc; ++i)
      for (uint32_t c = 0; c < dim; ++c)
        init[(t * hc + i) * dim + c] = x[t * dim + c];
  Near(Host(dh, n * flat), init, 0, "hc_init");

  const auto h = Random(n * flat, -2, 2);
  const auto scale = Random(2, 0.5f, 1.5f);
  const auto base = Random(2 * hc);
  for (bool layer : {true, false}) {
    const uint32_t n_mix = layer ? 2 * hc : hc;
    const auto fn = Random(n_mix * flat, -0.1f, 0.1f);
    float* out = Device(std::vector<float>(n * dim));
    float* post = layer ? Device(std::vector<float>(n * hc)) : nullptr;
    Expect(hy4_hc_pre(Device(h), Device(fn), Device(scale), Device(base), out,
                      post, n, hc, dim, eps, hc_eps, magnitude, nullptr) == 0,
           "hc_pre status");
    std::vector<double> want(n * dim), want_post(n * hc);
    for (uint32_t t = 0; t < n; ++t) {
      const float* ht = &h[t * flat];
      double ss = 0;
      for (uint32_t i = 0; i < flat; ++i)
        ss += (double)ht[i] * ht[i];
      const double rs = 1.0 / std::sqrt(ss / flat + eps);
      std::vector<double> mix(n_mix, 0.0);
      for (uint32_t m = 0; m < n_mix; ++m)
        for (uint32_t i = 0; i < flat; ++i)
          mix[m] += (double)fn[m * flat + i] * ht[i];
      for (uint32_t i = 0; i < hc; ++i) {
        const double pre = Sigmoid(mix[i] * rs * scale[0] + base[i]) + hc_eps;
        for (uint32_t c = 0; c < dim; ++c)
          want[t * dim + c] += pre * ht[i * dim + c];
        if (layer)
          want_post[t * hc + i] =
              Sigmoid(mix[hc + i] * rs * scale[1] + base[hc + i]) * magnitude +
              hc_eps;
      }
    }
    Near(Host(out, n * dim), want, 1e-5, layer ? "hc_pre" : "hc_pre head");
    if (layer)
      Near(Host(post, n * hc), want_post, 1e-5, "hc_pre post");
    // Passes under 8 tokens mix one token per block: bit-identical results.
    const uint32_t few = 3;
    float* out_few = Device(std::vector<float>(few * dim));
    float* post_few = layer ? Device(std::vector<float>(few * hc)) : nullptr;
    Expect(hy4_hc_pre(Device(h), Device(fn), Device(scale), Device(base),
                      out_few, post_few, few, hc, dim, eps, hc_eps, magnitude,
                      nullptr) == 0,
           "hc_pre few status");
    const auto a = Host(out, few * dim), b = Host(out_few, few * dim);
    Expect(a == b, "hc_pre identical for 3 and 11 tokens");
    if (layer)
      Expect(Host(post, few * hc) == Host(post_few, few * hc),
             "hc_pre post identical for 3 and 11 tokens");
  }

  const auto y = Random(n * dim);
  const auto residual = Random(n * flat);
  const auto post = Random(n * hc, 0, 2);
  float* dres = Device(residual);
  Expect(hy4_hc_post(Device(y), dres, Device(post), dres, n, hc, dim,
                     nullptr) == 0,
         "hc_post status");
  std::vector<double> want(n * flat);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < hc; ++i)
      for (uint32_t c = 0; c < dim; ++c)
        want[(t * hc + i) * dim + c] =
            residual[(t * hc + i) * dim + c] +
            (double)y[t * dim + c] * post[t * hc + i];
  Near(Host(dres, n * flat), want, 1e-6, "hc_post in place");

  // hc_post_pre: bit-identical to hc_post then hc_pre, with post_prev and
  // post the same buffer as the forward pass uses them, on the batched (11
  // tokens) and the per-token (3) paths.
  for (uint32_t m : {n, 3u})
    for (bool layer : {true, false}) {
      const uint32_t n_mix = layer ? 2 * hc : hc;
      const auto fn = Random(n_mix * flat, -0.1f, 0.1f);
      float* dy = Device(y);
      float* h_a = Device(residual);
      float* h_b = Device(residual);
      float* post_a = Device(post);
      float* post_b = Device(post);
      float* out_a = Device(std::vector<float>(m * dim));
      float* out_b = Device(std::vector<float>(m * dim));
      Expect(hy4_hc_post(dy, h_a, post_a, h_a, m, hc, dim, nullptr) == 0 &&
                 hy4_hc_pre(h_a, Device(fn), Device(scale), Device(base), out_a,
                            layer ? post_a : nullptr, m, hc, dim, eps, hc_eps,
                            magnitude, nullptr) == 0,
             "hc_post, hc_pre status");
      Expect(hy4_hc_post_pre(dy, post_b, h_b, Device(fn), Device(scale),
                             Device(base), out_b, layer ? post_b : nullptr, m,
                             hc, dim, eps, hc_eps, magnitude, nullptr) == 0,
             "hc_post_pre status");
      Expect(Host(h_a, m * flat) == Host(h_b, m * flat),
             "hc_post_pre streams identical");
      Expect(Host(out_a, m * dim) == Host(out_b, m * dim),
             "hc_post_pre output identical");
      Expect(Host(post_a, m * hc) == Host(post_b, m * hc),
             "hc_post_pre post identical");
    }
}

// hy4_hc_decode: bit-identical to the kernels it replaces (rms_norm of y,
// hc_post_pre or hc_pre, rms_norm of the mix) at the model's width, with and
// without the post, norm weights and post output, for 1, 3 and 11 tokens.
void TestHcDecodeShape(uint32_t hc, uint32_t dim) {
  const uint32_t flat = hc * dim;
  const float eps = 1e-6f, hc_eps = 1e-6f, magnitude = 2.0f;
  const auto scale = Random(2, 0.5f, 1.5f);
  const auto base = Random(2 * hc);
  const auto wy = Random(dim, 0.5f, 1.5f), wx = Random(dim, 0.5f, 1.5f);
  for (uint32_t n : {1u, 3u, 11u})
    for (bool with_y : {true, false})
      for (bool layer : {true, false})
        for (bool weights : {true, false}) {
          const std::string what =
              "hc_decode hc=" + std::to_string(hc) +
              " dim=" + std::to_string(dim) + " n=" + std::to_string(n) +
              (with_y ? " post" : "") + (layer ? " layer" : " head") +
              (weights ? " weights" : "");
          const uint32_t n_mix = layer ? 2 * hc : hc;
          const auto fn = Random(n_mix * flat, -0.1f, 0.1f);
          const auto h = Random(n * flat, -2, 2), y = Random(n * dim, -3, 3);
          const auto post0 = Random(n * hc, 0, 2);
          const float* dwy = weights ? Device(wy) : nullptr;
          const float* dwx = weights ? Device(wx) : nullptr;
          // The kernels it replaces.
          float *y_a = Device(y), *h_a = Device(h), *p_a = Device(post0);
          float *x_a = Device(std::vector<float>(n * dim)),
                *o_a = Device(std::vector<float>(n * dim));
          bool ok = true;
          if (with_y) {
            ok &= hy4_rms_norm(y_a, dwy, y_a, n, dim, dim, eps, nullptr) == 0;
            ok &=
                hy4_hc_post_pre(y_a, p_a, h_a, Device(fn), Device(scale),
                                Device(base), x_a, layer ? p_a : nullptr, n, hc,
                                dim, eps, hc_eps, magnitude, nullptr) == 0;
          } else {
            ok &= hy4_hc_pre(h_a, Device(fn), Device(scale), Device(base), x_a,
                             layer ? p_a : nullptr, n, hc, dim, eps, hc_eps,
                             magnitude, nullptr) == 0;
          }
          ok &= hy4_rms_norm(x_a, dwx, o_a, n, dim, dim, eps, nullptr) == 0;
          // The fused boundary.
          float *y_b = Device(y), *h_b = Device(h), *p_b = Device(post0);
          float *x_b = Device(std::vector<float>(n * dim)),
                *o_b = Device(std::vector<float>(n * dim));
          ok &= hy4_hc_decode(with_y ? y_b : nullptr, dwy, p_b, h_b, Device(fn),
                              Device(scale), Device(base), x_b, o_b, dwx,
                              layer ? p_b : nullptr, n, hc, dim, eps, hc_eps,
                              magnitude, nullptr) == 0;
          Expect(ok, what + " status");
          Expect(Host(h_a, n * flat) == Host(h_b, n * flat),
                 what + " streams identical");
          Expect(Host(x_a, n * dim) == Host(x_b, n * dim),
                 what + " mix identical");
          Expect(Host(o_a, n * dim) == Host(o_b, n * dim),
                 what + " normed mix identical");
          Expect(Host(p_a, n * hc) == Host(p_b, n * hc),
                 what + " post identical");
          Expect(Host(y_b, n * dim) == y, what + " leaves y");
        }
}

// The model's shape (the fast kernels), and two the general ones take.
void TestHcDecode() {
  TestHcDecodeShape(4, 3072);
  TestHcDecodeShape(3, 96);
  TestHcDecodeShape(4, 3500);
}

void TestGemv() {
  const uint32_t m = 37, k = 256;
  for (uint32_t n : {1u, 3u, 11u}) {
    const auto x = Random(n * k);
    const auto w = Random(m * k);
    std::vector<uint16_t> wh(m * k);
    for (std::size_t i = 0; i < wh.size(); ++i)
      wh[i] = HalfBits(w[i]);
    for (int type : {HY4_TYPE_F32, HY4_TYPE_F16}) {
      float* y = Device(std::vector<float>(n * m));
      const void* dw = type == HY4_TYPE_F32 ? (const void*)Device(w)
                                            : (const void*)Device(wh);
      Expect(hy4_gemv_float(type, dw, Device(x), y, m, k, n, nullptr) == 0,
             "gemv_float status");
      std::vector<double> want(n * m, 0.0);
      for (uint32_t t = 0; t < n; ++t)
        for (uint32_t r = 0; r < m; ++r)
          for (uint32_t i = 0; i < k; ++i)
            want[t * m + r] +=
                (type == HY4_TYPE_F32 ? (double)w[r * k + i]
                                      : HalfValue(wh[r * k + i])) *
                x[t * k + i];
      Near(Host(y, n * m), want, 1e-5,
           std::string("gemv_float ") + (type == HY4_TYPE_F32 ? "f32" : "f16") +
               " n=" + std::to_string(n));
      // A multi-token call computes each token as a one-token call would.
      const auto many = Host(y, n * m);
      float* dx = Device(x);
      bool same = true;
      for (uint32_t t = 0; t < n; ++t) {
        float* y1 = Device(std::vector<float>(m));
        Expect(hy4_gemv_float(type, dw, dx + t * k, y1, m, k, 1, nullptr) == 0,
               "gemv_float one-token status");
        const auto one = Host(y1, m);
        same &= std::memcmp(one.data(), many.data() + t * m, m * 4) == 0;
      }
      Expect(same,
             "gemv_float tokens equal one-token calls n=" + std::to_string(n));
    }
  }
  Expect(hy4_gemv_float(HY4_TYPE_Q8_0, nullptr, nullptr, nullptr, 1, 8, 1,
                        nullptr) != 0,
         "gemv_float rejects other types");

  // Grouped Q8_0 with padded token and group strides: k = 96 takes the
  // generic kernel, 192 and 512 (wk_b, wv_b) the whole-block rows; m - 2
  // rows (a partial row group) the generic one again.
  for (auto [mk, kk] :
       {std::pair<uint32_t, uint32_t>{64, 96}, {64, 192}, {32, 512}}) {
    const uint32_t groups = 3, n = 10;
    const std::string shape =
        " m=" + std::to_string(mk) + " k=" + std::to_string(kk);
    const uint64_t x_grp = kk + 32, x_tok = groups * x_grp + 16, y_grp = mk + 8,
                   y_tok = groups * y_grp + 4;
    const auto w = RandomQ8_0(groups * mk * kk / 32);
    const auto x = Random(n * x_tok);
    float* y = Device(std::vector<float>(n * y_tok, 7.0f));
    Expect(hy4_gemv_q8_0_grouped(Device(w), Device(x), y, mk, kk, groups, n,
                                 x_tok, x_grp, y_tok, y_grp, nullptr) == 0,
           "gemv_q8_0_grouped status");
    std::vector<double> want(n * y_tok, 7.0);
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t g = 0; g < groups; ++g)
        for (uint32_t r = 0; r < mk; ++r) {
          double acc = 0;
          for (uint32_t i = 0; i < kk; ++i)
            acc += Q8Value(w, ((std::size_t)g * mk + r) * kk + i) *
                   x[t * x_tok + g * x_grp + i];
          want[t * y_tok + g * y_grp + r] = acc;
        }
    Near(Host(y, n * y_tok), want, 1e-5, "gemv_q8_0_grouped" + shape);
    {  // Each token as a one-token call computes it, also with a partial row
       // group (m = 62).
      for (const uint32_t m2 : {mk, mk - 2}) {
        auto* dw = Device(w);
        auto* dx = Device(x);
        float* many = Device(std::vector<float>(n * y_tok, 7.0f));
        Expect(hy4_gemv_q8_0_grouped(dw, dx, many, m2, kk, groups, n, x_tok,
                                     x_grp, y_tok, y_grp, nullptr) == 0,
               "gemv_q8_0_grouped many status");
        const auto hm = Host(many, n * y_tok);
        bool same = true;
        for (uint32_t t = 0; t < n; ++t) {
          float* one = Device(std::vector<float>(y_tok, 7.0f));
          Expect(
              hy4_gemv_q8_0_grouped(dw, dx + t * x_tok, one, m2, kk, groups, 1,
                                    x_tok, x_grp, y_tok, y_grp, nullptr) == 0,
              "gemv_q8_0_grouped one status");
          const auto ho = Host(one, y_tok);
          same &= std::memcmp(ho.data(), hm.data() + t * y_tok, y_tok * 4) == 0;
        }
        Expect(same, "gemv_q8_0_grouped tokens equal one-token calls" + shape +
                         " m2=" + std::to_string(m2));
      }
    }
  }

  // The prefill absorption at Hy4's shape (wk_b 512 x 192, rope 64), with a
  // partial token tile: [W . q_nope, q_rope] as binary16, q rounded to
  // binary16 first.
  {
    const uint32_t m = 512, nope = 192, n_rot = 64, hk = nope + n_rot,
                   heads = 2, nt = 77, row = m + n_rot;
    const auto aw = RandomQ8_0(heads * m * nope / 32);
    const auto aq = Random(nt * heads * hk);
    uint16_t* wq = Device(std::vector<uint16_t>(aw.size() * 32));
    float* wd = Device(std::vector<float>(aw.size()));
    Expect(hy4_q8_0_split(Device(aw), aw.size(), wq, wd, nullptr) == 0,
           "q8_0_split status");
    std::vector<double> split(aw.size() * 32);
    {
      const auto bits = Host(wq, aw.size() * 32);
      const auto hd = Host(wd, aw.size());
      for (std::size_t i = 0; i < split.size(); ++i) {
        split[i] = HalfValue(bits[i]);
        Expect(split[i] == aw[i / 32].qs[i % 32], "q8_0_split values");
      }
      for (std::size_t b = 0; b < aw.size(); ++b)
        Expect(hd[b] == HalfValue(aw[b].d), "q8_0_split scales");
    }
    uint16_t* out = Device(std::vector<uint16_t>(nt * heads * row));
    Expect(hy4_mla_absorb(wq, wd, Device(aq), out, nt, heads, nope, n_rot, m,
                          nullptr) == 0,
           "mla_absorb status");
    const auto got_bits = Host(out, nt * heads * row);
    std::vector<float> got(nt * heads * row);
    std::vector<double> want(nt * heads * row);
    for (uint32_t t = 0; t < nt; ++t)
      for (uint32_t h = 0; h < heads; ++h) {
        const float* qr = &aq[(t * heads + h) * hk];
        const std::size_t o = ((std::size_t)t * heads + h) * row;
        for (uint32_t r = 0; r < m; ++r) {
          double acc = 0;
          for (uint32_t i = 0; i < nope; ++i)
            acc += Q8Value(aw, ((std::size_t)h * m + r) * nope + i) *
                   HalfValue(HalfBits(qr[i]));
          want[o + r] = acc;
        }
        for (uint32_t i = 0; i < n_rot; ++i)
          want[o + m + i] = HalfValue(HalfBits(qr[nope + i]));
        for (uint32_t i = 0; i < row; ++i)
          got[o + i] = (float)HalfValue(got_bits[o + i]);
      }
    Near(got, want, 1e-3, "mla_absorb");
    Expect(hy4_mla_absorb(nullptr, nullptr, nullptr, nullptr, 1, 1, nope, n_rot,
                          256, nullptr) != 0,
           "mla_absorb rejects kv_lora 256");
  }
}

void TestRope() {
  const uint32_t n = 3, heads = 2, n_rot = 64, offset = 8, head_stride = 80,
                 tok_stride = heads * head_stride + 4;
  const float base = 10000.0f;
  const std::vector<int32_t> pos{0, 17, 4093};
  const std::vector<int32_t> hw{0, 0, 21, 5, 4090, 4100};
  const auto x = Random(n * tok_stride);
  auto reference = [&](bool three_d) {
    std::vector<double> want(x.begin(), x.end());
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t h = 0; h < heads; ++h)
        for (uint32_t i = 0; i < n_rot / 2; ++i) {
          int32_t p = pos[t];
          if (three_d && i < 30 && i % 3 != 0)
            p = hw[2 * t + (i % 3 - 1)];
          const double theta = p * std::pow((double)base, -2.0 * i / n_rot);
          const std::size_t at =
              t * tok_stride + h * head_stride + offset + 2 * i;
          want[at] = x[at] * std::cos(theta) - x[at + 1] * std::sin(theta);
          want[at + 1] = x[at] * std::sin(theta) + x[at + 1] * std::cos(theta);
        }
    return want;
  };
  float* a = Device(x);
  Expect(hy4_rope(a, Device(pos), n, heads, tok_stride, head_stride, offset,
                  n_rot, base, nullptr) == 0,
         "rope status");
  const auto rope = Host(a, x.size());
  Near(rope, reference(false), 2e-4, "rope");
  float* b = Device(x);
  Expect(hy4_rope_3d(b, Device(pos), Device(hw), n, heads, tok_stride,
                     head_stride, offset, n_rot, base, nullptr) == 0,
         "rope_3d status");
  Near(Host(b, x.size()), reference(true), 2e-4, "rope_3d");
  // Text positions: a null pos_hw, or pos_hw equal to pos, is hy4_rope.
  float* c = Device(x);
  (void)hy4_rope_3d(c, Device(pos), nullptr, n, heads, tok_stride, head_stride,
                    offset, n_rot, base, nullptr);
  Expect(Host(c, x.size()) == rope, "rope_3d without pos_hw equals rope");
  std::vector<int32_t> same;
  for (int32_t p : pos)
    same.insert(same.end(), {p, p});
  float* d = Device(x);
  (void)hy4_rope_3d(d, Device(pos), Device(same), n, heads, tok_stride,
                    head_stride, offset, n_rot, base, nullptr);
  Expect(Host(d, x.size()) == rope, "rope_3d at text positions equals rope");
  Expect(hy4_rope_3d(d, Device(pos), Device(same), n, heads, tok_stride,
                     head_stride, 0, 32, base, nullptr) != 0,
         "rope_3d rejects n_rot != 64");
}

// softmax(scale q.k, sink) . latent over the given key positions.
std::vector<double> MlaReference(const std::vector<float>& q,
                                 const std::vector<float>& cache, uint32_t ring,
                                 const std::vector<std::vector<int32_t>>& keys,
                                 const float* sinks, float scale,
                                 uint32_t heads, uint32_t dk, uint32_t dv) {
  const uint32_t n = (uint32_t)keys.size();
  std::vector<double> out(n * heads * dv, 0.0);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t h = 0; h < heads; ++h) {
      std::vector<double> s;
      double mx = sinks ? sinks[h] : -INFINITY;
      for (int32_t p : keys[t]) {
        double d = 0;
        for (uint32_t i = 0; i < dk; ++i)
          d += (double)q[(t * heads + h) * dk + i] * cache[(p % ring) * dk + i];
        s.push_back(d * scale);
        mx = std::max(mx, s.back());
      }
      double den = sinks ? std::exp(sinks[h] - mx) : 0.0;
      for (double& e : s)
        den += (e = std::exp(e - mx));
      for (std::size_t j = 0; j < s.size(); ++j)
        for (uint32_t c = 0; c < dv; ++c)
          out[(t * heads + h) * dv + c] +=
              s[j] / den * cache[(keys[t][j] % ring) * dk + c];
    }
  return out;
}

// Decode attention at Hy4's shape against the reference: spans of one key,
// of a split boundary and of more than 2,048 keys (larger splits), windows
// wrapping the ring, and selections with empty, future and repeated slots.
// A token's result is bitwise the same alone or in a pass, and selecting
// every visible key gives the causal result bitwise.
// The decode MLA's fused prep and query rope: bit for bit what the norms,
// ropes and copy they replace compute, with 1-D and 3-D positions.
void TestMlaPrep() {
  const uint32_t n = 3, q_lora = 1024, kv_lora = 512, n_rot = 64,
                 dk = kv_lora + n_rot, heads = 8, head_k = 192,
                 nope = head_k - n_rot;
  const float base = 10000.0f, eps = 1e-6f;
  const std::vector<int32_t> pos = {5, 900, 70000};
  const std::vector<int32_t> hw = {1, 2, 30, 40, 7, 9};
  const auto qn = Random(q_lora, 0.5f, 1.5f), kn = Random(kv_lora, 0.5f, 1.5f);
  for (bool three_d : {false, true}) {
    const int32_t* dhw = three_d ? Device(hw) : nullptr;
    const auto qr = Random(n * q_lora, -3, 3), kva = Random(n * dk, -3, 3);
    float *qa = Device(qr), *ka = Device(kva), *qb = Device(qr),
          *kb = Device(kva);
    bool ok =
        hy4_rms_norm(qa, Device(qn), qa, n, q_lora, q_lora, eps, nullptr) ==
            0 &&
        hy4_rms_norm(ka, Device(kn), ka, n, kv_lora, dk, eps, nullptr) == 0 &&
        hy4_rope_3d(ka, Device(pos), dhw, n, 1, dk, 0, kv_lora, n_rot, base,
                    nullptr) == 0;
    ok &= hy4_mla_prep(qb, Device(qn), q_lora, kb, Device(kn), kv_lora, dk,
                       Device(pos), dhw, n_rot, base, eps, n, nullptr) == 0;
    Expect(ok, "mla_prep status");
    Expect(Host(qa, n * q_lora) == Host(qb, n * q_lora),
           std::string("mla_prep query latent") + (three_d ? " 3-D" : ""));
    Expect(Host(ka, n * dk) == Host(kb, n * dk),
           std::string("mla_prep key latent") + (three_d ? " 3-D" : ""));

    const auto q = Random(n * heads * head_k, -3, 3);
    const auto full = Random(n * heads * dk);
    float *q1 = Device(q), *q2 = Device(q), *f1 = Device(full),
          *f2 = Device(full);
    ok = hy4_rope_3d(q1, Device(pos), dhw, n, heads, heads * head_k, head_k,
                     nope, n_rot, base, nullptr) == 0 &&
         hy4_copy_2d(f1 + kv_lora, dk, q1 + nope, head_k, n_rot, n * heads,
                     nullptr) == 0;
    ok &= hy4_rope_3d_copy(q2, Device(pos), dhw, n, heads, heads * head_k,
                           head_k, nope, n_rot, base, f2 + kv_lora, heads * dk,
                           dk, nullptr) == 0;
    Expect(ok, "rope_3d_copy status");
    Expect(Host(q1, n * heads * head_k) == Host(q2, n * heads * head_k),
           std::string("rope_3d_copy rows") + (three_d ? " 3-D" : ""));
    Expect(Host(f1, n * heads * dk) == Host(f2, n * heads * dk),
           std::string("rope_3d_copy copies") + (three_d ? " 3-D" : ""));
  }
}

void TestMlaAttention() {
  const uint32_t dk = 576, dv = 512, ring = 2600;
  const float scale = 1.0f / 16.0f;
  // Binary16 cache rows; the reference reads their exact values.
  auto cache = Random(ring * dk, -1, 1);
  std::vector<uint16_t> cbits(cache.size());
  for (std::size_t i = 0; i < cache.size(); ++i) {
    cbits[i] = HalfBits(cache[i]);
    cache[i] = (float)HalfValue(cbits[i]);
  }
  uint16_t* dc = Device(cbits);
  auto run = [&](const std::vector<float>& q, const std::vector<int32_t>& pos,
                 uint32_t ring_used, uint32_t window,
                 const std::vector<int32_t>& sel, uint32_t n_sel,
                 const std::vector<float>& sinks, uint32_t heads) {
    const uint32_t n = (uint32_t)pos.size();
    const uint64_t work = hy4_mla_attention_work(n, heads, n_sel);
    float* dw = Device(std::vector<float>(work));
    float* out = Device(std::vector<float>(n * heads * dv));
    Expect(hy4_mla_attention(Device(q), dc, ring_used, Device(pos), window,
                             sel.empty() ? nullptr : Device(sel), n_sel,
                             sinks.empty() ? nullptr : Device(sinks), scale,
                             out, n, heads, dk, dv, dw, work, nullptr) == 0,
           "mla_attention status");
    return Host(out, n * heads * dv);
  };
  auto alone = [&](const std::vector<float>& q, const std::vector<int32_t>& pos,
                   uint32_t ring_used, uint32_t window,
                   const std::vector<int32_t>& sel, uint32_t n_sel,
                   const std::vector<float>& sinks, uint32_t heads) {
    std::vector<float> all;
    for (std::size_t t = 0; t < pos.size(); ++t) {
      const std::vector<float> qt(q.begin() + t * heads * dk,
                                  q.begin() + (t + 1) * heads * dk);
      const std::vector<int32_t> st =
          sel.empty() ? sel
                      : std::vector<int32_t>(sel.begin() + t * n_sel,
                                             sel.begin() + (t + 1) * n_sel);
      const auto o =
          run(qt, {pos[t]}, ring_used, window, st, n_sel, sinks, heads);
      all.insert(all.end(), o.begin(), o.end());
    }
    return all;
  };
  for (uint32_t window : {0u, 512u}) {
    const uint32_t heads = window ? 64 : 32;
    const uint32_t ring_used = window ? 700 : ring;
    const std::vector<int32_t> pos =
        window ? std::vector<int32_t>{0, 40, 1500, 2899}
               : std::vector<int32_t>{0, 31, 32, 2500};
    const uint32_t n = (uint32_t)pos.size();
    const auto q = Random(n * heads * dk, -1, 1);
    const auto sinks = Random(heads, -1, 1);
    for (bool sink : {true, false}) {
      const std::vector<float> sk = sink ? sinks : std::vector<float>{};
      std::vector<std::vector<int32_t>> keys(n);
      for (uint32_t t = 0; t < n; ++t)
        for (int32_t p = window ? std::max(0, pos[t] - (int32_t)window + 1) : 0;
             p <= pos[t]; ++p)
          keys[t].push_back(p);
      const auto got = run(q, pos, ring_used, window, {}, 0, sk, heads);
      const std::string what = "mla_attention window " +
                               std::to_string(window) + (sink ? " sinks" : "");
      Near(got,
           MlaReference(q, cache, ring_used, keys,
                        sink ? sinks.data() : nullptr, scale, heads, dk, dv),
           1e-5, what);
      Expect(got == alone(q, pos, ring_used, window, {}, 0, sk, heads),
             what + " equals one-token calls");
    }
  }
  // Selections: shuffled positions with -1, future and repeated slots; the
  // last token selects every key at or below its position.
  const uint32_t heads = 32, n_sel = 2048;
  const std::vector<int32_t> pos{100, 2000, 2400};
  const uint32_t n = (uint32_t)pos.size();
  std::vector<int32_t> sel(n * n_sel, -1);
  std::vector<std::vector<int32_t>> keys(n);
  for (uint32_t t = 0; t < n; ++t) {
    std::vector<int32_t> pick;
    if (t + 1 < n) {
      for (int32_t p = 0; p <= pos[t] + 20; p += 3)
        pick.push_back(p);
      pick.push_back(pos[t]);
      pick.push_back(6);  // repeated
    } else {
      for (int32_t p = pos[t] - (int32_t)n_sel + 1; p <= pos[t]; ++p)
        pick.push_back(p);
    }
    std::shuffle(pick.begin(), pick.end(), g_rng);
    pick.resize(std::min<std::size_t>(pick.size(), n_sel - 10));
    for (std::size_t i = 0; i < pick.size(); ++i)
      sel[t * n_sel + i] = pick[i];
    std::set<int32_t> uniq;
    for (int32_t p : pick)
      if (p <= pos[t])
        uniq.insert(p);
    keys[t].assign(uniq.begin(), uniq.end());
  }
  const auto q = Random(n * heads * dk, -1, 1);
  const auto sinks = Random(heads, -1, 1);
  const auto got = run(q, pos, ring, 0, sel, n_sel, sinks, heads);
  Near(got,
       MlaReference(q, cache, ring, keys, sinks.data(), scale, heads, dk, dv),
       1e-5, "mla_attention selection");
  Expect(got == alone(q, pos, ring, 0, sel, n_sel, sinks, heads),
         "mla_attention selection equals one-token calls");
  // Selecting every visible key equals no selection (positions 0 .. pos).
  const std::vector<int32_t> p1{1000};
  std::vector<int32_t> every(n_sel, -1);
  for (int32_t p = 0; p <= p1[0]; ++p)
    every[(p * 7) % 1003] = p;  // shuffled slots
  const std::vector<float> q1(q.begin(), q.begin() + heads * dk);
  Expect(run(q1, p1, ring, 0, every, n_sel, sinks, heads) ==
             run(q1, p1, ring, 0, {}, 0, sinks, heads),
         "mla_attention selecting every key equals the causal span");
  Expect(hy4_mla_attention(nullptr, dc, ring, nullptr, 0, nullptr, 0, nullptr,
                           scale, nullptr, 1, heads, 96, 64, nullptr, 0,
                           nullptr) != 0,
         "mla_attention rejects other shapes");
}

// The matrix-core prefill attention against the same reference, at Hy4's
// shape: full causal spans, a gathered window starting past position 0, and
// indexer selections with empty, future and out-of-window slots.
void TestMlaPrefill() {
  const uint32_t heads = 32, dk = 576, dv = 512, ring = 400;
  const float scale = 1.0f / 24.0f;
  const auto cache = Random(ring * dk, -1, 1);
  const auto sinks = Random(heads, -1, 1);
  const std::vector<int32_t> pos{0, 31, 32, 150, 299};
  const uint32_t n = (uint32_t)pos.size();
  auto q = Random(n * heads * dk, -1, 1);
  std::vector<uint16_t> qbits(q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    qbits[i] = HalfBits(q[i]);
    q[i] = (float)HalfValue(qbits[i]);
  }
  auto* dq = Device(qbits);
  // The binary16 cache rows, as hy4_store_rows_half writes them.
  std::vector<uint16_t> cbits(cache.size());
  for (std::size_t i = 0; i < cache.size(); ++i)
    cbits[i] = HalfBits(cache[i]);
  auto* dc = Device(cbits);
  auto* dp = Device(pos);
  auto* ds = Device(sinks);
  const auto run = [&](uint32_t ring_used, int32_t lo, uint32_t window,
                       const std::vector<int32_t>* sel, uint32_t n_sel,
                       bool sink,
                       const std::vector<std::vector<int32_t>>& want_keys,
                       const std::string& what) -> std::vector<uint16_t> {
    auto* out = Device(std::vector<uint16_t>(n * heads * dv));
    Expect(hy4_mla_prefill(dq, dc, ring_used, lo, dp, window,
                           sel ? Device(*sel) : nullptr, n_sel,
                           sink ? ds : nullptr, scale, out, n, heads, dk, dv,
                           nullptr) == 0,
           what + " status");
    const auto bits = Host(out, n * heads * dv);
    std::vector<float> got(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
      got[i] = (float)HalfValue(bits[i]);
    Near(got,
         MlaReference(q, cache, ring_used, want_keys,
                      sink ? sinks.data() : nullptr, scale, heads, dk, dv),
         4e-3, what);
    return bits;
  };
  std::vector<uint16_t> causal_sinks;
  for (bool sink : {true, false}) {
    std::vector<std::vector<int32_t>> keys(n);
    for (uint32_t t = 0; t < n; ++t)
      for (int32_t k = 0; k <= pos[t]; ++k)
        keys[t].push_back(k);
    const auto bits =
        run(ring, 0, 0, nullptr, 0, sink, keys,
            std::string("mla_prefill causal") + (sink ? " sinks" : ""));
    if (sink)
      causal_sinks = bits;
  }
  {
    // Window 64 from position 20 (positions below 20 are never read: token 0
    // sees none and keeps only the sink), in a 256-row ring that position
    // 299's window wraps.
    const int32_t lo = 20;
    const uint32_t window = 64;
    std::vector<std::vector<int32_t>> keys(n);
    for (uint32_t t = 0; t < n; ++t)
      for (int32_t k = std::max(lo, pos[t] - (int32_t)window + 1); k <= pos[t];
           ++k)
        keys[t].push_back(k);
    run(256, lo, window, nullptr, 0, true, keys, "mla_prefill window");
  }
  {
    const uint32_t n_sel = 40;
    std::vector<int32_t> sel(n * n_sel, -1);
    std::vector<std::vector<int32_t>> keys(n);
    std::uniform_int_distribution<int32_t> any(0, ring - 1);
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t i = 0; i < n_sel; ++i) {
        if (i % 5 == 4)
          continue;  // empty slot
        const int32_t k = any(g_rng);
        sel[t * n_sel + i] = k;
        if (k <= pos[t] && pos[t] >= (int32_t)n_sel)
          keys[t].push_back(k);  // a future key is skipped
      }
    // A token below position n_sel selects every earlier position, so it
    // reads them in causal order whatever its selection row holds.
    for (uint32_t t = 0; t < n; ++t)
      for (int32_t k = 0; pos[t] < (int32_t)n_sel && k <= pos[t]; ++k)
        keys[t].push_back(k);
    const auto bits =
        run(ring, 0, 0, &sel, n_sel, true, keys, "mla_prefill selection");
    int differ = 0;
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t i = 0; pos[t] < (int32_t)n_sel && i < heads * dv; ++i)
        differ += bits[t * heads * dv + i] != causal_sinks[t * heads * dv + i];
    Expect(differ == 0, "mla_prefill selection below n_sel matches causal (" +
                            std::to_string(differ) + ")");
  }
  Expect(hy4_mla_prefill(dq, dc, ring, 0, dp, 0, nullptr, 0, ds, scale, nullptr,
                         n, 24, dk, dv, nullptr) != 0,
         "mla_prefill rejects 24 heads");
}

void TestRows() {
  const uint32_t ring = 6, dim = 40, n = 4;
  const uint64_t src_stride = 48;
  const auto src = Random(n * src_stride);
  const auto start = Random(ring * dim);
  const std::vector<int32_t> pos{4, 5, 6, 7};  // 6 and 7 wrap to rows 0 and 1
  float* ring_rows = Device(start);
  Expect(hy4_store_rows(Device(src), src_stride, ring_rows, ring, Device(pos),
                        n, dim, nullptr) == 0,
         "store_rows status");
  std::vector<double> want(start.begin(), start.end());
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t c = 0; c < dim; ++c)
      want[(pos[t] % ring) * dim + c] = src[t * src_stride + c];
  Near(Host(ring_rows, ring * dim), want, 0, "store_rows");

  // The same rows into a binary16 ring: each value rounded once.
  auto* half_rows = Device(std::vector<uint16_t>(ring * dim));
  Expect(hy4_store_rows_half(Device(src), src_stride, half_rows, ring,
                             Device(pos), n, dim, nullptr) == 0,
         "store_rows_half status");
  const auto half_stored = Host(half_rows, ring * dim);
  bool rounded = true;
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t c = 0; c < dim; ++c)
      rounded = rounded && half_stored[(pos[t] % ring) * dim + c] ==
                               HalfBits(src[t * src_stride + c]);
  Expect(rounded, "store_rows_half");

  const std::vector<int32_t> idx{7, -1, 2, 9, 5};
  const int32_t limit = 7;
  const uint32_t mask_rows = 2, n_keys = 8;
  float* gathered = Device(std::vector<float>(idx.size() * dim));
  float* mask = Device(std::vector<float>(mask_rows * n_keys, 3.0f));
  Expect(hy4_gather_indexed(half_rows, ring, Device(idx), (uint32_t)idx.size(),
                            limit, dim, gathered, mask, mask_rows, n_keys,
                            nullptr) == 0,
         "gather_indexed status");
  std::vector<double> g(idx.size() * dim, 0.0), m(mask_rows * n_keys, 0.0);
  for (uint32_t r = 0; r < idx.size(); ++r) {
    const bool ok = idx[r] >= 0 && idx[r] <= limit;
    for (uint32_t c = 0; c < dim; ++c)
      g[r * dim + c] =
          ok ? HalfValue(half_stored[(idx[r] % ring) * dim + c]) : 0.0;
    for (uint32_t t = 0; t < mask_rows; ++t)
      m[t * n_keys + r] = ok ? 0.0 : -INFINITY;
  }
  Near(Host(gathered, idx.size() * dim), g, 0, "gather_indexed rows");
  Near(Host(mask, mask_rows * n_keys), m, 0, "gather_indexed mask");
}

void TestElementwise() {
  const uint32_t rows = 6, dim = 100;
  const uint64_t count = rows * dim;
  const auto a = Random(count, -6, 6), b = Random(count, -6, 6);
  {
    float* x = Device(a);
    Expect(hy4_sigmoid_mul(x, Device(b), count, nullptr) == 0,
           "sigmoid_mul status");
    std::vector<double> want(count);
    for (uint64_t i = 0; i < count; ++i)
      want[i] = a[i] * Sigmoid(b[i]);
    Near(Host(x, count), want, 1e-6, "sigmoid_mul");
  }
  const auto rw = Random(rows, 0, 1);
  for (float limit : {0.0f, 3.5f}) {
    for (bool weighted : {false, true}) {
      float* y = Device(std::vector<float>(count));
      Expect(hy4_swiglu(Device(a), Device(b), weighted ? Device(rw) : nullptr,
                        y, rows, dim, limit, nullptr) == 0,
             "swiglu status");
      std::vector<double> want(count);
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t c = 0; c < dim; ++c) {
          double g = a[r * dim + c], u = b[r * dim + c];
          if (limit > 0) {
            g = std::min(g, (double)limit);
            u = std::clamp(u, -(double)limit, (double)limit);
          }
          want[r * dim + c] = g * Sigmoid(g) * u * (weighted ? rw[r] : 1.0);
        }
      Near(Host(y, count), want, 1e-6,
           "swiglu limit " + std::to_string(limit) +
               (weighted ? " weighted" : ""));
    }
  }
  {
    float* x = Device(a);
    Expect(hy4_add(x, Device(b), x, count, nullptr) == 0, "add status");
    std::vector<double> want(count);
    for (uint64_t i = 0; i < count; ++i)
      want[i] = (double)a[i] + b[i];
    Near(Host(x, count), want, 1e-7, "add in place");
  }
  {
    const uint32_t n = 2, k = 3;
    float* out = Device(std::vector<float>(n * dim));
    Expect(hy4_sum_slots(Device(a), out, n, k, dim, nullptr) == 0,
           "sum_slots status");
    std::vector<double> want(n * dim, 0.0);
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t s = 0; s < k; ++s)
        for (uint32_t c = 0; c < dim; ++c)
          want[t * dim + c] += a[(t * k + s) * dim + c];
    Near(Host(out, n * dim), want, 1e-6, "sum_slots");
  }
  {
    const uint32_t n = 2, per = 3;
    const auto bias = Random(per * dim);
    float* x = Device(a);
    Expect(hy4_add_bias_rows(x, Device(bias), n, per, dim, nullptr) == 0,
           "add_bias_rows status");
    std::vector<double> want(a.begin(), a.end());
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t r = 0; r < per; ++r)
        for (uint32_t c = 0; c < dim; ++c)
          want[(t * per + r) * dim + c] += bias[r * dim + c];
    Near(Host(x, count), want, 1e-7, "add_bias_rows");
  }
  {
    const uint64_t period = 150;
    const auto c = Random(period);
    float* x = Device(a);
    Expect(hy4_mul_add(x, Device(b), Device(c), count, period, nullptr) == 0,
           "mul_add status");
    std::vector<double> want(count);
    for (uint64_t i = 0; i < count; ++i)
      want[i] = a[i] + (double)b[i] * c[i % period];
    Near(Host(x, count), want, 1e-6, "mul_add");
  }
}

void TestArgmax() {
  float* work = Device(std::vector<float>(hy4_argmax_work_floats()));
  int32_t* out = Device(std::vector<int32_t>(1));
  auto check = [&](const std::vector<float>& x, const std::string& what) {
    Expect(hy4_argmax(Device(x), (uint32_t)x.size(), work, out, nullptr) == 0,
           what + " status");
    int32_t want = 0;
    auto key = [&](std::size_t i) {
      return std::isnan(x[i]) ? -INFINITY : x[i];
    };
    for (std::size_t i = 1; i < x.size(); ++i)
      if (key(i) > key(want))
        want = (int32_t)i;
    Expect(Host(out, 1)[0] == want, what);
  };
  // The Hy4 vocabulary: a tie across blocks goes to the lower index.
  auto x = Random(120832, -4, 4);
  x[77001] = x[501] = 9.0f;
  x[3] = NAN;
  check(x, "argmax tie");
  x[64000] = INFINITY;
  check(x, "argmax inf");
  // Most blocks hold only -inf; NaN never wins.
  std::vector<float> sparse(5000, -INFINITY);
  sparse[17] = NAN;
  sparse[4999] = 1.0f;
  check(sparse, "argmax sparse");
  check(std::vector<float>(300, -INFINITY), "argmax all -inf");
  check(Random(1), "argmax one");
}

void TestQuantizeHalf() {
  const uint64_t blocks = 3000;
  std::vector<uint16_t> src(blocks * 32);
  std::vector<float> values(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const _Float16 h = (_Float16)Random(1, -3, 3)[0];
    std::memcpy(&src[i], &h, 2);
    values[i] = (float)h;
  }
  for (int i = 0; i < 32; ++i) {  // an all-zero block
    src[32 + i] = 0;
    values[32 + i] = 0.0f;
  }
  BlockQ8_0* out = Device(std::vector<BlockQ8_0>(blocks));
  Expect(hy4_quantize_q8_0_half(Device(src), blocks, out, nullptr) == 0,
         "quantize_q8_0_half status");
  const auto got = Host(out, blocks);
  int mismatches = 0;
  for (uint64_t b = 0; b < blocks; ++b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(values[b * 32 + i]));
    const float d = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    const _Float16 dh = (_Float16)d;
    uint16_t dbits;
    std::memcpy(&dbits, &dh, 2);
    mismatches += got[b].d != dbits;
    for (int i = 0; i < 32; ++i)
      mismatches +=
          got[b].qs[i] != (int8_t)std::nearbyint(values[b * 32 + i] * inv);
  }
  Expect(mismatches == 0,
         "quantize_q8_0_half (" + std::to_string(mismatches) + " mismatches)");
}

// The same blocks over bfloat16 weights, the LM head of a trunk quantized
// with BF16 matrices.
void TestQuantizeBf16() {
  const uint64_t blocks = 3000;
  std::vector<uint16_t> src(blocks * 32);
  std::vector<float> values(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    uint32_t u;
    const float x = Random(1, -3, 3)[0];
    std::memcpy(&u, &x, 4);
    // Round to the nearest bfloat16, then widen back as the kernel does.
    const uint16_t bits =
        static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    src[i] = bits;
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    std::memcpy(&values[i], &widened, 4);
  }
  for (int i = 0; i < 32; ++i) {  // an all-zero block
    src[32 + i] = 0;
    values[32 + i] = 0.0f;
  }
  BlockQ8_0* out = Device(std::vector<BlockQ8_0>(blocks));
  Expect(hy4_quantize_q8_0_bf16(Device(src), blocks, out, nullptr) == 0,
         "quantize_q8_0_bf16 status");
  const auto got = Host(out, blocks);
  int mismatches = 0;
  for (uint64_t b = 0; b < blocks; ++b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i)
      amax = std::max(amax, std::fabs(values[b * 32 + i]));
    const float d = amax / 127.0f;
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    const _Float16 dh = (_Float16)d;
    uint16_t dbits;
    std::memcpy(&dbits, &dh, 2);
    mismatches += got[b].d != dbits;
    for (int i = 0; i < 32; ++i)
      mismatches +=
          got[b].qs[i] != (int8_t)std::nearbyint(values[b * 32 + i] * inv);
  }
  Expect(mismatches == 0,
         "quantize_q8_0_bf16 (" + std::to_string(mismatches) + " mismatches)");
}

void TestGatherRows() {
  const uint32_t rows = 50, words = 816;  // a 3,072-wide Q8_0 row
  std::vector<uint32_t> src(rows * words);
  for (std::size_t i = 0; i < src.size(); ++i)
    src[i] = (uint32_t)(i * 2654435761u);
  const std::vector<int32_t> ids = {49, 0, 7, 7, 31};
  uint32_t* dst = Device(std::vector<uint32_t>(ids.size() * words));
  Expect(hy4_gather_rows(Device(src), words * 4, Device(ids),
                         (uint32_t)ids.size(), dst, nullptr) == 0,
         "gather_rows status");
  const auto got = Host(dst, ids.size() * words);
  int mismatches = 0;
  for (std::size_t r = 0; r < ids.size(); ++r)
    for (uint32_t w = 0; w < words; ++w)
      mismatches += got[r * words + w] != src[ids[r] * words + w];
  Expect(mismatches == 0,
         "gather_rows (" + std::to_string(mismatches) + " mismatches)");
}

void TestRoute() {
  const uint32_t n = 3, experts = 192, k = 8;
  const float scale = 2.5f;
  const auto logits = Random(n * experts, -4, 4);
  const auto bias = Random(experts, -0.2f, 0.2f);
  for (int norm : {1, 0}) {
    int32_t* ids = Device(std::vector<int32_t>(n * k));
    float* weights = Device(std::vector<float>(n * k));
    Expect(hy4_route(Device(logits), Device(bias), ids, weights, n, experts, k,
                     norm, scale, nullptr) == 0,
           "route status");
    std::vector<int32_t> want_ids(n * k);
    std::vector<double> want_w(n * k);
    for (uint32_t t = 0; t < n; ++t) {
      std::vector<int32_t> order(experts);
      std::iota(order.begin(), order.end(), 0);
      auto sel = [&](int32_t e) {
        return Sigmoid(logits[t * experts + e]) + bias[e];
      };
      std::stable_sort(order.begin(), order.end(),
                       [&](int32_t x, int32_t y) { return sel(x) > sel(y); });
      double sum = 0;
      for (uint32_t s = 0; s < k; ++s)
        sum += Sigmoid(logits[t * experts + order[s]]);
      for (uint32_t s = 0; s < k; ++s) {
        want_ids[t * k + s] = order[s];
        want_w[t * k + s] = Sigmoid(logits[t * experts + order[s]]) /
                            (norm ? sum : 1.0) * scale;
      }
    }
    Expect(Host(ids, n * k) == want_ids,
           std::string("route ids") + (norm ? "" : " unnormalized"));
    Near(Host(weights, n * k), want_w, 1e-6,
         std::string("route weights") + (norm ? "" : " unnormalized"));
  }
  // Ties pick the lowest index first; 512 experts, the model's count, and
  // 16 picks. Logits take 24 distinct values, so every pick ties.
  {
    const uint32_t ne = 512, kk = 16;
    std::vector<float> tied(ne);
    for (uint32_t e = 0; e < ne; ++e)
      tied[e] = (float)((e * 7) % 24) / 8.0f - 1.5f;
    int32_t* ids = Device(std::vector<int32_t>(kk));
    float* weights = Device(std::vector<float>(kk));
    Expect(hy4_route(Device(tied), nullptr, ids, weights, 1, ne, kk, 1, 1.0f,
                     nullptr) == 0,
           "route ties status");
    std::vector<int32_t> order(ne);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int32_t x, int32_t y) { return tied[x] > tied[y]; });
    order.resize(kk);
    Expect(Host(ids, kk) == order, "route ties pick the lowest index");
  }
  Expect(hy4_route(nullptr, nullptr, nullptr, nullptr, 1, 8, 9, 1, 1.0f,
                   nullptr) != 0,
         "route rejects k > experts");
}

void TestEngram() {
  const uint32_t n = 3, cols = 5, d = 64;
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<uint8_t> rows(n * cols * d);
  for (auto& b : rows) {
    do
      b = (uint8_t)byte(g_rng);
    while ((b & 0x7f) == 0x7f);  // NaN codes
  }
  std::vector<float> valid(n * cols);
  for (std::size_t i = 0; i < valid.size(); ++i)
    valid[i] = i % 4 == 2 ? 0.0f : 1.0f;
  const auto norm_w = Random(cols * d, 0.5f, 1.5f);
  float* out = Device(std::vector<float>(n * cols * d));
  Expect(hy4_engram_rows(Device(rows), Device(valid), Device(norm_w), out, n,
                         cols, d, 1e-6f, nullptr) == 0,
         "engram_rows status");
  std::vector<double> want(n * cols * d);
  for (uint32_t r = 0; r < n * cols; ++r) {
    double ss = 0;
    for (uint32_t i = 0; i < d; ++i)
      ss += E4M3(rows[r * d + i]) * E4M3(rows[r * d + i]);
    const double s = 1.0 / std::sqrt(ss / d + 1e-6);
    for (uint32_t i = 0; i < d; ++i)
      want[r * d + i] =
          E4M3(rows[r * d + i]) * s * norm_w[(r % cols) * d + i] * valid[r];
  }
  Near(Host(out, n * cols * d), want, 2e-6, "engram_rows");

  const uint32_t hc = 4, dim = 128;
  const float eps = 1e-6f;
  const auto h = Random(n * hc * dim, -2, 2);
  const auto kv = Random(n * (hc + 1) * dim, -2, 2);
  const auto gate_w = Random(hc * dim, 0, 2);
  const auto hc_scale = Random(hc * dim);
  const auto gamma = Random(dim, 0.5f, 1.5f);
  const std::vector<float> any{1.0f, 0.0f, 1.0f};
  float* dh = Device(h);
  Expect(hy4_engram_apply(dh, Device(kv), Device(gate_w), Device(hc_scale),
                          Device(gamma), Device(any), n, hc, dim, eps,
                          nullptr) == 0,
         "engram_apply status");
  std::vector<double> want_h(h.begin(), h.end());
  for (uint32_t t = 0; t < n; ++t) {
    const auto value = RmsRow(&kv[(t * (hc + 1) + hc) * dim], dim, eps);
    for (uint32_t i = 0; i < hc; ++i) {
      const auto hn = RmsRow(&h[(t * hc + i) * dim], dim, eps);
      const auto key = RmsRow(&kv[(t * (hc + 1) + i) * dim], dim, eps);
      double dot = 0;
      for (uint32_t c = 0; c < dim; ++c)
        dot += hn[c] * key[c] * gate_w[i * dim + c];
      dot /= std::sqrt((double)dim);
      const double mag = std::sqrt(std::max(std::fabs(dot), 1e-6));
      const double g = Sigmoid(dot < 0 ? -mag : mag);
      for (uint32_t c = 0; c < dim; ++c)
        want_h[(t * hc + i) * dim + c] += (any[t] > 0 ? 1.0 : 0.0) * value[c] *
                                          gamma[c] *
                                          (g + hc_scale[i * dim + c]);
    }
  }
  Near(Host(dh, n * hc * dim), want_h, 1e-5, "engram_apply");
}

void TestKpool() {
  const uint32_t dim = 128, kpool = 4, ring = 12, pool_ring = 5;
  const float eps = 1e-6f;
  const auto rows = Random(ring * 2 * dim, -2, 2);
  const auto ape = Random(kpool * dim);
  const auto norm_w = Random(dim, 0.5f, 1.5f);
  const std::vector<int32_t> groups{
      2, 3, 4};  // 3 and 4 wrap the row ring and the pool ring
  const auto start = Random(pool_ring * dim);
  float* pooled = Device(start);
  Expect(
      hy4_kpool_pool(Device(rows), ring, Device(groups),
                     (uint32_t)groups.size(), kpool, Device(ape),
                     Device(norm_w), eps, pooled, pool_ring, dim, nullptr) == 0,
      "kpool_pool status");
  std::vector<double> want(start.begin(), start.end());
  for (int32_t g : groups) {
    std::vector<double> p(dim);
    for (uint32_t c = 0; c < dim; ++c) {
      double mx = -INFINITY, den = 0, acc = 0;
      for (uint32_t j = 0; j < kpool; ++j)
        mx = std::max(
            mx, (double)rows[((g * kpool + j) % ring) * 2 * dim + dim + c] +
                    ape[j * dim + c]);
      for (uint32_t j = 0; j < kpool; ++j) {
        const float* r = &rows[((g * kpool + j) % ring) * 2 * dim];
        const double e = std::exp(r[dim + c] + (double)ape[j * dim + c] - mx);
        den += e;
        acc += e * r[c];
      }
      p[c] = acc / den;
    }
    double ss = 0;
    for (double v : p)
      ss += v * v;
    const double s = 1.0 / std::sqrt(ss / dim + eps);
    for (uint32_t c = 0; c < dim; ++c)
      want[(g % pool_ring) * dim + c] = p[c] * s * norm_w[c];
  }
  Near(Host(pooled, pool_ring * dim), want, 1e-5, "kpool_pool");

  // Indexer scores over each token's visible pools, then the top-k selection.
  const uint32_t n = 3, heads = 4, n_pool_max = 9, k = 4;
  const auto q = Random(n * heads * dim);
  const auto w = Random(n * heads, 0, 1);
  const auto pool_rows = Random(n_pool_max * dim);
  const std::vector<int32_t> n_pool{2, 9, 6};
  float* score = Device(std::vector<float>(n * n_pool_max));
  Expect(
      hy4_indexer_score(Device(q), Device(w), Device(pool_rows), Device(n_pool),
                        n_pool_max, score, n, heads, dim, nullptr) == 0,
      "indexer_score status");
  std::vector<double> want_score(n * n_pool_max, -INFINITY);
  for (uint32_t t = 0; t < n; ++t)
    for (int32_t g = 0; g < n_pool[t]; ++g) {
      double total = 0;
      for (uint32_t h = 0; h < heads; ++h) {
        double d = 0;
        for (uint32_t i = 0; i < dim; ++i)
          d += (double)q[(t * heads + h) * dim + i] * pool_rows[g * dim + i];
        total += std::max(d, 0.0) * w[t * heads + h];
      }
      want_score[t * n_pool_max + g] = total;
    }
  auto scores = Host(score, n * n_pool_max);
  Near(scores, want_score, 1e-5, "indexer_score");
  {  // The model's 16 heads, several blocks of groups and of tokens.
    const uint32_t n2 = 6, heads2 = 16, groups2 = 300;
    const auto q2 = Random(n2 * heads2 * dim);
    const auto w2 = Random(n2 * heads2, 0, 1);
    const auto rows2 = Random(groups2 * dim);
    const std::vector<int32_t> visible{1, 300, 257, 0, 120, 299};
    float* score2 = Device(std::vector<float>(n2 * groups2));
    Expect(hy4_indexer_score(Device(q2), Device(w2), Device(rows2),
                             Device(visible), groups2, score2, n2, heads2, dim,
                             nullptr) == 0,
           "indexer_score 16 heads status");
    std::vector<double> want2(n2 * groups2, -INFINITY);
    for (uint32_t t = 0; t < n2; ++t)
      for (int32_t g = 0; g < visible[t]; ++g) {
        double total = 0;
        for (uint32_t h = 0; h < heads2; ++h) {
          double d = 0;
          for (uint32_t i = 0; i < dim; ++i)
            d += (double)q2[(t * heads2 + h) * dim + i] * rows2[g * dim + i];
          total += std::max(d, 0.0) * w2[t * heads2 + h];
        }
        want2[t * groups2 + g] = total;
      }
    Near(Host(score2, n2 * groups2), want2, 1e-5, "indexer_score 16 heads");
  }

  // Equal scores pick the lower group first.
  scores[1 * n_pool_max + 7] = scores[1 * n_pool_max + 3] = 1e3f;
  const std::vector<int32_t> pos{9, 37, 24};
  const uint32_t n_sel = (k + 1) * kpool;
  int32_t* sel = Device(std::vector<int32_t>(n * n_sel, 99));
  Expect(hy4_kpool_select(Device(scores), Device(n_pool), n_pool_max,
                          Device(pos), k, kpool, sel, n, nullptr) == 0,
         "kpool_select status");
  std::vector<int32_t> want_sel(n * n_sel, -1);
  for (uint32_t t = 0; t < n; ++t) {
    std::vector<int32_t> order(n_pool[t]);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return scores[t * n_pool_max + a] > scores[t * n_pool_max + b];
    });
    for (uint32_t s = 0; s < std::min<uint32_t>(k, n_pool[t]); ++s)
      for (uint32_t j = 0; j < kpool; ++j)
        want_sel[t * n_sel + s * kpool + j] = order[s] * kpool + j;
    const int32_t first = pos[t] - pos[t] % kpool;
    for (uint32_t j = 0; j < kpool; ++j)
      want_sel[t * n_sel + k * kpool + j] =
          first + (int32_t)j <= pos[t] ? first + (int32_t)j : -1;
  }
  Expect(Host(sel, n * n_sel) == want_sel, "kpool_select");
}

void TestAttnSoftmax() {
  const uint32_t n = 3, heads = 2, n_keys = 24;
  const int32_t key_pos0 = 10;
  const std::vector<int32_t> pos{12, 25, 40};
  const auto scores = Random(n * heads * n_keys, -3, 3);
  const auto sinks = Random(heads);
  // Selection mask: token 0 sees nothing selected (only the sink).
  const std::vector<int32_t> sel{-1, -1, -1, 3, 5, 6, 20, 23, -1};
  std::vector<float> mask_host(n * n_keys, -INFINITY);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < 3; ++i)
      if (sel[t * 3 + i] >= 0)
        mask_host[t * n_keys + sel[t * 3 + i]] = 0.0f;
  float* mask = Device(mask_host);

  for (uint32_t window : {0u, 6u}) {
    for (bool masked : {false, true}) {
      for (bool sink : {true, false}) {
        float* s = Device(scores);
        Expect(hy4_attn_softmax(s, n, heads, n_keys, key_pos0, Device(pos),
                                window, masked ? mask : nullptr,
                                sink ? Device(sinks) : nullptr, nullptr) == 0,
               "attn_softmax status");
        std::vector<double> want(n * heads * n_keys, 0.0);
        for (uint32_t t = 0; t < n; ++t)
          for (uint32_t h = 0; h < heads; ++h) {
            const uint32_t r = t * heads + h;
            auto visible = [&](uint32_t j) {
              const int32_t kp = key_pos0 + (int32_t)j;
              return kp <= pos[t] &&
                     (!window || kp > pos[t] - (int32_t)window) &&
                     (!masked || mask_host[t * n_keys + j] == 0.0f);
            };
            double mx = sink ? sinks[h] : -INFINITY;
            for (uint32_t j = 0; j < n_keys; ++j)
              if (visible(j))
                mx = std::max(mx, (double)scores[r * n_keys + j]);
            double den = sink ? std::exp(sinks[h] - mx) : 0.0;
            for (uint32_t j = 0; j < n_keys; ++j)
              if (visible(j))
                den += std::exp(scores[r * n_keys + j] - mx);
            for (uint32_t j = 0; j < n_keys; ++j)
              if (visible(j) && den > 0)
                want[r * n_keys + j] =
                    std::exp(scores[r * n_keys + j] - mx) / den;
          }
        Near(Host(s, n * heads * n_keys), want, 1e-6,
             "attn_softmax window " + std::to_string(window) +
                 (masked ? " masked" : "") + (sink ? " sinks" : ""));
      }
    }
  }
}

}  // namespace

// Random Q8_0, Q4_K or Q6_K rows as raw blocks, and their values from the
// ggml reference dequantization (dequantize_row_q4_K / _q6_K).
std::vector<uint8_t> RandomQuant(int type, std::size_t rows, std::size_t k) {
  const std::size_t bytes = type == HY4_TYPE_Q8_0   ? 34
                            : type == HY4_TYPE_Q4_K ? 144
                                                    : 210;
  const std::size_t values = type == HY4_TYPE_Q8_0 ? 32 : 256;
  std::uniform_int_distribution<int> byte(0, 255), s(1, 60);
  std::vector<uint8_t> v(rows * k / values * bytes);
  for (auto& b : v)
    b = (uint8_t)byte(g_rng);
  for (std::size_t i = 0; i < v.size() / bytes; ++i) {
    uint8_t* b = v.data() + i * bytes;
    const auto put = [&](std::size_t at, float x) {
      const uint16_t h = HalfBits(x);
      std::memcpy(b + at, &h, 2);
    };
    // Weights of magnitude up to about 0.1, as in a trained model.
    if (type == HY4_TYPE_Q8_0) {
      put(0, (float)s(g_rng) / 65536.0f);
    } else if (type == HY4_TYPE_Q4_K) {
      put(0, (float)s(g_rng) / 262144.0f);
      put(2, (float)s(g_rng) / 262144.0f);
    } else {
      put(208, (float)s(g_rng) / 4194304.0f);
    }
  }
  return v;
}

std::vector<double> Dequant(int type, const std::vector<uint8_t>& v) {
  std::vector<double> y;
  if (type == HY4_TYPE_Q8_0) {
    for (std::size_t i = 0; i < v.size(); i += 34) {
      uint16_t d;
      std::memcpy(&d, &v[i], 2);
      for (int j = 0; j < 32; ++j)
        y.push_back(HalfValue(d) * (int8_t)v[i + 2 + j]);
    }
    return y;
  }
  if (type == HY4_TYPE_Q4_K) {
    for (std::size_t i = 0; i < v.size(); i += 144) {
      const uint8_t* b = &v[i];
      uint16_t dh, mh;
      std::memcpy(&dh, b, 2);
      std::memcpy(&mh, b + 2, 2);
      const double d = HalfValue(dh), dmin = HalfValue(mh);
      const uint8_t* sc = b + 4;
      const uint8_t* q = b + 16;
      const auto scale_min = [&](int j, int* scale, int* min) {
        if (j < 4) {
          *scale = sc[j] & 63;
          *min = sc[j + 4] & 63;
        } else {
          *scale = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
          *min = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
      };
      for (int j = 0, is = 0; j < 256; j += 64, is += 2, q += 32) {
        int s1, m1, s2, m2;
        scale_min(is, &s1, &m1);
        scale_min(is + 1, &s2, &m2);
        for (int l = 0; l < 32; ++l)
          y.push_back(d * s1 * (q[l] & 0xF) - dmin * m1);
        for (int l = 0; l < 32; ++l)
          y.push_back(d * s2 * (q[l] >> 4) - dmin * m2);
      }
    }
    return y;
  }
  for (std::size_t i = 0; i < v.size(); i += 210) {
    const uint8_t* ql = &v[i];
    const uint8_t* qh = ql + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(ql + 192);
    uint16_t dh;
    std::memcpy(&dh, ql + 208, 2);
    const double d = HalfValue(dh);
    double out[256];
    for (int n = 0; n < 256; n += 128, ql += 64, qh += 32, sc += 8)
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        out[n + l] =
            d * sc[is] * (((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        out[n + l + 32] =
            d * sc[is + 2] *
            (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        out[n + l + 64] =
            d * sc[is + 4] * (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        out[n + l + 96] =
            d * sc[is + 6] *
            (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
    y.insert(y.end(), out, out + 256);
  }
  return y;
}

// The FP16 matrix-core projections against the dequantized weights and the
// FP16-rounded activations: rows past 256-row blocks, partial token tiles,
// groups, pairs of different heights, FP16 inputs and outputs, and routed
// experts with a bucket over one tile, an empty expert and the clamp.
void TestSgemm() {
  const uint32_t m = 150, k = 96, n = 70;
  const auto x = Random(n * k), w = Random(m * k);
  std::vector<double> want(n * m);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t r = 0; r < m; ++r) {
      double acc = 0;
      for (uint32_t c = 0; c < k; ++c)
        acc += (double)x[t * k + c] * w[r * k + c];
      want[t * m + r] = acc;
    }
  auto* dx = Device(x);
  auto* dw = Device(w);
  float* dy = Device(std::vector<float>(n * m));
  Expect(hy4_sgemm_nt(dx, dw, dy, n, m, k, nullptr) == 0, "sgemm_nt status");
  const auto got = Host(dy, n * m);
  Near(got, want, 1e-6, "sgemm_nt");
  float* d5 = Device(std::vector<float>(5 * m));
  Expect(hy4_sgemm_nt(dx, dw, d5, 5, m, k, nullptr) == 0, "sgemm_nt 5 status");
  Expect(std::memcmp(Host(d5, 5 * m).data(), got.data(),
                     5 * m * sizeof(float)) == 0,
         "sgemm_nt rows independent of the pass");
  Expect(hy4_sgemm_nt(dx, dw, dy, n, m, 100, nullptr) != 0,
         "sgemm_nt rejects k % 32");
}

// BF16 weights (the router): the conversion's exactness flag and rounding,
// the split GEMV against double precision and its one-token calls, and the
// batched GEMM against the F32 GEMM on the widened weights.
void TestBf16() {
  const auto bf16 = [](float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return static_cast<uint16_t>(u >> 16);
  };
  const auto widen = [](uint16_t b) {
    const uint32_t u = static_cast<uint32_t>(b) << 16;
    float v;
    std::memcpy(&v, &u, 4);
    return v;
  };
  const uint32_t m = 37, k = 256;
  auto w = Random(m * k);
  std::vector<uint16_t> wb(m * k);
  for (std::size_t i = 0; i < w.size(); ++i) {
    wb[i] = bf16(w[i]);
    w[i] = widen(wb[i]);
  }
  int32_t* flag = Device(std::vector<int32_t>(1, 0));
  uint16_t* conv = Device(std::vector<uint16_t>(m * k));
  Expect(hy4_f32_to_bf16(Device(w), m * k, conv, flag, nullptr) == 0,
         "f32_to_bf16 status");
  Expect(Host(flag, 1)[0] == 0, "f32_to_bf16 exact values");
  Expect(Host(conv, m * k) == wb, "f32_to_bf16 exact bits");
  // 1 + 2^-8 + 2^-9 rounds up to 1 + 2^-7; 1 + 2^-8 ties to even (1).
  const std::vector<float> odd = {1.0f + 0x1p-8f + 0x1p-9f, 1.0f + 0x1p-8f};
  uint16_t* oc = Device(std::vector<uint16_t>(2));
  Expect(hy4_f32_to_bf16(Device(odd), 2, oc, flag, nullptr) == 0,
         "f32_to_bf16 inexact status");
  Expect(Host(flag, 1)[0] == 1, "f32_to_bf16 flags inexact values");
  const auto ob = Host(oc, 2);
  Expect(ob[0] == bf16(1.0f + 0x1p-7f) && ob[1] == bf16(1.0f),
         "f32_to_bf16 rounds to nearest even");

  const uint16_t* dwb = Device(wb);
  for (uint32_t n : {1u, 3u, 11u}) {
    const auto x = Random(n * k);
    float* dx = Device(x);
    float* y = Device(std::vector<float>(n * m));
    Expect(hy4_gemv_float(HY4_TYPE_BF16, dwb, dx, y, m, k, n, nullptr) == 0,
           "gemv_float bf16 status");
    std::vector<double> want(n * m, 0.0);
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t r = 0; r < m; ++r)
        for (uint32_t i = 0; i < k; ++i)
          want[t * m + r] += (double)w[r * k + i] * x[t * k + i];
    const auto many = Host(y, n * m);
    Near(many, want, 1e-5, "gemv_float bf16 n=" + std::to_string(n));
    bool same = true;
    for (uint32_t t = 0; t < n; ++t) {
      float* y1 = Device(std::vector<float>(m));
      Expect(hy4_gemv_float(HY4_TYPE_BF16, dwb, dx + t * k, y1, m, k, 1,
                            nullptr) == 0,
             "gemv_float bf16 one-token status");
      same &= std::memcmp(Host(y1, m).data(), many.data() + t * m, m * 4) == 0;
    }
    Expect(same, "gemv_float bf16 tokens equal one-token calls n=" +
                     std::to_string(n));
  }
  Expect(hy4_gemv_float(HY4_TYPE_BF16, dwb, nullptr, nullptr, m, 72, 1,
                        nullptr) != 0,
         "gemv_float bf16 rejects k % 32");

  const uint32_t n = 70;
  const auto x = Random(n * k);
  float* dx = Device(x);
  float* a = Device(std::vector<float>(n * m));
  float* b = Device(std::vector<float>(n * m));
  Expect(hy4_sgemm_nt(dx, Device(w), a, n, m, k, nullptr) == 0 &&
             hy4_sgemm_nt_bf16(dx, dwb, b, n, m, k, nullptr) == 0,
         "sgemm_nt_bf16 status");
  Expect(Host(a, n * m) == Host(b, n * m),
         "sgemm_nt_bf16 equals sgemm_nt on the widened weights");
}

// The FP32-activation decode GEMVs against double precision on the
// dequantized weights: rows past a 32-row tile, every token count the
// dispatch distinguishes, each token equal to a one-token call, the split
// layout (scratch) equal to the lane layout (none), pairs equal to single
// calls, and the fused SwiGLU.
// hy4_gemv_experts against a double reference, and verify passes (slots
// sharing experts) bit for bit against one-slot launches.
void TestExpertGemv() {
  const uint32_t m = 70, k = 512;
  std::mt19937 rng(17);
  for (int type : {HY4_TYPE_Q8_0, HY4_TYPE_Q4_K, HY4_TYPE_Q6_K})
    for (auto [n_exp, used] : {std::pair<uint32_t, uint32_t>{6, 3}, {24, 16}}) {
      const std::string name = "experts type " + std::to_string(type) +
                               " used " + std::to_string(used);
      const uint64_t bytes = RandomQuant(type, m, k).size();
      std::vector<uint8_t> gq, uq, dq;
      for (uint32_t e = 0; e < n_exp; ++e) {
        for (auto* v : {&gq, &uq}) {
          const auto q = RandomQuant(type, m, k);
          v->insert(v->end(), q.begin(), q.end());
        }
        // The down matrices: also m x k, over their own input rows.
        const auto q = RandomQuant(type, m, k);
        dq.insert(dq.end(), q.begin(), q.end());
      }
      const auto gd = Dequant(type, gq), ud = Dequant(type, uq),
                 dd = Dequant(type, dq);
      const uint8_t *dg = Device(gq), *du = Device(uq), *ddn = Device(dq);
      for (uint32_t n : {1u, 2u, 3u, 5u, 8u, 16u}) {
        const uint32_t slots = n * used;
        std::vector<int32_t> ids(slots);
        for (uint32_t t = 0; t < n; ++t) {
          std::vector<int32_t> p(n_exp);
          for (uint32_t e = 0; e < n_exp; ++e)
            p[e] = (int32_t)e;
          std::shuffle(p.begin(), p.end(), rng);
          for (uint32_t j = 0; j < used; ++j)
            ids[t * used + j] = p[j];
        }
        const auto x = Random(n * k), rw = Random(slots, 0.1f, 1.0f);
        const auto mid = Random(slots * k);
        int32_t* did = Device(ids);
        float *dx = Device(x), *drw = Device(rw), *dmid = Device(mid);
        for (float limit : {0.0f, 0.05f}) {
          std::vector<double> want(slots * m), want_d(slots * m);
          for (uint32_t j = 0; j < slots; ++j) {
            const uint64_t e = ids[j];
            const float* xr = x.data() + (j / used) * k;
            for (uint32_t r = 0; r < m; ++r) {
              double g = 0, u = 0, d = 0;
              for (uint32_t i = 0; i < k; ++i) {
                g += gd[(e * m + r) * k + i] * xr[i];
                u += ud[(e * m + r) * k + i] * xr[i];
                d += dd[(e * m + r) * k + i] * mid[j * k + i];
              }
              if (limit > 0) {
                u = std::min(std::max(u, (double)-limit), (double)limit);
                g = std::min(g, (double)limit);
              }
              want[j * m + r] = g / (1 + std::exp(-g)) * u * rw[j];
              want_d[j * m + r] = d;
            }
          }
          std::vector<float> one, one_d;
          for (uint32_t cols : {1u, n}) {
            float *y = Device(std::vector<float>(slots * m)),
                  *yd = Device(std::vector<float>(slots * m));
            Expect(
                hy4_gemv_experts(type, dg, du, bytes, dx, used, did, drw, limit,
                                 y, m, k, slots, cols, nullptr) == 0 &&
                    hy4_gemv_experts(type, ddn, nullptr, bytes, dmid, 1, did,
                                     nullptr, 0.0f, yd, m, k, slots, cols,
                                     nullptr) == 0,
                name + " status");
            const auto got = Host(y, slots * m), got_d = Host(yd, slots * m);
            const std::string what = name + " n=" + std::to_string(n) +
                                     " limit=" + std::to_string(limit);
            if (cols == 1) {
              Near(got, want, 1e-5, what + " swiglu");
              Near(got_d, want_d, 1e-5, what + " down");
              one = got;
              one_d = got_d;
            } else {
              Expect(got == one && got_d == one_d,
                     what + " shared experts equal one-slot launches");
            }
          }
        }
      }
    }
}

void TestDecodeGemv() {
  const uint32_t k = 512;
  void* scratch = nullptr;
  Expect(hipMalloc(&scratch, hy4_gemv_scratch_bytes()) == hipSuccess &&
             hipMemset(scratch, 0, hy4_gemv_scratch_bytes()) == hipSuccess,
         "gemv scratch");
  for (int type : {HY4_TYPE_Q8_0, HY4_TYPE_Q4_K, HY4_TYPE_Q6_K}) {
    const std::string name = "gemv type " + std::to_string(type);
    for (uint32_t m : {96u, 70u}) {
      const auto wq = RandomQuant(type, m, k);
      const auto wd = Dequant(type, wq);
      const auto x = Random(16 * k);
      const uint8_t* dw = Device(wq);
      float* dx = Device(x);
      std::vector<std::vector<float>> one(16);
      for (uint32_t t = 0; t < 16; ++t) {
        float* y = Device(std::vector<float>(m));
        Expect(
            hy4_gemv(type, dw, dx + t * k, y, m, k, 1, scratch, nullptr) == 0,
            name + " status");
        one[t] = Host(y, m);
      }
      std::vector<double> want(16 * m, 0.0);
      for (uint32_t t = 0; t < 16; ++t)
        for (uint32_t r = 0; r < m; ++r)
          for (uint32_t i = 0; i < k; ++i)
            want[t * m + r] += wd[r * k + i] * x[t * k + i];
      std::vector<float> got;
      for (const auto& v : one)
        got.insert(got.end(), v.begin(), v.end());
      Near(got, want, 1e-5, name + " m=" + std::to_string(m));
      for (uint32_t n : {2u, 3u, 4u, 5u, 6u, 8u, 11u, 16u}) {
        for (void* sc : {scratch, static_cast<void*>(nullptr)}) {
          float* y = Device(std::vector<float>(n * m));
          Expect(hy4_gemv(type, dw, dx, y, m, k, n, sc, nullptr) == 0,
                 name + " status");
          const auto many = Host(y, n * m);
          bool same = true;
          for (uint32_t t = 0; t < n; ++t)
            same &= std::memcmp(many.data() + t * m, one[t].data(), m * 4) == 0;
          Expect(same,
                 name + " m=" + std::to_string(m) + " n=" + std::to_string(n) +
                     (sc ? " split" : " lanes") + " equals one-token calls");
        }
      }
    }
    // A pair as two single calls, and the SwiGLU of gate and up.
    const uint32_t m1 = 64, m2 = 96, n = 6;
    const auto w1 = RandomQuant(type, m1, k), w2 = RandomQuant(type, m2, k);
    const auto x = Random(n * k);
    const uint8_t *d1 = Device(w1), *d2 = Device(w2);
    float* dx = Device(x);
    float *a1 = Device(std::vector<float>(n * m1)),
          *a2 = Device(std::vector<float>(n * m2)),
          *b1 = Device(std::vector<float>(n * m1)),
          *b2 = Device(std::vector<float>(n * m2));
    Expect(hy4_gemv_pair(type, d1, a1, m1, d2, a2, m2, dx, k, n, scratch,
                         nullptr) == 0 &&
               hy4_gemv(type, d1, dx, b1, m1, k, n, scratch, nullptr) == 0 &&
               hy4_gemv(type, d2, dx, b2, m2, k, n, scratch, nullptr) == 0,
           name + " pair status");
    Expect(Host(a1, n * m1) == Host(b1, n * m1) &&
               Host(a2, n * m2) == Host(b2, n * m2),
           name + " pair equals single calls");
    const auto up = RandomQuant(type, m1, k);
    const uint8_t* du = Device(up);
    float* u = Device(std::vector<float>(n * m1));
    for (uint32_t cols : {1u, n}) {
      for (float limit : {0.0f, 0.05f}) {
        float* y = Device(std::vector<float>(cols * m1));
        Expect(
            hy4_gemv_swiglu(type, d1, du, dx, y, m1, k, limit, cols, scratch,
                            nullptr) == 0 &&
                hy4_gemv(type, du, dx, u, m1, k, cols, scratch, nullptr) == 0,
            name + " swiglu status");
        const auto g = Host(b1, cols * m1), uv = Host(u, cols * m1);
        std::vector<double> want(cols * m1);
        for (std::size_t i = 0; i < want.size(); ++i) {
          double gi = g[i], ui = uv[i];
          if (limit > 0) {
            gi = std::min<double>(gi, limit);
            ui = std::clamp<double>(ui, -limit, limit);
          }
          want[i] = gi / (1.0 + std::exp(-gi)) * ui;
        }
        Near(Host(y, cols * m1), want, 1e-6,
             name + " swiglu n=" + std::to_string(cols) +
                 " limit=" + std::to_string(limit));
        // As hy4_gemv's gate and up then hy4_swiglu compute it.
        float* s2 = Device(std::vector<float>(cols * m1));
        Expect(hy4_swiglu(b1, u, nullptr, s2, cols, m1, limit, nullptr) == 0,
               name + " swiglu reference status");
        Expect(Host(y, cols * m1) == Host(s2, cols * m1),
               name + " swiglu equals hy4_swiglu n=" + std::to_string(cols));
      }
    }
  }
  for (int type : {HY4_TYPE_Q8_0, HY4_TYPE_Q4_K, HY4_TYPE_Q6_K})
    for (uint32_t n : {1u, 2u, 6u, 16u}) {  // the lane and split layouts
      const uint32_t m = 96;
      const auto wq = RandomQuant(type, m, k);
      const auto x = Random(n * k), add = Random(n * m);
      float *a = Device(std::vector<float>(n * m)),
            *b = Device(std::vector<float>(n * m));
      Expect(hy4_gemv(type, Device(wq), Device(x), a, m, k, n, scratch,
                      nullptr) == 0 &&
                 hy4_add(a, Device(add), a, n * m, nullptr) == 0 &&
                 hy4_gemv_add(type, Device(wq), Device(x), Device(add), b, m, k,
                              n, scratch, nullptr) == 0,
             "gemv_add status");
      Expect(Host(a, n * m) == Host(b, n * m),
             "gemv_add equals gemv then add, type " + std::to_string(type) +
                 " n=" + std::to_string(n));
      // The gate mode: y *= sigmoid(W x), as hy4_sigmoid_mul scales it.
      const auto y0 = Random(n * m, -2, 2);
      float *g = Device(std::vector<float>(n * m)), *ya = Device(y0),
            *yb = Device(y0);
      Expect(hy4_gemv(type, Device(wq), Device(x), g, m, k, n, scratch,
                      nullptr) == 0 &&
                 hy4_sigmoid_mul(ya, g, n * m, nullptr) == 0 &&
                 hy4_gemv_gate(type, Device(wq), Device(x), yb, m, k, n,
                               scratch, nullptr) == 0,
             "gemv_gate status");
      Expect(Host(ya, n * m) == Host(yb, n * m),
             "gemv_gate equals gemv then sigmoid_mul, type " +
                 std::to_string(type) + " n=" + std::to_string(n));
    }
  Expect(hy4_gemv(HY4_TYPE_Q8_0, nullptr, nullptr, nullptr, 32, 96, 1, scratch,
                  nullptr) != 0,
         "gemv rejects k % 256");
  Expect(hy4_gemv(HY4_TYPE_Q8_0, nullptr, nullptr, nullptr, 32, 256, 17,
                  scratch, nullptr) != 0,
         "gemv rejects more than 16 tokens");
  Expect(hy4_gemv(HY4_TYPE_F16, nullptr, nullptr, nullptr, 32, 256, 1, scratch,
                  nullptr) != 0,
         "gemv rejects other types");
  (void)hipFree(scratch);
}

void TestF16Matmul() {
  const auto half_round = [](const std::vector<float>& x) {
    std::vector<double> r(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      r[i] = HalfValue(HalfBits(x[i]));
    return r;
  };
  const auto project = [](const std::vector<double>& w,
                          const std::vector<double>& x, uint32_t m, uint32_t n,
                          uint32_t k, uint32_t groups) {
    std::vector<double> y((std::size_t)n * groups * m);
    for (uint32_t t = 0; t < n; ++t)
      for (uint32_t z = 0; z < groups; ++z)
        for (uint32_t r = 0; r < m; ++r) {
          double acc = 0;
          for (uint32_t c = 0; c < k; ++c)
            acc += w[((std::size_t)z * m + r) * k + c] *
                   x[((std::size_t)t * groups + z) * k + c];
          y[((std::size_t)t * groups + z) * m + r] = acc;
        }
    return y;
  };
  struct Case {
    int type;
    uint32_t m, k, n, groups;
    const char* name;
  };
  for (const Case& c : {Case{HY4_TYPE_Q8_0, 272, 512, 150, 1, "q8_0"},
                        Case{HY4_TYPE_Q4_K, 48, 256, 37, 1, "q4_k"},
                        Case{HY4_TYPE_Q6_K, 32, 512, 130, 3, "q6_k grouped"}}) {
    const auto w = RandomQuant(c.type, (std::size_t)c.m * c.groups, c.k);
    const auto x = Random((std::size_t)c.n * c.groups * c.k);
    const auto want =
        project(Dequant(c.type, w), half_round(x), c.m, c.n, c.k, c.groups);
    auto* dw = Device(w);
    auto* dx = Device(x);
    const std::size_t outs = (std::size_t)c.n * c.groups * c.m;
    float* dy = Device(std::vector<float>(outs));
    Expect(hy4_f16_dense(c.type, dw, dx, dy, c.m, c.n, c.k, c.groups, 0,
                         nullptr) == 0,
           std::string("f16_dense status ") + c.name);
    const auto full = Host(dy, outs);
    Near(full, want, 2e-3, std::string("f16_dense ") + c.name);
    // A token's rows do not depend on the pass it is in: the first 5 tokens
    // alone give the same bits (prompt passes rely on it).
    {
      const uint32_t few = 5;
      const std::size_t few_outs = (std::size_t)few * c.groups * c.m;
      float* dyf = Device(std::vector<float>(few_outs));
      Expect(hy4_f16_dense(c.type, dw, dx, dyf, c.m, few, c.k, c.groups, 0,
                           nullptr) == 0,
             std::string("f16_dense few status ") + c.name);
      const auto part = Host(dyf, few_outs);
      Expect(
          std::memcmp(part.data(), full.data(), few_outs * sizeof(float)) == 0,
          std::string("f16_dense rows independent of the pass ") + c.name);
      (void)hipFree(dyf);
    }
    // FP16 in and out: the same sums, rounded once more on the way out.
    std::vector<uint16_t> xh(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      xh[i] = HalfBits(x[i]);
    auto* dxh = Device(xh);
    auto* dyh = Device(std::vector<uint16_t>(outs));
    Expect(hy4_f16_dense(c.type, dw, dxh, dyh, c.m, c.n, c.k, c.groups,
                         HY4_F16_X_HALF | HY4_F16_OUT_HALF, nullptr) == 0,
           std::string("f16_dense half status ") + c.name);
    const auto yh = Host(dyh, outs);
    std::vector<float> got(outs);
    for (std::size_t i = 0; i < outs; ++i)
      got[i] = (float)HalfValue(yh[i]);
    Near(got, want, 3e-3, std::string("f16_dense half ") + c.name);
    (void)hipFree(dw);
    (void)hipFree(dx);
    (void)hipFree(dy);
    (void)hipFree(dxh);
    (void)hipFree(dyh);
  }

  {  // Gated: outputs times sigmoid(gate), stored as binary16.
    const uint32_t m = 48, k = 256, n = 140, groups = 2;
    const auto w = RandomQuant(HY4_TYPE_Q8_0, (std::size_t)m * groups, k);
    const auto x = Random((std::size_t)n * groups * k);
    const auto g = Random((std::size_t)n * groups * m, -4.0f, 4.0f);
    auto want =
        project(Dequant(HY4_TYPE_Q8_0, w), half_round(x), m, n, k, groups);
    for (std::size_t i = 0; i < want.size(); ++i)
      want[i] *= Sigmoid(g[i]);
    auto* dw = Device(w);
    auto* dx = Device(x);
    auto* dg = Device(g);
    auto* dy = Device(std::vector<uint16_t>(want.size()));
    Expect(hy4_f16_dense_gated(HY4_TYPE_Q8_0, dw, dx, dy, m, n, k, groups,
                               HY4_F16_OUT_HALF, dg, nullptr) == 0,
           "f16_dense_gated status");
    const auto bits = Host(dy, want.size());
    std::vector<float> got(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
      got[i] = (float)HalfValue(bits[i]);
    Near(got, want, 3e-3, "f16_dense_gated");
    for (void* p : {(void*)dw, (void*)dx, (void*)dg, (void*)dy})
      (void)hipFree(p);
  }

  {  // Pair of different heights, both partial 256-row blocks.
    const uint32_t m1 = 32, m2 = 48, k = 256, n = 140;
    const auto w1 = RandomQuant(HY4_TYPE_Q4_K, m1, k),
               w2 = RandomQuant(HY4_TYPE_Q4_K, m2, k);
    const auto x = Random((std::size_t)n * k);
    const auto xr = half_round(x);
    auto* dw1 = Device(w1);
    auto* dw2 = Device(w2);
    auto* dx = Device(x);
    float* y1 = Device(std::vector<float>((std::size_t)n * m1));
    float* y2 = Device(std::vector<float>((std::size_t)n * m2));
    Expect(hy4_f16_dense_pair(HY4_TYPE_Q4_K, dw1, y1, m1, dw2, y2, m2, dx, n, k,
                              0, nullptr) == 0,
           "f16_dense_pair status");
    Near(Host(y1, (std::size_t)n * m1),
         project(Dequant(HY4_TYPE_Q4_K, w1), xr, m1, n, k, 1), 2e-3,
         "f16_dense_pair first");
    Near(Host(y2, (std::size_t)n * m2),
         project(Dequant(HY4_TYPE_Q4_K, w2), xr, m2, n, k, 1), 2e-3,
         "f16_dense_pair second");
    (void)hipFree(dw1);
    (void)hipFree(dw2);
    (void)hipFree(dx);
    (void)hipFree(y1);
    (void)hipFree(y2);
  }

  {  // Routed: gate/up (Q4_K) with SwiGLU, then down (Q6_K).
    const uint32_t experts = 5, used = 2, n = 140, f = 256, lat = 256, out = 48;
    const uint32_t slots = n * used;
    std::vector<int32_t> ids(slots);
    std::uniform_int_distribution<int> pick(
        1, experts - 2);  // expert 4 stays empty
    for (uint32_t t = 0; t < n; ++t) {
      ids[t * used] = 0;  // 140 slots: a full tile and a partial one
      ids[t * used + 1] = pick(g_rng);
    }
    const auto rw = Random(slots, 0.1f, 1.0f);
    const auto wg = RandomQuant(HY4_TYPE_Q4_K, (std::size_t)experts * f, lat);
    const auto wu = RandomQuant(HY4_TYPE_Q4_K, (std::size_t)experts * f, lat);
    const auto wd = RandomQuant(HY4_TYPE_Q6_K, (std::size_t)experts * out, f);
    const auto x = Random((std::size_t)n * lat);
    const auto xr = half_round(x);
    const auto g = Dequant(HY4_TYPE_Q4_K, wg), u = Dequant(HY4_TYPE_Q4_K, wu),
               dn = Dequant(HY4_TYPE_Q6_K, wd);
    const float limit = 0.3f;  // clamps some gate and up sums
    std::vector<double> mid((std::size_t)slots * f);
    std::size_t clamped = 0;
    for (uint32_t s = 0; s < slots; ++s)
      for (uint32_t r = 0; r < f; ++r) {
        double a = 0, b = 0;
        for (uint32_t c = 0; c < lat; ++c) {
          a += g[((std::size_t)ids[s] * f + r) * lat + c] *
               xr[(std::size_t)(s / used) * lat + c];
          b += u[((std::size_t)ids[s] * f + r) * lat + c] *
               xr[(std::size_t)(s / used) * lat + c];
        }
        clamped += a > limit || std::fabs(b) > limit;
        a = std::min<double>(a, limit);
        b = std::clamp<double>(b, -limit, limit);
        mid[(std::size_t)s * f + r] = a * Sigmoid(a) * b * rw[s];
      }
    Expect(clamped > 0 && clamped < mid.size() / 2,
           "f16_routed clamp engaged " + std::to_string(clamped));
    std::vector<int32_t> rows(slots), bounds(experts + 1),
        tiles(slots / 128 + experts);
    const uint32_t n_tiles = hy4_f16_routed_plan(
        ids.data(), slots, experts, rows.data(), bounds.data(), tiles.data());
    Expect(n_tiles == 2 + 3,
           "f16_routed_plan tiles " + std::to_string(n_tiles));
    auto* dids = Device(rows);
    auto* dbounds = Device(bounds);
    auto* dtiles = Device(tiles);
    auto* dwg = Device(wg);
    auto* dwu = Device(wu);
    auto* dwd = Device(wd);
    std::vector<uint16_t> xh(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      xh[i] = HalfBits(x[i]);
    auto* dx = Device(xh);
    auto* drw = Device(rw);
    auto* dmid = Device(std::vector<uint16_t>((std::size_t)slots * f));
    float* dy = Device(std::vector<float>((std::size_t)slots * out));
    Expect(
        hy4_f16_routed(HY4_TYPE_Q4_K, dwg, dwu, (uint64_t)f * lat / 256 * 144,
                       dx, used, dids, dbounds, dtiles, n_tiles, drw, limit,
                       dmid, f, lat, nullptr) == 0,
        "f16_routed gate/up status");
    const auto mh = Host(dmid, (std::size_t)slots * f);
    std::vector<float> got(mh.size());
    std::vector<double> mid_r(mh.size());
    for (std::size_t i = 0; i < mh.size(); ++i)
      got[i] = (float)(mid_r[i] = HalfValue(mh[i]));
    Near(got, mid, 4e-3, "f16_routed gate/up");
    Expect(hy4_f16_routed(HY4_TYPE_Q6_K, dwd, nullptr,
                          (uint64_t)out * f / 256 * 210, dmid, 1, dids, dbounds,
                          dtiles, n_tiles, nullptr, 0.0f, dy, out, f,
                          nullptr) == 0,
           "f16_routed down status");
    std::vector<double> want((std::size_t)slots * out);
    for (uint32_t s = 0; s < slots; ++s)
      for (uint32_t r = 0; r < out; ++r) {
        double a = 0;
        for (uint32_t c = 0; c < f; ++c)
          a += dn[((std::size_t)ids[s] * out + r) * f + c] *
               mid_r[(std::size_t)s * f + c];
        want[(std::size_t)s * out + r] = a;
      }
    Near(Host(dy, (std::size_t)slots * out), want, 2e-3, "f16_routed down");
    for (void* p :
         {(void*)dids, (void*)dbounds, (void*)dtiles, (void*)dwg, (void*)dwu,
          (void*)dwd, (void*)dx, (void*)drw, (void*)dmid, (void*)dy})
      (void)hipFree(p);
  }
}

// The verify pass's shared-expert matvecs against the one-token calls the
// decoder makes: bitwise equal, with experts repeated across tokens and an
// invalid (-1) router id.
void TestMoeShared() {
  Expect(hy4_mmq_init(0) == 0, "mmq init");
  const uint32_t experts = 6, used = 4, n = 5, k = 512, m = 96;
  std::vector<int32_t> ids(n * used);
  for (uint32_t t = 0; t < n; ++t) {
    std::vector<int32_t> e(experts);
    std::iota(e.begin(), e.end(), 0);
    std::shuffle(e.begin(), e.end(), g_rng);
    for (uint32_t j = 0; j < used; ++j)
      ids[t * used + j] = e[j];
  }
  ids[2 * used + 1] = -1;
  auto* dids = Device(ids);
  {  // gate/up: one launch per token before
    const auto w = RandomQuant(HY4_TYPE_Q4_K, (std::size_t)experts * m, k);
    const auto x = Random(n * k);
    auto* dw = Device(w);
    auto* dx = Device(x);
    float* a = Device(std::vector<float>(n * used * m, 7.0f));
    float* b = Device(std::vector<float>(n * used * m, 9.0f));
    bool ok = true;
    for (uint32_t t = 0; t < n; ++t)
      ok &= hy4_mmq_moe_vec_typed(HY4_TYPE_Q4_K, dw, dx + t * k,
                                  dids + t * used, a + t * used * m, m, k, 1,
                                  experts, used, nullptr) == 0;
    ok &= hy4_mmq_moe_vec_tokens_typed(HY4_TYPE_Q4_K, dw, dx, dids, b, m, k, n,
                                       experts, used, nullptr) == 0;
    Expect(ok, "moe_vec_tokens status");
    Expect(Host(a, n * used * m) == Host(b, n * used * m),
           "moe_vec_tokens equals one-token calls");
  }
  for (int type : {HY4_TYPE_Q4_K, HY4_TYPE_Q6_K}) {  // down, every slot
    const uint32_t slots = n * used;
    const auto w = RandomQuant(type, (std::size_t)experts * m, k);
    const auto x = Random(slots * k);
    auto* dw = Device(w);
    auto* dx = Device(x);
    float* a = Device(std::vector<float>(slots * m, 7.0f));
    float* b = Device(std::vector<float>(slots * m, 9.0f));
    Expect(hy4_mmq_moe_vec_typed(type, dw, dx, dids, a, m, k, slots, experts, 1,
                                 nullptr) == 0 &&
               hy4_mmq_moe_vec_slots_typed(type, dw, dx, dids, b, m, k, slots,
                                           experts, nullptr) == 0,
           "moe_vec_slots status");
    Expect(
        Host(a, slots * m) == Host(b, slots * m),
        "moe_vec_slots equals the slot matvec, type " + std::to_string(type));
  }
}

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "SKIP: no HIP device\n");
    return 77;
  }
  TestQ8_0();
  TestRmsNorm();
  TestHyperConnections();
  TestHcDecode();
  TestGemv();
  TestRope();
  TestMlaPrep();
  TestMlaAttention();
  TestMlaPrefill();
  TestRows();
  TestArgmax();
  TestQuantizeHalf();
  TestQuantizeBf16();
  TestGatherRows();
  TestElementwise();
  TestRoute();
  TestEngram();
  TestKpool();
  TestAttnSoftmax();
  TestF16Matmul();
  TestDecodeGemv();
  TestExpertGemv();
  TestSgemm();
  TestBf16();
  TestMoeShared();
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 kernels: all checks passed\n");
  return 0;
}
