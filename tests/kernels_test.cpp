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
#include <string>
#include <vector>

#include "kernels/rocm/hy4_kernels.h"

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
    }
  }
  Expect(hy4_gemv_float(HY4_TYPE_Q8_0, nullptr, nullptr, nullptr, 1, 8, 1,
                        nullptr) != 0,
         "gemv_float rejects other types");

  // Grouped Q8_0 with padded token and group strides.
  const uint32_t groups = 3, mk = 64, kk = 96, n = 10;
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
  Near(Host(y, n * y_tok), want, 1e-5, "gemv_q8_0_grouped");

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

void TestMlaAttention() {
  const uint32_t heads = 4, dk = 96, dv = 64, ring = 40;
  const float scale = 0.125f;
  const auto cache = Random(ring * dk, -2, 2);
  const auto sinks = Random(heads, -1, 1);
  const std::vector<int32_t> pos{5, 39, 57};
  const uint32_t n = (uint32_t)pos.size();
  const auto q = Random(n * heads * dk, -2, 2);
  auto* dq = Device(q);
  auto* dc = Device(cache);
  auto* dp = Device(pos);
  auto* ds = Device(sinks);
  for (uint32_t window : {0u, 16u}) {
    for (bool sink : {true, false}) {
      // Window 0 has no lower bound: its spans stay inside the ring.
      const std::vector<int32_t> use =
          window ? pos : std::vector<int32_t>{5, 30, 39};
      std::vector<std::vector<int32_t>> keys(n);
      for (uint32_t t = 0; t < n; ++t)
        for (int32_t p = window ? std::max(0, use[t] - (int32_t)window + 1) : 0;
             p <= use[t]; ++p)
          keys[t].push_back(p);
      float* out = Device(std::vector<float>(n * heads * dv));
      Expect(hy4_mla_attention(dq, dc, ring, Device(use), window, nullptr, 0,
                               sink ? ds : nullptr, scale, out, n, heads, dk,
                               dv, nullptr) == 0,
             "mla_attention status");
      Near(Host(out, n * heads * dv),
           MlaReference(q, cache, ring, keys, sink ? sinks.data() : nullptr,
                        scale, heads, dk, dv),
           1e-5,
           "mla_attention window " + std::to_string(window) +
               (sink ? " sinks" : ""));
    }
  }
  // Selection: listed positions (wrapping the ring), -1 slots skipped.
  const uint32_t n_sel = 8;
  const std::vector<int32_t> sel{0,  3,  -1, 5,  -1, -1, -1, -1,  //
                                 39, 12, 20, 21, 22, 23, -1, 38,  //
                                 57, 56, -1, 41, 50, -1, -1, -1};
  std::vector<std::vector<int32_t>> keys(n);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < n_sel; ++i)
      if (sel[t * n_sel + i] >= 0)
        keys[t].push_back(sel[t * n_sel + i]);
  float* out = Device(std::vector<float>(n * heads * dv));
  Expect(hy4_mla_attention(dq, dc, ring, dp, 0, Device(sel), n_sel, ds, scale,
                           out, n, heads, dk, dv, nullptr) == 0,
         "mla_attention selection status");
  Near(Host(out, n * heads * dv),
       MlaReference(q, cache, ring, keys, sinks.data(), scale, heads, dk, dv),
       1e-5, "mla_attention selection");
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
  // Keys as hy4_gather_rows_half writes them: binary16 rows from position 0.
  auto* dcf = Device(cache);
  auto* dc = Device(std::vector<uint16_t>(cache.size()));
  Expect(hy4_gather_rows_half(dcf, ring, 0, ring, dk, dc, nullptr) == 0,
         "gather_rows_half status");
  {
    std::vector<uint16_t> want(cache.size());
    for (std::size_t i = 0; i < cache.size(); ++i)
      want[i] = HalfBits(cache[i]);
    Expect(Host(dc, cache.size()) == want, "gather_rows_half");
    // Wrapping the ring: positions ring - 2 .. ring + 1.
    auto* wrap = Device(std::vector<uint16_t>(4 * dk));
    Expect(hy4_gather_rows_half(dcf, ring, (int32_t)ring - 2, 4, dk, wrap,
                                nullptr) == 0,
           "gather_rows_half wrap status");
    const auto got = Host(wrap, 4 * dk);
    bool same = true;
    for (uint32_t r = 0; r < 4; ++r)
      for (uint32_t i = 0; i < dk; ++i)
        same =
            same && got[r * dk + i] == want[((ring - 2 + r) % ring) * dk + i];
    Expect(same, "gather_rows_half wrap");
  }
  auto* dp = Device(pos);
  auto* ds = Device(sinks);
  const auto run = [&](const uint16_t* keys, int32_t lo, uint32_t window,
                       const std::vector<int32_t>* sel, uint32_t n_sel,
                       bool sink,
                       const std::vector<std::vector<int32_t>>& want_keys,
                       const std::string& what) {
    auto* out = Device(std::vector<uint16_t>(n * heads * dv));
    Expect(hy4_mla_prefill(
               dq, keys, lo, dp, window, sel ? Device(*sel) : nullptr, n_sel,
               sink ? ds : nullptr, scale, out, n, heads, dk, dv, nullptr) == 0,
           what + " status");
    const auto bits = Host(out, n * heads * dv);
    std::vector<float> got(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i)
      got[i] = (float)HalfValue(bits[i]);
    Near(got,
         MlaReference(q, cache, ring, want_keys, sink ? sinks.data() : nullptr,
                      scale, heads, dk, dv),
         4e-3, what);
  };
  for (bool sink : {true, false}) {
    std::vector<std::vector<int32_t>> keys(n);
    for (uint32_t t = 0; t < n; ++t)
      for (int32_t k = 0; k <= pos[t]; ++k)
        keys[t].push_back(k);
    run(dc, 0, 0, nullptr, 0, sink, keys,
        std::string("mla_prefill causal") + (sink ? " sinks" : ""));
  }
  {
    // Window 64 over keys gathered from position 20 (positions below 20 are
    // never read: token 0 sees none and keeps only the sink).
    const int32_t lo = 20;
    const uint32_t window = 64;
    std::vector<std::vector<int32_t>> keys(n);
    for (uint32_t t = 0; t < n; ++t)
      for (int32_t k = std::max(lo, pos[t] - (int32_t)window + 1); k <= pos[t];
           ++k)
        keys[t].push_back(k);
    run(dc + std::size_t{lo} * dk, lo, window, nullptr, 0, true, keys,
        "mla_prefill window");
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
        if (k <= pos[t])
          keys[t].push_back(k);  // a future key is skipped
      }
    run(dc, 0, 0, &sel, n_sel, true, keys, "mla_prefill selection");
  }
  Expect(hy4_mla_prefill(dq, dc, 0, dp, 0, nullptr, 0, ds, scale, nullptr, n,
                         24, dk, dv, nullptr) != 0,
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
  const auto stored = Host(ring_rows, ring * dim);

  const uint32_t count = 5;
  float* dst = Device(std::vector<float>(count * dim));
  Expect(hy4_gather_rows(ring_rows, ring, 3, count, dim, dst, nullptr) == 0,
         "gather_rows status");
  std::vector<double> rows(count * dim);
  for (uint32_t r = 0; r < count; ++r)
    for (uint32_t c = 0; c < dim; ++c)
      rows[r * dim + c] = stored[((3 + r) % ring) * dim + c];
  Near(Host(dst, count * dim), rows, 0, "gather_rows");

  const std::vector<int32_t> idx{7, -1, 2, 9, 5};
  const int32_t limit = 7;
  const uint32_t mask_rows = 2, n_keys = 8;
  float* gathered = Device(std::vector<float>(idx.size() * dim));
  float* mask = Device(std::vector<float>(mask_rows * n_keys, 3.0f));
  Expect(hy4_gather_indexed(ring_rows, ring, Device(idx), (uint32_t)idx.size(),
                            limit, dim, gathered, mask, mask_rows, n_keys,
                            nullptr) == 0,
         "gather_indexed status");
  std::vector<double> g(idx.size() * dim, 0.0), m(mask_rows * n_keys, 0.0);
  for (uint32_t r = 0; r < idx.size(); ++r) {
    const bool ok = idx[r] >= 0 && idx[r] <= limit;
    for (uint32_t c = 0; c < dim; ++c)
      g[r * dim + c] = ok ? stored[(idx[r] % ring) * dim + c] : 0.0;
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
  {
    // Ties go to the lowest index; an all -inf row gives 0.
    std::vector<float> x = Random(4 * dim);
    x[0 * dim + 10] = x[0 * dim + 70] = 5.0f;
    x[1 * dim + 99] = 4.0f;
    std::fill(x.begin() + 2 * dim, x.begin() + 3 * dim, -INFINITY);
    int32_t* out = Device(std::vector<int32_t>(4));
    Expect(hy4_argmax(Device(x), 4, dim, out, nullptr) == 0, "argmax status");
    const auto got = Host(out, 4);
    const int32_t last =
        (int32_t)(std::max_element(x.begin() + 3 * dim, x.end()) -
                  (x.begin() + 3 * dim));
    Expect(got == std::vector<int32_t>{10, 99, 0, last}, "argmax");
  }
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
  float* mask = Device(std::vector<float>(n * n_keys, 1.0f));
  Expect(hy4_selection_mask(Device(sel), 3, mask, n, n_keys, nullptr) == 0,
         "selection_mask status");
  std::vector<double> want_mask(n * n_keys, -INFINITY);
  for (uint32_t t = 0; t < n; ++t)
    for (uint32_t i = 0; i < 3; ++i)
      if (sel[t * 3 + i] >= 0)
        want_mask[t * n_keys + sel[t * 3 + i]] = 0.0;
  const auto mask_host = Host(mask, n * n_keys);
  Near(mask_host, want_mask, 0, "selection_mask");

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

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::fprintf(stderr, "SKIP: no HIP device\n");
    return 77;
  }
  TestQ8_0();
  TestRmsNorm();
  TestHyperConnections();
  TestGemv();
  TestRope();
  TestMlaAttention();
  TestMlaPrefill();
  TestRows();
  TestElementwise();
  TestRoute();
  TestEngram();
  TestKpool();
  TestAttnSoftmax();
  TestF16Matmul();
  TestSgemm();
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 kernels: all checks passed\n");
  return 0;
}
