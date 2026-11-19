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

// Binary16 bits of v (round to nearest even; normal range only, which the
// test values keep to) and back.
uint16_t HalfBits(float v) {
  uint32_t x;
  std::memcpy(&x, &v, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  const uint32_t mant = x & 0x7fffffu;
  if (v == 0.0f)
    return (uint16_t)sign;
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
  }
}

void TestHyperConnections() {
  const uint32_t n = 3, hc = 4, dim = 96, flat = hc * dim;
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
  TestRows();
  TestElementwise();
  TestRoute();
  TestEngram();
  TestKpool();
  TestAttnSoftmax();
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 kernels: all checks passed\n");
  return 0;
}
