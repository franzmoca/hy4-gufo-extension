/* Times the batched FP16 projection kernels (kernels/rocm/hy4_f16_mm.hip.cpp)
 * on Hy4 Flash's prefill shapes with random weights, and checks a variant
 * build against the reference outputs bit for bit.
 *
 *   hy4_dense_bench [tokens] [--dump FILE | --check FILE]
 *
 * Build from the repository root inside Gufo's nix develop:
 *   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. \
 *     tools/bench/hy4_dense_bench.hip.cpp -o hy4_dense_bench
 *
 * Shapes (m x k): attn_gate and q_b of a 64-head layer, wo, q_a + kv_a, the
 * shared expert's gate/up and down, the expert latent projections. */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../kernels/rocm/hy4_f16_mm.hip.cpp"

namespace {

struct Shape {
  const char* name;
  int type;
  uint32_t m, k;
  uint32_t io;
};

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

std::vector<uint8_t> RandomQ8(uint64_t rows, uint64_t k, std::mt19937& rng) {
  std::vector<uint8_t> w(rows * k / 32 * 34);
  std::uniform_int_distribution<int> q(-127, 127);
  std::uniform_real_distribution<float> d(0.001f, 0.02f);
  for (uint64_t b = 0; b < rows * k / 32; ++b) {
    const _Float16 s = (_Float16)d(rng);
    std::memcpy(&w[b * 34], &s, 2);
    for (int i = 0; i < 32; ++i) w[b * 34 + 2 + i] = (uint8_t)(int8_t)q(rng);
  }
  return w;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t n = 4096;
  std::string dump, check;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dump" && i + 1 < argc) dump = argv[++i];
    else if (a == "--check" && i + 1 < argc) check = argv[++i];
    else n = (uint32_t)std::atoi(argv[i]);
  }
  const Shape shapes[] = {
      {"attn_gate 16384x3072", HY4_TYPE_Q8_0, 16384, 3072, HY4_F16_X_HALF},
      {"q_b 16384x1024", HY4_TYPE_Q8_0, 16384, 1024, HY4_F16_X_HALF},
      {"wo 3072x16384", HY4_TYPE_Q8_0, 3072, 16384, HY4_F16_X_HALF},
      {"q_a 1024x3072", HY4_TYPE_Q8_0, 1024, 3072, HY4_F16_X_HALF},
      {"shexp gate 1536x3072", HY4_TYPE_Q8_0, 1536, 3072, HY4_F16_X_HALF},
      {"shexp down 3072x1536 f32", HY4_TYPE_Q8_0, 3072, 1536, 0},
      {"routed_down 1536x3072", HY4_TYPE_Q8_0, 1536, 3072, HY4_F16_X_HALF | HY4_F16_OUT_HALF},
      {"routed_up 3072x1536 f32", HY4_TYPE_Q8_0, 3072, 1536, 0},
  };
  std::mt19937 rng(7);
  FILE* out = dump.empty() ? nullptr : std::fopen(dump.c_str(), "wb");
  FILE* ref = check.empty() ? nullptr : std::fopen(check.c_str(), "rb");
  int mismatched = 0;
  double total_ms = 0;
  for (const Shape& s : shapes) {
    const auto w = RandomQ8(s.m, s.k, rng);
    const bool xh = s.io & HY4_F16_X_HALF, oh = s.io & HY4_F16_OUT_HALF;
    std::vector<uint8_t> x((uint64_t)n * s.k * (xh ? 2 : 4));
    std::uniform_real_distribution<float> v(-2.0f, 2.0f);
    for (uint64_t i = 0; i < (uint64_t)n * s.k; ++i) {
      const float f = v(rng);
      if (xh) {
        const _Float16 h = (_Float16)f;
        std::memcpy(&x[i * 2], &h, 2);
      } else {
        std::memcpy(&x[i * 4], &f, 4);
      }
    }
    const uint64_t out_bytes = (uint64_t)n * s.m * (oh ? 2 : 4);
    void *dw, *dx, *dy;
    Check(hipMalloc(&dw, w.size()), "malloc");
    Check(hipMalloc(&dx, x.size()), "malloc");
    Check(hipMalloc(&dy, out_bytes), "malloc");
    Check(hipMemcpy(dw, w.data(), w.size(), hipMemcpyHostToDevice), "copy");
    Check(hipMemcpy(dx, x.data(), x.size(), hipMemcpyHostToDevice), "copy");
    for (int i = 0; i < 3; ++i)
      if (hy4_f16_dense(s.type, dw, dx, dy, s.m, n, s.k, 1, s.io, nullptr)) return 1;
    hipEvent_t a, b;
    Check(hipEventCreate(&a), "event");
    Check(hipEventCreate(&b), "event");
    const int reps = 10;
    Check(hipEventRecord(a), "record");
    for (int i = 0; i < reps; ++i) (void)hy4_f16_dense(s.type, dw, dx, dy, s.m, n, s.k, 1, s.io, nullptr);
    Check(hipEventRecord(b), "record");
    Check(hipEventSynchronize(b), "sync");
    float ms = 0;
    Check(hipEventElapsedTime(&ms, a, b), "elapsed");
    ms /= reps;
    total_ms += ms;
    std::vector<uint8_t> y(out_bytes);
    Check(hipMemcpy(y.data(), dy, out_bytes, hipMemcpyDeviceToHost), "copy");
    if (out) std::fwrite(y.data(), 1, y.size(), out);
    if (ref) {
      std::vector<uint8_t> r(out_bytes);
      if (std::fread(r.data(), 1, r.size(), ref) != r.size() || r != y) ++mismatched, std::printf("  MISMATCH ");
    }
    std::printf("%-26s %8.3f ms %6.1f TFLOPS\n", s.name, ms, 2.0 * n * s.m * s.k / ms / 1e9);
    (void)hipFree(dw), (void)hipFree(dx), (void)hipFree(dy);
  }
  std::printf("total %.3f ms at %u tokens%s\n", total_ms, n,
              ref ? (mismatched ? " (outputs differ)" : " (outputs identical)") : "");
  if (out) std::fclose(out);
  if (ref) std::fclose(ref);
  return mismatched ? 2 : 0;
}
