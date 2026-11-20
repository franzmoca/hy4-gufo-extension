/* Times MlaPrefill (kernels/rocm/hy4_kernels.hip.cpp) on Hy4 Flash's prefill
 * shapes with random operands, and checks a variant build against the
 * reference outputs bit for bit.
 *
 *   hy4_mla_bench [tokens] [--dump FILE | --check FILE]
 *
 * Build from the repository root inside Gufo's nix develop:
 *   hipcc -O3 --offload-arch=gfx1151 -std=c++20 -I. \
 *     tools/bench/hy4_mla_bench.hip.cpp -o hy4_mla_bench
 *
 * Cases: a 64-head SWA layer (window 512) and a 32-head indexer layer
 * (2,048 selected keys, kpool 4), both over one pass from position 0. */
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "../../kernels/rocm/hy4_kernels.hip.cpp"

namespace {

template<typename T>
T* Upload(const std::vector<T>& v) {
  T* d = nullptr;
  if (hipMalloc(&d, v.size() * sizeof(T)) != hipSuccess)
    std::exit(1);
  (void)hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

std::vector<uint16_t> RandomHalf(std::size_t n, std::mt19937& g, float a) {
  std::uniform_real_distribution<float> u(-a, a);
  std::vector<uint16_t> v(n);
  for (auto& x : v) {
    const _Float16 h = (_Float16)u(g);
    std::memcpy(&x, &h, 2);
  }
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  uint32_t n = 3712;
  std::string dump, check;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dump" && i + 1 < argc)
      dump = argv[++i];
    else if (a == "--check" && i + 1 < argc)
      check = argv[++i];
    else
      n = (uint32_t)std::atoi(argv[i]);
  }
  std::mt19937 g(7);
  const uint32_t kSel = 2048, kPool = 4;
  std::vector<int32_t> pos(n);
  for (uint32_t t = 0; t < n; ++t)
    pos[t] = (int32_t)t;
  // Selections: 512 random kpool groups at or before the token, sorted.
  std::vector<int32_t> sel((std::size_t)n * kSel, -1);
  for (uint32_t t = 0; t < n; ++t) {
    const uint32_t groups = t / kPool + 1;
    std::vector<uint32_t> gs(groups);
    for (uint32_t i = 0; i < groups; ++i)
      gs[i] = i;
    if (groups > kSel / kPool) {
      std::shuffle(gs.begin(), gs.end(), g);
      gs.resize(kSel / kPool);
      std::sort(gs.begin(), gs.end());
    }
    uint32_t j = 0;
    for (uint32_t grp : gs)
      for (uint32_t k = 0; k < kPool && j < kSel; ++k)
        sel[(std::size_t)t * kSel + j++] = (int32_t)(grp * kPool + k);
  }
  struct Case {
    const char* name;
    uint32_t heads, window;
    bool select;
  };
  const Case cases[] = {{"swa 64 heads", 64, 512, false},
                        {"indexer 32 heads", 32, 0, true}};
  std::vector<uint16_t> all;
  double total = 0;
  for (const Case& c : cases) {
    const auto q = RandomHalf((std::size_t)n * c.heads * kMlaDk, g, 1.0f);
    const auto keys = RandomHalf((std::size_t)n * kMlaDk, g, 1.0f);
    std::vector<float> sinks(c.heads);
    for (auto& s : sinks)
      s = std::uniform_real_distribution<float>(-1, 1)(g);
    auto* dq = Upload(q);
    auto* dk = Upload(keys);
    auto* dpos = Upload(pos);
    auto* dsel = c.select ? Upload(sel) : nullptr;
    auto* dsinks = Upload(sinks);
    uint16_t* dout = nullptr;
    const std::size_t out_n = (std::size_t)n * c.heads * kMlaDv;
    (void)hipMalloc(&dout, out_n * 2);
    const auto run = [&] {
      return hy4_mla_prefill(dq, dk, n, 0, dpos, c.window, dsel,
                             c.select ? kSel : 0, dsinks, 0.0625f, dout, n,
                             c.heads, kMlaDk, kMlaDv, nullptr);
    };
    if (run())
      return 1;
    (void)hipDeviceSynchronize();
    hipEvent_t e0, e1;
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
    const int reps = 5;
    (void)hipEventRecord(e0);
    for (int r = 0; r < reps; ++r)
      run();
    (void)hipEventRecord(e1);
    (void)hipEventSynchronize(e1);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, e0, e1);
    ms /= reps;
    total += ms;
    std::printf("%-20s %8.3f ms\n", c.name, ms);
    std::vector<uint16_t> out(out_n);
    (void)hipMemcpy(out.data(), dout, out_n * 2, hipMemcpyDeviceToHost);
    all.insert(all.end(), out.begin(), out.end());
    (void)hipFree(dq);
    (void)hipFree(dk);
    (void)hipFree(dpos);
    if (dsel)
      (void)hipFree(dsel);
    (void)hipFree(dsinks);
    (void)hipFree(dout);
  }
  std::printf("total %.3f ms at %u tokens", total, n);
  if (!dump.empty()) {
    FILE* f = std::fopen(dump.c_str(), "wb");
    std::fwrite(all.data(), 2, all.size(), f);
    std::fclose(f);
  }
  if (!check.empty()) {
    std::vector<uint16_t> ref(all.size());
    FILE* f = std::fopen(check.c_str(), "rb");
    const bool ok = f &&
                    std::fread(ref.data(), 2, ref.size(), f) == ref.size() &&
                    ref == all;
    if (f)
      std::fclose(f);
    std::printf(ok ? " (outputs identical)" : " (OUTPUTS DIFFER)");
  }
  std::printf("\n");
  return 0;
}
