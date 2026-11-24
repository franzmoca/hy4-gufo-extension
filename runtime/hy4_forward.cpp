// Hy4 Flash evaluation: one chunk of up to Model::MaxChunk() tokens through
// the 42-layer iHC stack, following llama.cpp's hy_v4_flash graph.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <memory>
#include <string>
#include <vector>

#include "kernels/rocm/hy4_kernels.h"
#include "kernels/rocm/mmq/hy4_mmq.h"
#include "runtime/hy4_engram.hpp"
#include "runtime/hy4_internal.hpp"
#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

namespace {

// The reference Engram norms leave eps unset: the FP32 machine epsilon.
constexpr float kEngramEps = 1.1920929e-07f;

bool Fail(std::string* error, const std::string& what) {
  if (error) *error = what;
  return false;
}

#define HY4_CHECK(expr)                                                     \
  do {                                                                      \
    if ((expr) != 0) return Fail(error, std::string("hy4: failed: ") + #expr); \
  } while (0)

template <typename T>
bool DeviceAlloc(T** p, std::size_t count, std::vector<void*>* owned) {
  void* v = nullptr;
  if (gufo::hip::AllocateDevice(&v, std::max<std::size_t>(count, 1) * sizeof(T)) != hipSuccess) return false;
  // hipMemset runs on the null stream, which the engine's non-blocking stream
  // does not wait for: finish it before any kernel can touch the buffer.
  if (hipMemset(v, 0, std::max<std::size_t>(count, 1) * sizeof(T)) != hipSuccess ||
      hipDeviceSynchronize() != hipSuccess) {
    (void)hipFree(v);
    return false;
  }
  owned->push_back(v);
  *p = static_cast<T*>(v);
  return true;
}

template <typename T>
bool HostAlloc(T** p, std::size_t count, std::vector<void*>* owned) {
  void* v = nullptr;
  if (hipHostMalloc(&v, std::max<std::size_t>(count, 1) * sizeof(T), hipHostMallocDefault) != hipSuccess) {
    return false;
  }
  owned->push_back(v);
  *p = static_cast<T*>(v);
  return true;
}

// Bytes of one expert's [dims[1]][dims[0]] matrix of a 3-D expert tensor.
std::uint64_t ExpertBytes(const Tensor& w) {
  const std::uint64_t block = w.type == HY4_TYPE_Q8_0   ? 34
                              : w.type == HY4_TYPE_Q4_K ? 144
                                                        : 210;
  const std::uint64_t values = w.type == HY4_TYPE_Q8_0 ? 32 : 256;
  return w.dims[0] * w.dims[1] / values * block;
}

}  // namespace

static_assert(Model::kVecMax <= HY4_GEMV_MAX_TOKENS,
              "decode passes fit hy4_gemv");

bool MatVec(const Tensor& w, const float* x, float* y, std::uint32_t m,
            std::uint32_t k, std::uint32_t n, const Exec& ex,
            std::string* error, const std::uint16_t* xh) {
  if (w.dims.size() < 2 || w.dims[0] != k || w.dims[1] != m) {
    return Fail(error, "hy4: matvec shape mismatch (" + std::to_string(m) + "x" + std::to_string(k) + ")");
  }
  hipStream_t stream = ex.stream;
  const bool batched = n > Model::kVecMax || ex.prompt;
  switch (w.type) {
    case HY4_TYPE_F32:
      if (batched) {
        // Row-major y[n][m] = x[n][k] W^T; each row is summed in the same
        // order at any n, so a prompt's results do not depend on its passes.
        HY4_CHECK(hy4_sgemm_nt(x, w.f32(), y, n, m, k, stream));
        return true;
      }
      [[fallthrough]];
    case HY4_TYPE_F16:
      HY4_CHECK(hy4_gemv_float(w.type, w.data, x, y, m, k, n, stream));
      return true;
    case HY4_TYPE_BF16:
      if (batched)
        HY4_CHECK(hy4_sgemm_nt_bf16(
            x, static_cast<const std::uint16_t*>(w.data), y, n, m, k, stream));
      else
        HY4_CHECK(hy4_gemv_float(w.type, w.data, x, y, m, k, n, stream));
      return true;
    case HY4_TYPE_Q8_0:
    case HY4_TYPE_Q4_K:
    case HY4_TYPE_Q6_K:
      if (k % 256 == 0) {
        if (batched && m % 16 == 0) {
          HY4_CHECK(
              xh ? hy4_f16_dense(w.type, w.data, xh, y, m, n, k, 1,
                                 HY4_F16_X_HALF, stream)
                 : hy4_f16_dense(w.type, w.data, x, y, m, n, k, 1, 0, stream));
          return true;
        }
        if (batched) {
          HY4_CHECK(hy4_mmq_dense_typed(w.type, w.data, x, y, static_cast<int>(m), static_cast<int>(n),
                                        static_cast<int>(k), stream));
          return true;
        }
        // FP32 activations; each token sums as a one-token call does.
        HY4_CHECK(
            hy4_gemv(w.type, w.data, x, y, m, k, n, ex.gemv_scratch, stream));
        return true;
      }
      if (w.type == HY4_TYPE_Q8_0) {
        HY4_CHECK(hy4_gemv_q8_0_grouped(w.data, x, y, m, k, 1, n, k, 0, m, 0, stream));
        return true;
      }
      break;
    default:
      break;
  }
  return Fail(error, "hy4: unsupported matvec type " + std::to_string(w.type));
}

bool DenseF16(const Tensor& w, const void* x, void* y, std::uint32_t m,
              std::uint32_t k, std::uint32_t n, std::uint32_t io,
              const Exec& ex, std::string* error) {
  const bool quantized = w.type == HY4_TYPE_Q8_0 || w.type == HY4_TYPE_Q4_K ||
                         w.type == HY4_TYPE_Q6_K;
  if (!quantized || k % 256 != 0 || m % 16 != 0 || w.dims.size() < 2 ||
      w.dims[0] != k || w.dims[1] != m)
    return Fail(error, "hy4: no FP16 kernel for a " + std::to_string(m) + "x" +
                           std::to_string(k) + " projection of type " +
                           std::to_string(w.type));
  HY4_CHECK(hy4_f16_dense(w.type, w.data, x, y, m, n, k, 1, io, ex.stream));
  return true;
}

bool MatVecPair(const Tensor& w1, float* y1, std::uint32_t m1, const Tensor& w2,
                float* y2, std::uint32_t m2, const float* x, std::uint32_t k,
                std::uint32_t n, const Exec& ex, std::string* error,
                const std::uint16_t* xh) {
  const bool quantized = w1.type == HY4_TYPE_Q8_0 || w1.type == HY4_TYPE_Q4_K ||
                         w1.type == HY4_TYPE_Q6_K;
  if ((n > Model::kVecMax || ex.prompt) && quantized && w2.type == w1.type &&
      k % 256 == 0 && m1 % 16 == 0 && m2 % 16 == 0 && w1.dims.size() >= 2 &&
      w1.dims[0] == k && w1.dims[1] == m1 && w2.dims.size() >= 2 &&
      w2.dims[0] == k && w2.dims[1] == m2) {
    HY4_CHECK(hy4_f16_dense_pair(w1.type, w1.data, y1, m1, w2.data, y2, m2,
                                 xh ? static_cast<const void*>(xh) : x, n, k,
                                 xh ? HY4_F16_X_HALF : 0, ex.stream));
    return true;
  }
  if (n <= Model::kVecMax && !ex.prompt && quantized && w2.type == w1.type &&
      k % 256 == 0 && w1.dims.size() >= 2 && w1.dims[0] == k &&
      w1.dims[1] == m1 && w2.dims.size() >= 2 && w2.dims[0] == k &&
      w2.dims[1] == m2) {
    HY4_CHECK(hy4_gemv_pair(w1.type, w1.data, y1, m1, w2.data, y2, m2, x, k, n,
                            ex.gemv_scratch, ex.stream));
    return true;
  }
  return MatVec(w1, x, y1, m1, k, n, ex, error, xh) &&
         MatVec(w2, x, y2, m2, k, n, ex, error, xh);
}

namespace {

// A decode pass's quantized matvec with k % 256 == 0: hy4_gemv's shapes.
bool DecodeGemv(const Tensor& w, std::uint32_t m, std::uint32_t k,
                std::uint32_t n, const Exec& ex) {
  return n <= Model::kVecMax && !ex.prompt && k % 256 == 0 &&
         (w.type == HY4_TYPE_Q8_0 || w.type == HY4_TYPE_Q4_K ||
          w.type == HY4_TYPE_Q6_K) &&
         w.dims.size() >= 2 && w.dims[0] == k && w.dims[1] == m;
}

}  // namespace

bool SwigluVec(const Tensor& gate, const Tensor& up, float* a, float* b,
               float* y, std::uint32_t f, const float* x, std::uint32_t k,
               std::uint32_t n, const Exec& ex, std::string* error,
               const std::uint16_t* xh) {
  if (DecodeGemv(gate, f, k, n, ex) && up.type == gate.type &&
      DecodeGemv(up, f, k, n, ex)) {
    HY4_CHECK(hy4_gemv_swiglu(gate.type, gate.data, up.data, x, y, f, k, 0.0f,
                              n, ex.gemv_scratch, ex.stream));
    return true;
  }
  if (!MatVecPair(gate, a, f, up, b, f, x, k, n, ex, error, xh))
    return false;
  HY4_CHECK(hy4_swiglu(a, b, nullptr, y, n, f, 0.0f, ex.stream));
  return true;
}

bool MatVecGate(const Tensor& w, const float* x, float* gate, float* y,
                std::uint32_t m, std::uint32_t k, std::uint32_t n,
                const Exec& ex, std::string* error) {
  if (DecodeGemv(w, m, k, n, ex)) {
    HY4_CHECK(hy4_gemv_gate(w.type, w.data, x, y, m, k, n, ex.gemv_scratch,
                            ex.stream));
    return true;
  }
  if (!MatVec(w, x, gate, m, k, n, ex, error))
    return false;
  HY4_CHECK(hy4_sigmoid_mul(y, gate, std::uint64_t{n} * m, ex.stream));
  return true;
}

bool MatVecAdd(const Tensor& w, const float* x, const float* add, float* y,
               std::uint32_t m, std::uint32_t k, std::uint32_t n,
               const Exec& ex, std::string* error) {
  if (DecodeGemv(w, m, k, n, ex)) {
    HY4_CHECK(hy4_gemv_add(w.type, w.data, x, add, y, m, k, n, ex.gemv_scratch,
                           ex.stream));
    return true;
  }
  if (!MatVec(w, x, y, m, k, n, ex, error))
    return false;
  HY4_CHECK(hy4_add(y, add, y, std::uint64_t{n} * m, ex.stream));
  return true;
}

bool Model::AllocScratchRows(Scratch& s, std::uint32_t n,
                             std::uint32_t resident, mem::ScratchArena* arena,
                             std::size_t* measure) {
  if (resident == 0)
    resident = n;
  const Config& c = config_;
  const std::uint32_t hmax = *std::max_element(c.n_head.begin(), c.n_head.end());
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t modules = static_cast<std::uint32_t>(c.engram_layers.size());
  const std::uint32_t cols = c.engram_layers.empty() ? 0 : c.EngramCols();
  auto ffn = [&](std::size_t rows) {
    return std::max<std::size_t>(
        std::size_t{c.n_ff_dense} * rows,
        std::size_t{c.n_expert_used} * rows * std::max(c.n_ff_exp, c.n_latent));
  };
  auto plan = [&](std::size_t rows) {
    const std::size_t slots = std::size_t{c.n_expert_used} * rows;
    return std::max<std::size_t>(
        slots + c.n_expert + 1 + slots / 128 + c.n_expert,
        hy4_expert_plan_ints());
  };
  auto* owned = &allocations_;
  // A device buffer of count(n) elements. With an arena, its first
  // count(resident) elements stay backed and the rest only while
  // AcquireWidePasses maps them; `measure` sums the arena bytes instead.
  // (The head runs at most kVecMax rows at a time: Pass.)
  auto Dev = [&](auto** p, auto count) {
    using T = std::remove_pointer_t<std::remove_reference_t<decltype(*p)>>;
    const std::size_t wide =
        std::max<std::size_t>(count(std::size_t{n}), 1) * sizeof(T);
    const std::size_t narrow =
        std::max<std::size_t>(count(std::size_t{resident}), 1) * sizeof(T);
    if (measure) {
      *measure += (wide + mem::ScratchArena::kPage - 1) /
                  mem::ScratchArena::kPage * mem::ScratchArena::kPage;
      return true;
    }
    if (!arena)
      return DeviceAlloc(p, wide / sizeof(T), owned);
    void* v = arena->Take(wide, narrow, nullptr);
    if (!v || hipMemset(v, 0, narrow) != hipSuccess ||
        hipDeviceSynchronize() != hipSuccess)
      return false;
    *p = static_cast<T*>(v);
    return true;
  };
  auto HostAlloc = [&](auto** p, std::size_t count, std::vector<void*>* list) {
    return measure != nullptr || hy4_flash::HostAlloc(p, count, list);
  };
  const bool ok =
      Dev(&s.tokens, [&](std::size_t n) { return n; }) &&
      Dev(&s.pos, [&](std::size_t n) { return n; }) &&
      Dev(&s.groups, [&](std::size_t n) { return n; }) &&
      Dev(&s.group_pos, [&](std::size_t n) { return n; }) &&
      Dev(&s.group_hw, [&](std::size_t n) { return 2 * n; }) &&
      Dev(&s.n_pool, [&](std::size_t n) { return n; }) &&
      Dev(&s.rope, [&](std::size_t n) { return n; }) &&
      Dev(&s.rope_hw, [&](std::size_t n) { return 2 * n; }) &&
      HostAlloc(&s.engram_host, std::size_t{modules} * n * cols * c.engram_dim,
                owned) &&
      HostAlloc(&s.engram_valid_host, std::size_t{modules} * n * cols, owned) &&
      HostAlloc(&s.engram_any_host, std::size_t{modules} * n, owned) &&
      Dev(&s.engram_rows,
          [&](std::size_t n) {
            return std::size_t{modules} * n * cols * c.engram_dim;
          }) &&
      Dev(&s.engram_valid,
          [&](std::size_t n) { return std::size_t{modules} * n * cols; }) &&
      Dev(&s.engram_any,
          [&](std::size_t n) { return std::size_t{modules} * n; }) &&
      Dev(&s.engram_normed,
          [&](std::size_t n) {
            return std::size_t{n} * cols * c.engram_dim;
          }) &&
      Dev(&s.engram_kv,
          [&](std::size_t n) {
            return std::size_t{n} * (c.hc + 1) * c.n_embd;
          }) &&
      Dev(&s.h,
          [&](std::size_t n) { return std::size_t{n} * c.hc * c.n_embd; }) &&
      Dev(&s.post, [&](std::size_t n) { return std::size_t{n} * c.hc; }) &&
      Dev(&s.x, [&](std::size_t n) { return std::size_t{n} * c.n_embd; }) &&
      Dev(&s.cur, [&](std::size_t n) { return std::size_t{n} * c.n_embd; }) &&
      Dev(&s.cur_half,
          [&](std::size_t n) { return std::size_t{n} * c.n_embd; }) &&
      Dev(&s.y, [&](std::size_t n) { return std::size_t{n} * c.n_embd; }) &&
      Dev(&s.qr, [&](std::size_t n) { return std::size_t{n} * c.q_lora; }) &&
      Dev(&s.qr_half,
          [&](std::size_t n) { return std::size_t{n} * c.q_lora; }) &&
      Dev(&s.q,
          [&](std::size_t n) { return std::size_t{n} * hmax * c.head_k; }) &&
      Dev(&s.kva, [&](std::size_t n) { return std::size_t{n} * dk; }) &&
      Dev(&s.qfull,
          [&](std::size_t n) { return std::size_t{n} * hmax * dk; }) &&
      Dev(&s.att,
          [&](std::size_t n) { return std::size_t{n} * hmax * c.kv_lora; }) &&
      Dev(&s.attv,
          [&](std::size_t n) { return std::size_t{n} * hmax * c.head_v; }) &&
      Dev(&s.gate,
          [&](std::size_t n) { return std::size_t{n} * hmax * c.head_v; }) &&
      Dev(&s.ik, [&](std::size_t n) { return std::size_t{n} * c.idx_dim; }) &&
      Dev(&s.ivg,
          [&](std::size_t n) { return std::size_t{n} * 2 * c.idx_dim; }) &&
      Dev(&s.iq,
          [&](std::size_t n) {
            return std::size_t{n} * c.idx_heads * c.idx_dim;
          }) &&
      Dev(&s.iw, [&](std::size_t n) { return std::size_t{n} * c.idx_heads; }) &&
      Dev(&s.ffn_a, [&](std::size_t n) { return ffn(n); }) &&
      Dev(&s.ffn_b, [&](std::size_t n) { return ffn(n); }) &&
      Dev(&s.ffn_c, [&](std::size_t n) { return ffn(n); }) &&
      Dev(&s.latent,
          [&](std::size_t n) { return std::size_t{n} * c.n_latent; }) &&
      Dev(&s.logits_r,
          [&](std::size_t n) { return std::size_t{n} * c.n_expert; }) &&
      Dev(&s.ids,
          [&](std::size_t n) { return std::size_t{n} * c.n_expert_used; }) &&
      HostAlloc(&s.ids_host, std::size_t{n} * c.n_expert_used, owned) &&
      Dev(&s.route_plan, [&](std::size_t n) { return plan(n); }) &&
      HostAlloc(&s.route_plan_host, plan(n), owned) &&
      Dev(&s.route_w,
          [&](std::size_t n) { return std::size_t{n} * c.n_expert_used; }) &&
      Dev(&s.moe, [&](std::size_t n) { return std::size_t{n} * c.n_latent; }) &&
      Dev(&s.shared,
          [&](std::size_t n) { return std::size_t{n} * c.n_embd; }) &&
      Dev(&s.logits,
          [&](std::size_t n) {
            return std::min<std::size_t>(n, kVecMax) * c.n_vocab;
          }) &&
      HostAlloc(&s.host_h, std::size_t{n} * c.hc * c.n_embd, owned);
  s.attn_work_floats = hy4_mla_attention_work(kVecMax, hmax, c.idx_top_k);
  return ok &&
         Dev(&s.attn_work, [&](std::size_t) { return s.attn_work_floats; });
}

bool Model::AllocScratch(std::string* error) {
  if (hy4_mmq_init(0) != 0)
    return Fail(error, "hy4: mmq init failed");
  // Prompt passes take the wide scratch; decode passes (up to kVecMax rows)
  // their own, so a decode round can run while a prompt pass is paused
  // between layer groups (Session::SetPrefillYield).
  scratch_ = std::make_unique<Scratch>();
  decode_scratch_ = std::make_unique<Scratch>();
  // Prompt passes of up to kResidentRows rows run on resident memory; the
  // rest of their max_chunk_ rows is mapped only for wider passes
  // (AcquireWidePasses), where device memory is demand-mapped.
  const std::uint32_t resident = std::min(max_chunk_, kResidentRows);
  if (resident < max_chunk_) {
    std::size_t bytes = 0;
    Scratch sizing;
    if (!AllocScratchRows(sizing, max_chunk_, resident, nullptr, &bytes))
      return Fail(error, "hy4: scratch sizing failed");
    std::string arena_error;
    arena_ = mem::ScratchArena::Reserve(bytes, &arena_error);
  }
  if (!AllocScratchRows(*scratch_, max_chunk_, arena_ ? resident : max_chunk_,
                        arena_.get()) ||
      !AllocScratchRows(*decode_scratch_, kVecMax))
    return Fail(error, "hy4: scratch allocation failed");
  if (arena_)
    std::fprintf(stderr,
                 "hy4: prompt scratch for %u-row passes, %.2f GiB beyond %u "
                 "rows mapped only for them\n",
                 max_chunk_, arena_->transient_bytes() / 1073741824.0,
                 resident);
  auto* owned = &allocations_;
  hipblasHandle_t blas = nullptr;
  if (hipblasCreate(&blas) != HIPBLAS_STATUS_SUCCESS || hipblasSetStream(blas, stream_) != HIPBLAS_STATUS_SUCCESS) {
    return Fail(error, "hy4: hipblas init failed");
  }
  blas_ = blas;
  // Zeroed, as hy4_gemv's split counters must start.
  std::uint8_t* gemv = nullptr;
  if (!DeviceAlloc(&gemv, hy4_gemv_scratch_bytes(), owned))
    return Fail(error, "hy4: gemv scratch allocation failed");
  gemv_scratch_ = gemv;
  for (Layer& l : weights_.layers) {
    const std::uint64_t blocks =
        l.wk_b.dims[0] * l.wk_b.dims[1] * l.wk_b.dims[2] / 32;
    std::uint16_t* q = nullptr;
    float* d = nullptr;
    if (!DeviceAlloc(&q, blocks * 32, owned) ||
        !DeviceAlloc(&d, blocks, owned) ||
        hy4_q8_0_split(l.wk_b.data, blocks, q, d, stream_) != 0) {
      return Fail(error, "hy4: wk_b split failed");
    }
    l.wk_b_q = q;
    l.wk_b_d = d;
  }
  if (hipStreamSynchronize(stream_) != hipSuccess)
    return Fail(error, "hy4: wk_b split failed");
  return RouterToBf16(error);
}

// The router is stored as F32 whose values are BF16 (the checkpoint's): kept
// as BF16 it is the same matrix at half the bytes per token. A router with
// values BF16 cannot hold stays F32.
bool Model::RouterToBf16(std::string* error) {
  std::uint16_t* tmp = nullptr;
  std::int32_t* inexact = nullptr;
  std::uint64_t tmp_count = 0;
  struct Free {
    std::uint16_t*& tmp;
    std::int32_t*& inexact;
    ~Free() {
      if (tmp)
        (void)hipFree(tmp);
      if (inexact)
        (void)hipFree(inexact);
    }
  } free_all{tmp, inexact};
  if (hipMalloc(&inexact, sizeof(std::int32_t)) != hipSuccess)
    return Fail(error, "hy4: router conversion failed");
  std::uint32_t converted = 0;
  for (Layer& l : weights_.layers) {
    Tensor& w = l.gate_inp;
    if (!w || w.type != HY4_TYPE_F32)
      continue;
    const std::uint64_t count = w.dims[0] * w.dims[1];
    if (count > tmp_count) {
      if (tmp)
        (void)hipFree(tmp);
      tmp = nullptr;
      if (hipMalloc(&tmp, count * sizeof(std::uint16_t)) != hipSuccess)
        return Fail(error, "hy4: router conversion failed");
      tmp_count = count;
    }
    std::int32_t flag = 0;
    if (hipMemsetAsync(inexact, 0, sizeof(flag), stream_) != hipSuccess ||
        hy4_f32_to_bf16(w.f32(), count, tmp, inexact, stream_) != 0 ||
        hipMemcpyAsync(&flag, inexact, sizeof(flag), hipMemcpyDeviceToHost,
                       stream_) != hipSuccess ||
        hipStreamSynchronize(stream_) != hipSuccess)
      return Fail(error, "hy4: router conversion failed");
    if (flag)
      continue;
    // The BF16 rows take the first half of the tensor's own allocation.
    if (hipMemcpyAsync(const_cast<void*>(w.data), tmp,
                       count * sizeof(std::uint16_t), hipMemcpyDeviceToDevice,
                       stream_) != hipSuccess)
      return Fail(error, "hy4: router conversion failed");
    w.type = HY4_TYPE_BF16;
    ++converted;
  }
  if (hipStreamSynchronize(stream_) != hipSuccess)
    return Fail(error, "hy4: router conversion failed");
  if (converted)
    std::fprintf(stderr, "hy4: %u routers kept as BF16\n", converted);
  return true;
}

std::uint32_t Model::SessionContext(std::uint32_t max_context) const {
  const Config& c = config_;
  max_context = std::max<std::uint32_t>(max_context, c.kpool);
  return (max_context + c.kpool - 1) / c.kpool * c.kpool;
}

std::uint32_t Model::KvRing(std::uint32_t layer, std::uint32_t max_context) const {
  max_context = SessionContext(max_context);
  return config_.is_swa[layer]
             ? std::min(max_context, config_.swa_window + max_chunk_)
             : max_context;
}

std::unique_ptr<Session> Model::CreateSession(std::uint32_t max_context, std::string* error) {
  const Config& c = config_;
  max_context = SessionContext(max_context);
  std::unique_ptr<Session> s(new Session(this, max_context));
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t pools = max_context / c.kpool + 1;
  s->cache_.assign(c.n_layer, nullptr);
  s->ring_.assign(c.n_layer, 0);
  s->idx_rows_.assign(c.n_layer, nullptr);
  s->idx_pooled_.assign(c.n_layer, nullptr);
  bool ok = true;
  for (std::uint32_t i = 0; ok && i < c.n_layer; ++i) {
    s->ring_[i] = KvRing(i, max_context);
    ok = DeviceAlloc(&s->cache_[i], std::size_t{s->ring_[i]} * dk, &s->allocations_);
    if (ok && c.has_indexer[i]) {
      ok = DeviceAlloc(&s->idx_rows_[i], std::size_t{Model::kIndexerRing} * 2 * c.idx_dim, &s->allocations_) &&
           DeviceAlloc(&s->idx_pooled_[i], std::size_t{pools} * c.idx_dim, &s->allocations_);
    }
  }
  const std::uint32_t top_pools = c.idx_top_k / c.kpool - 1;
  ok =
      ok &&
      DeviceAlloc(&s->score_, std::size_t{kScoreRows} * pools,
                  &s->allocations_) &&
      DeviceAlloc(&s->sel_, std::size_t{max_chunk_} * (top_pools + 1) * c.kpool,
                  &s->allocations_);
  if (!ok) {
    Fail(error, "hy4: session allocation failed");
    return nullptr;
  }
  return s;
}

Session::~Session() {
  for (void* p : allocations_) (void)hipFree(p);
}

bool Session::Evaluate(std::span<const std::int32_t> tokens,
                       std::vector<float>* logits, bool all_logits,
                       std::string* error, bool prompt) {
  prompt_pass_ = prompt;
  struct Reset {
    bool& flag;
    ~Reset() { flag = false; }
  } reset{prompt_pass_};
  if (logits) logits->clear();
  std::vector<float> part;
  const std::size_t chunk =
      prompt ? model_->PassRows(tokens.size()) : model_->MaxChunk();
  for (std::size_t i = 0; i < tokens.size(); i += chunk) {
    const std::size_t n = std::min<std::size_t>(chunk, tokens.size() - i);
    const bool last = i + n == tokens.size();
    std::vector<float>* want = logits && (all_logits || last) ? &part : nullptr;
    if (!EvaluateChunk(tokens.subspan(i, n), want, all_logits, error)) return false;
    if (want) logits->insert(logits->end(), part.begin(), part.end());
  }
  return true;
}

bool Session::PromptThrough(std::span<const std::int32_t> tokens,
                            std::span<const std::uint32_t> rows,
                            std::vector<std::vector<float>>* row_logits,
                            std::string* error) {
  if (tokens.size() > model_->PassRows(tokens.size()) || !row_logits ||
      (!rows.empty() && rows.back() + 1 >= tokens.size()))
    return Fail(error, "hy4: invalid in-pass logits rows");
  row_logits->assign(rows.size(), {});
  through_rows_ = rows;
  through_logits_ = row_logits;
  struct Clear {
    Session& s;
    ~Clear() {
      s.through_rows_ = {};
      s.through_logits_ = nullptr;
    }
  } clear{*this};
  return Prompt(tokens, error);
}

bool Session::Verify(std::span<const std::int32_t> tokens, std::string* error) {
  if (tokens.empty() || tokens.size() > kTapRows)
    return Fail(error, "hy4: verify takes 1 to 16 tokens");
  pass_pos0_ = Position();
  tap_ready_ = false;
  if (!Evaluate(tokens, &pass_logits_, true, error)) return false;
  last_logits_.assign(pass_logits_.end() - model_->config().n_vocab, pass_logits_.end());
  return true;
}

bool Session::DecodeBatch(std::span<Session* const> sessions,
                          std::span<const std::span<const std::int32_t>> tokens,
                          std::string* error) {
  if (sessions.size() != tokens.size() || sessions.empty())
    return Fail(error, "hy4: invalid decode batch");
  if (sessions.size() == 1) {
    return tokens[0].size() == 1 ? sessions[0]->Step(tokens[0], error)
                                 : sessions[0]->Verify(tokens[0], error);
  }
  std::vector<PassPart> parts;
  for (std::size_t i = 0; i < sessions.size(); ++i) {
    Session& se = *sessions[i];
    if (tokens[i].empty() || tokens[i].size() > kTapRows)
      return Fail(error, "hy4: verify takes 1 to 16 tokens");
    se.prompt_pass_ = false;
    if (tokens[i].size() == 1) {
      parts.push_back({&se, tokens[i], &se.last_logits_, false});
    } else {
      se.pass_pos0_ = se.Position();
      se.tap_ready_ = false;
      parts.push_back({&se, tokens[i], &se.pass_logits_, true});
    }
  }
  if (!Pass(parts, error))
    return false;
  for (const PassPart& part : parts) {
    Session& se = *part.session;
    if (part.all_logits)
      se.last_logits_.assign(
          se.pass_logits_.end() - se.model_->config().n_vocab,
          se.pass_logits_.end());
  }
  return true;
}

std::span<const float> Session::PassLogits(std::uint32_t row) const {
  const std::size_t v = model_->config().n_vocab;
  return std::span<const float>(pass_logits_).subspan(row * v, v);
}

void Session::Keep(std::uint32_t tokens) {
  const std::uint32_t evaluated = Position() - pass_pos0_;
  tokens = std::clamp<std::uint32_t>(tokens, 1, evaluated);
  history_.resize(pass_pos0_ + tokens);
  rope_.resize(history_.size());
  const auto row = PassLogits(tokens - 1);
  last_logits_.assign(row.begin(), row.end());
  // The pass left its last tap_rows_ (= evaluated) rows.
  if (tap_ready_) tap_row_ = tap_rows_ - (evaluated - tokens) - 1;
}

std::array<std::int32_t, 3> Session::PassRope(std::uint32_t p,
                                              std::uint32_t pos0) const {
  return p < pos0 ? rope_.at(p) : chunk_rope_.at(p - pos0);
}

void Session::RestoreRope(std::int32_t delta, std::uint32_t group_first,
                          std::array<std::int32_t, 3> group_rope) {
  rope_delta_ = delta;
  rope_.resize(history_.size());
  for (std::size_t p = 0; p < rope_.size(); ++p) {
    const std::int32_t t = static_cast<std::int32_t>(p) + delta;
    rope_[p] = {t, t, t};
  }
  if (group_first < rope_.size())
    rope_[group_first] = group_rope;
}

bool Session::EnableTaps(std::span<const std::uint32_t> layers, std::string* error) {
  const Config& c = model_->config();
  if (tap_hidden_) return Fail(error, "hy4: taps already enabled");
  tap_index_.assign(c.n_layer, -1);
  tap_sel_count_ = c.idx_top_k / c.kpool * c.kpool;
  for (const std::uint32_t layer : layers) {
    if (layer >= c.n_layer || !c.has_indexer[layer] || c.is_swa[layer]) {
      return Fail(error, "hy4: a tapped layer must be a sparse indexer layer");
    }
    tap_index_[layer] = static_cast<int>(tap_layers_.size());
    tap_layers_.push_back(layer);
    tap_sel_.push_back(nullptr);
    if (!DeviceAlloc(&tap_sel_.back(), std::size_t{kTapRows} * tap_sel_count_, &allocations_)) {
      return Fail(error, "hy4: tap allocation failed");
    }
  }
  tap_has_sel_.assign(layers.size(), false);
  if (!DeviceAlloc(&tap_hidden_, std::size_t{kTapRows} * c.n_embd, &allocations_)) {
    return Fail(error, "hy4: tap allocation failed");
  }
  return true;
}

const float* Session::TapHidden() const {
  return tap_hidden_ + std::size_t{tap_row_} * model_->config().n_embd;
}

const std::int32_t* Session::TapSelection(std::uint32_t k, std::uint32_t* count) const {
  *count = tap_sel_count_;
  return tap_has_sel_[k] ? tap_sel_[k] + std::size_t{tap_row_} * tap_sel_count_ : nullptr;
}

bool Session::EvaluateChunk(std::span<const std::int32_t> tokens, std::vector<float>* logits, bool all_logits,
                            std::string* error) {
  const PassPart part{this, tokens, logits, all_logits};
  return Pass(std::span(&part, 1), error);
}

bool Session::PrepareRows(std::span<const std::int32_t> tokens,
                          std::uint32_t row0, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  Model::Scratch& s = *scratch_;
  hipStream_t st = m.stream_;
  const std::uint32_t n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t pos0 = Position();
  if (pos0 + n > max_context_) return Fail(error, "hy4: context full");
  for (const std::int32_t t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.n_vocab) return Fail(error, "hy4: token out of range");
  }

  // Rope positions (decision 19): an image row (r, c) rotates at (s, s + r,
  // s + c), with s the position its image starts at; text continues after
  // the image's square of positions. Committed with the history.
  image_copies_.clear();
  std::vector<std::array<std::int32_t, 3>> rope(n);
  std::int32_t delta = rope_delta_;
  const std::span<const ImageSpan> spans =
      images_ ? images_->Spans() : std::span<const ImageSpan>{};
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t p = pos0 + i;
    const auto it = std::ranges::find_if(spans, [&](const ImageSpan& span) {
      return p >= span.first && p < span.End();
    });
    if (it == spans.end()) {
      const std::int32_t t = static_cast<std::int32_t>(p) + delta;
      rope[i] = {t, t, t};
      continue;
    }
    const std::uint32_t k = static_cast<std::uint32_t>(it - spans.begin());
    const std::uint32_t offset = p - it->first;
    const std::int32_t start = static_cast<std::int32_t>(it->first) + delta;
    rope[i] = {start,
               start + static_cast<std::int32_t>(offset / (it->width + 1)),
               start + static_cast<std::int32_t>(offset % (it->width + 1))};
    s.rope_3d = true;
    if (!image_copies_.empty() && image_copies_.back().image == k &&
        image_copies_.back().row + image_copies_.back().count == row0 + i) {
      ++image_copies_.back().count;
    } else {
      image_copies_.push_back({row0 + i, k, offset, 1, nullptr});
    }
    if (p + 1 == it->End()) {
      delta = start +
              static_cast<std::int32_t>(std::max(it->height, it->width + 1)) -
              static_cast<std::int32_t>(it->End());
    }
  }
  // Encode before this pass writes any row.
  for (ImageCopy& copy : image_copies_) {
    copy.rows = images_->Embeddings(copy.image, error);
    if (copy.rows == nullptr)
      return false;
  }

  if (write_hook_) write_hook_(write_user_, pos0, pos0 + n);
  std::vector<std::int32_t> pos(n), rope_t(n), rope_hw(2 * n);
  for (std::uint32_t i = 0; i < n; ++i) {
    pos[i] = static_cast<std::int32_t>(pos0 + i);
    rope_t[i] = rope[i][0];
    rope_hw[2 * i] = rope[i][1];
    rope_hw[2 * i + 1] = rope[i][2];
  }
  HY4_CHECK(hipMemcpyAsync(s.tokens + row0, tokens.data(),
                           n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                           st));
  HY4_CHECK(hipMemcpyAsync(s.pos + row0, pos.data(), n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, st));
  HY4_CHECK(hipMemcpyAsync(s.rope + row0, rope_t.data(),
                           n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                           st));
  HY4_CHECK(hipMemcpyAsync(s.rope_hw + 2 * row0, rope_hw.data(),
                           2 * n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                           st));
  chunk_rope_ = std::move(rope);
  chunk_delta_ = delta;
  return true;
}

std::uint32_t Model::PassRows(std::size_t tokens) const {
  const std::uint32_t resident = std::min(max_chunk_, kResidentRows);
  if (!arena_ || arena_->transient_mapped())
    return max_chunk_;
  if (tokens <= resident)
    return resident;
  // The wide rows, with room to spare, must fit the device heap after the
  // state promised to sessions and leave the host its free floor. Mapping
  // may reclaim clean page cache, never the RAM cache.
  constexpr std::size_t kSpare = std::size_t{1} << 30;
  const std::size_t bytes = arena_->transient_bytes() + kSpare;
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  if (mem::MemGetInfo(&free_bytes, &total_bytes) != hipSuccess) {
    (void)hipGetLastError();
    return resident;
  }
  if (free_bytes < bytes ||
      mem::HostAvailableBytes() < mem::HostFreeFloorBytes() + bytes)
    return resident;
  return arena_->MapTransient(nullptr) ? max_chunk_ : resident;
}

void Model::ReleaseWidePasses() const {
  // Unmapping waits for in-flight GPU work.
  if (arena_ && arena_->transient_mapped() && !wide_pass_running_)
    arena_->ReleaseTransient();
}

bool Session::Pass(std::span<const PassPart> parts, std::string* error) {
  Session& first = *parts.front().session;
  Model& m = *first.model_;
  const Config& c = m.config_;
  const Weights& w = m.weights_;
  hipStream_t st = m.stream_;
  const bool prompt = first.prompt_pass_;
  const Exec ex{st, m.blas_, prompt, m.gemv_scratch_};
  // Each part's rows follow the previous part's.
  std::vector<Rows> rows;
  std::uint32_t n = 0;
  for (const PassPart& part : parts) {
    if (part.session->model_ != &m || part.session->prompt_pass_ != prompt)
      return Fail(error, "hy4: a pass mixes models or prompt and decode rows");
    rows.push_back(
        {part.session, n, static_cast<std::uint32_t>(part.tokens.size())});
    n += rows.back().n;
  }
  if (n == 0)
    return true;
  // Several sessions share only decode passes, whose rows compute what a
  // one-token pass would.
  if (parts.size() > 1 && (prompt || n > Model::kVecMax))
    return Fail(error,
                "hy4: sessions share only decode passes of up to 16 rows");
  // A decode pass returns the wide prompt rows' memory; a pass using them
  // keeps them mapped through its yields.
  if (!prompt)
    m.ReleaseWidePasses();
  if (prompt && n > std::min(m.max_chunk_, Model::kResidentRows) && m.arena_ &&
      !m.arena_->transient_mapped())
    return Fail(error, "hy4: a wide prompt pass without its rows mapped");
  const bool wide_rows = prompt && n > Model::kResidentRows;
  struct WideRun {
    bool& flag;
    bool set;
    ~WideRun() {
      if (set)
        flag = false;
    }
  } wide_run{m.wide_pass_running_, wide_rows};
  if (wide_rows)
    m.wide_pass_running_ = true;
  // Decode passes take their own scratch; a prompt pass the wide one, which
  // a paused prompt pass holds (only decode rounds run in its yields).
  const bool wide = prompt || n > Model::kVecMax;
  if (wide && m.prompt_paused_)
    return Fail(error, "hy4: a paused prompt pass holds the prefill scratch");
  Model::Scratch& s = wide ? *m.scratch_ : *m.decode_scratch_;
  s.rope_3d = false;
  // Batched passes also keep cur in binary16 for the FP16 kernels.
  s.batched = n > Model::kVecMax || prompt;
  const bool batched = s.batched;
  for (const Rows& part : rows) {
    part.session->scratch_ = &s;
    if (part.n == 0)
      continue;
    const auto& tokens = parts[&part - rows.data()].tokens;
    if (!part.session->PrepareRows(tokens, part.row0, error))
      return false;
  }
  // Layers after which a prompt pass yields to the host's decode rounds.
  std::vector<std::uint32_t> yield_after;
  if (prompt && first.yield_ && first.yield_groups_ > 1) {
    for (std::size_t k = 1; k < first.yield_groups_; ++k) {
      const auto layer =
          static_cast<std::uint32_t>(k * c.n_layer / first.yield_groups_) - 1;
      if (layer + 1 < c.n_layer &&
          (yield_after.empty() || yield_after.back() != layer))
        yield_after.push_back(layer);
    }
  }

  // Engram rows for every module, hashed against each session's history.
  const std::uint32_t modules = static_cast<std::uint32_t>(c.engram_layers.size());
  const std::uint32_t cols = modules ? c.EngramCols() : 0;
  // Each module's rows are read from the table by the Engram readers while
  // the layers before it run, and copied to the device just before its
  // layer: a new prompt's rows are mostly read from the drive.
  std::vector<std::uint64_t> engram_ids(std::size_t{modules} * n * cols);
  std::vector<std::shared_ptr<EngramReader::Gather>> gathers;
  struct WaitAll {
    EngramReader* reader;
    std::vector<std::shared_ptr<EngramReader::Gather>>& gathers;
    ~WaitAll() {
      for (auto& g : gathers)
        if (g)
          (void)reader->Wait(*g);
    }
  } wait_all{modules ? &m.Engram() : nullptr, gathers};
  for (std::uint32_t e = 0; e < modules; ++e) {
    const std::size_t at = std::size_t{e} * n * cols;
    for (const Rows& part : rows) {
      const std::size_t row = at + std::size_t{part.row0} * cols;
      m.EngramRows(part.session->history_, parts[&part - rows.data()].tokens,
                   part.session->Position(), e, engram_ids.data() + row,
                   s.engram_valid_host + row);
    }
    for (std::uint32_t i = 0; i < n; ++i) {
      float any = 0.0f;
      for (std::uint32_t col = 0; col < cols; ++col)
        any += s.engram_valid_host[at + std::size_t{i} * cols + col];
      s.engram_any_host[e * n + i] = any > 0.0f ? 1.0f : 0.0f;
    }
    // A decode pass's rows go before a paused prompt pass's queued reads.
    gathers.push_back(m.Engram().Start(
        std::span(engram_ids).subspan(at, std::size_t{n} * cols),
        s.engram_host + at * c.engram_dim, !prompt));
  }
  const auto engram_ready = [&](std::uint32_t e) -> hipError_t {
    const auto g0 = std::chrono::steady_clock::now();
    const bool read = m.Engram().Wait(*gathers[e]);
    gathers[e].reset();
    first.engram_seconds_ +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - g0)
            .count();
    if (!read)
      return hipErrorFileNotFound;
    const std::size_t rows = std::size_t{e} * n * cols;
    hipError_t rc = hipMemcpyAsync(s.engram_rows + rows * c.engram_dim,
                                   s.engram_host + rows * c.engram_dim,
                                   std::size_t{n} * cols * c.engram_dim,
                                   hipMemcpyHostToDevice, st);
    if (rc == hipSuccess)
      rc = hipMemcpyAsync(s.engram_valid + rows, s.engram_valid_host + rows,
                          std::size_t{n} * cols * sizeof(float),
                          hipMemcpyHostToDevice, st);
    if (rc == hipSuccess)
      rc = hipMemcpyAsync(s.engram_any + std::size_t{e} * n,
                          s.engram_any_host + std::size_t{e} * n,
                          std::size_t{n} * sizeof(float), hipMemcpyHostToDevice,
                          st);
    return rc;
  };

  HY4_CHECK(hy4_embed_q8_0(w.tok_embd.data, s.tokens, n, c.n_embd, s.x, st));
  for (const Rows& part : rows) {
    for (const ImageCopy& copy : part.session->image_copies_) {
      HY4_CHECK(
          hipMemcpyAsync(s.x + std::size_t{copy.row} * c.n_embd,
                         copy.rows + std::size_t{copy.offset} * c.n_embd,
                         std::size_t{copy.count} * c.n_embd * sizeof(float),
                         hipMemcpyDeviceToDevice, st));
    }
  }
  HY4_CHECK(hy4_hc_init(s.x, s.h, n, c.hc, c.n_embd, st));

  auto norm_cur = [&](const Tensor& weight) {
    return batched ? hy4_rms_norm_half(s.x, weight.f32(), s.cur, s.cur_half, n,
                                       c.n_embd, c.n_embd, c.rms_eps, st)
                   : hy4_rms_norm(s.x, weight.f32(), s.cur, n, c.n_embd,
                                  c.n_embd, c.rms_eps, st);
  };

  // A half's hy4_hc_post is deferred into the next pre-mix
  // (hy4_hc_post_pre) unless something reads h in between. Decode passes
  // also defer the half's output norm (weight y_norm) and run the boundary,
  // norms included, as hy4_hc_decode.
  bool post_pending = false;
  const float* y_norm = nullptr;
  auto hc_pre = [&](const Tensor& fn, const Tensor& scale, const Tensor& base,
                    float* post, float magnitude) {
    if (!post_pending)
      return hy4_hc_pre(s.h, fn.f32(), scale.f32(), base.f32(), s.x, post, n,
                        c.hc, c.n_embd, c.rms_eps, c.hc_eps, magnitude, st);
    post_pending = false;
    return hy4_hc_post_pre(s.y, s.post, s.h, fn.f32(), scale.f32(), base.f32(),
                           s.x, post, n, c.hc, c.n_embd, c.rms_eps, c.hc_eps,
                           magnitude, st);
  };
  // The boundary into a half (or the head): pre-mix into s.x, then s.cur =
  // rms_norm(s.x, norm).
  auto boundary = [&](const Tensor& fn, const Tensor& scale, const Tensor& base,
                      float* post, float magnitude, const Tensor& norm,
                      bool half) {
    if (!batched) {
      const float* y = post_pending ? s.y : nullptr;
      post_pending = false;
      const float* yn = y_norm;
      y_norm = nullptr;
      return hy4_hc_decode(y, yn, s.post, s.h, fn.f32(), scale.f32(),
                           base.f32(), s.x, s.cur, norm.f32(), post, n, c.hc,
                           c.n_embd, c.rms_eps, c.hc_eps, magnitude, st);
    }
    if (int rc = hc_pre(fn, scale, base, post, magnitude))
      return rc;
    return half ? norm_cur(norm)
                : hy4_rms_norm(s.x, norm.f32(), s.cur, n, c.n_embd, c.n_embd,
                               c.rms_eps, st);
  };
  // A pending post applied now, for something that reads h.
  auto apply_post = [&]() {
    if (y_norm &&
        hy4_rms_norm(s.y, y_norm, s.y, n, c.n_embd, c.n_embd, c.rms_eps, st))
      return 1;
    y_norm = nullptr;
    post_pending = false;
    return hy4_hc_post(s.y, s.h, s.post, s.h, n, c.hc, c.n_embd, st);
  };

  for (std::uint32_t il = 0; il < c.n_layer; ++il) {
    const Layer& l = w.layers[il];
    const int e = c.EngramIndex(il);
    if (e >= 0 && post_pending)
      HY4_CHECK(apply_post());
    if (e >= 0) {
      HY4_CHECK(engram_ready(static_cast<std::uint32_t>(e)));
      const std::uint32_t width = cols * c.engram_dim;
      HY4_CHECK(hy4_engram_rows(s.engram_rows + std::size_t(e) * n * width, s.engram_valid + std::size_t(e) * n * cols,
                                l.engram_lookup_norm.f32(), s.engram_normed, n, cols, c.engram_dim, kEngramEps, st));
      if (!MatVec(l.engram_kv, s.engram_normed, s.engram_kv, (c.hc + 1) * c.n_embd, width, n, ex, error)) {
        return false;
      }
      HY4_CHECK(hy4_add_bias_rows(s.engram_kv, l.engram_kv_b.f32(), n, c.hc + 1, c.n_embd, st));
      HY4_CHECK(hy4_engram_apply(s.h, s.engram_kv, l.engram_gate.f32(), l.engram_hc_scale.f32(),
                                 l.engram_pre_gamma.f32(), s.engram_any + std::size_t(e) * n, n, c.hc, c.n_embd,
                                 kEngramEps, st));
    }

    HY4_CHECK(boundary(l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, s.post,
                       c.hc_magnitude, l.attn_norm, true));
    if (!Attention(rows, il, n, error))
      return false;
    // Attention normed s.y in batched passes only.
    post_pending = true;
    if (!batched)
      y_norm = l.attn_post_norm.f32();

    HY4_CHECK(boundary(l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, s.post,
                       c.hc_magnitude, l.ffn_norm, true));
    if (!first.Ffn(il, n, error))
      return false;
    post_pending = true;
    if (batched)
      HY4_CHECK(hy4_rms_norm(s.y, l.ffn_post_norm.f32(), s.y, n, c.n_embd,
                             c.n_embd, c.rms_eps, st));
    else
      y_norm = l.ffn_post_norm.f32();

    if (first.probe_) {
      HY4_CHECK(apply_post());
      HY4_CHECK(hipMemcpyAsync(s.host_h, s.h, std::size_t{n} * c.hc * c.n_embd * sizeof(float),
                               hipMemcpyDeviceToHost, st));
      HY4_CHECK(hipStreamSynchronize(st));
      first.probe_(first.probe_user_, static_cast<int>(il), s.host_h, n);
    }
    if (std::ranges::binary_search(yield_after, il)) {
      // The queued layers finish first: the host's decode round syncs the
      // stream and must not wait behind them mid-round.
      HY4_CHECK(hipStreamSynchronize(st));
      m.prompt_paused_ = true;
      struct Resume {
        Model& m;
        ~Resume() { m.prompt_paused_ = false; }
      } resume{m};
      first.yield_();
    }
  }

  // Head: reduce the streams, norm, project.
  HY4_CHECK(boundary(w.hc_head_fn, w.hc_head_scale, w.hc_head_base, nullptr,
                     1.0f, w.output_norm, false));
  for (const Rows& part : rows) {
    Session& se = *part.session;
    if (!se.tap_hidden_ || part.n == 0)
      continue;
    const std::uint32_t r = std::min(part.n, kTapRows);
    HY4_CHECK(hipMemcpyAsync(
        se.tap_hidden_, s.cur + std::size_t{part.row0 + part.n - r} * c.n_embd,
        std::size_t{r} * c.n_embd * sizeof(float), hipMemcpyDeviceToDevice,
        st));
    se.tap_rows_ = r;
    se.tap_row_ = r - 1;
    se.tap_ready_ = true;
  }
  // In-pass checkpoints' logits: each row through the head as a pass's last
  // row is (one row per product).
  for (std::size_t i = 0;
       first.through_logits_ && i < first.through_rows_.size(); ++i) {
    const std::uint32_t r = first.through_rows_[i];
    if (!MatVec(w.output, s.cur + std::size_t{r} * c.n_embd, s.logits,
                c.n_vocab, c.n_embd, 1, ex, error)) {
      return false;
    }
    auto& out = (*first.through_logits_)[i];
    out.resize(c.n_vocab);
    HY4_CHECK(hipMemcpyAsync(out.data(), s.logits, out.size() * sizeof(float),
                             hipMemcpyDeviceToHost, st));
  }
  if (parts.size() == 1) {
    if (auto* logits = parts.front().logits) {
      const std::uint32_t first_row = parts.front().all_logits ? 0 : n - 1;
      const std::uint32_t count = n - first_row;
      logits->resize(std::size_t{count} * c.n_vocab);
      // kVecMax rows per product: the scratch holds that many logit rows.
      for (std::uint32_t r = 0; r < count; r += Model::kVecMax) {
        const std::uint32_t rows = std::min(Model::kVecMax, count - r);
        if (!MatVec(w.output, s.cur + std::size_t{first_row + r} * c.n_embd,
                    s.logits, c.n_vocab, c.n_embd, rows, ex, error)) {
          return false;
        }
        HY4_CHECK(hipMemcpyAsync(logits->data() + std::size_t{r} * c.n_vocab,
                                 s.logits,
                                 std::size_t{rows} * c.n_vocab * sizeof(float),
                                 hipMemcpyDeviceToHost, st));
      }
    }
  } else {
    // One head product over every session's rows; each takes its own.
    if (!MatVec(w.output, s.cur, s.logits, c.n_vocab, c.n_embd, n, ex, error))
      return false;
    for (const Rows& part : rows) {
      auto* logits = parts[&part - rows.data()].logits;
      if (!logits || part.n == 0)
        continue;
      const std::uint32_t first_row =
          parts[&part - rows.data()].all_logits ? 0 : part.n - 1;
      logits->resize(std::size_t{part.n - first_row} * c.n_vocab);
      HY4_CHECK(hipMemcpyAsync(
          logits->data(),
          s.logits + std::size_t{part.row0 + first_row} * c.n_vocab,
          logits->size() * sizeof(float), hipMemcpyDeviceToHost, st));
    }
  }
  HY4_CHECK(hipStreamSynchronize(st));
  for (const Rows& part : rows) {
    Session& se = *part.session;
    const auto& tokens = parts[&part - rows.data()].tokens;
    se.history_.insert(se.history_.end(), tokens.begin(), tokens.end());
    se.rope_.insert(se.rope_.end(), se.chunk_rope_.begin(),
                    se.chunk_rope_.end());
    se.rope_delta_ = se.chunk_delta_;
  }
  return true;
}

bool Session::Attention(std::span<const Rows> parts, std::uint32_t il,
                        std::uint32_t n, std::string* error) {
  Session& first = *parts.front().session;
  Model& m = *first.model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *first.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_, first.prompt_pass_, m.gemv_scratch_};
  const bool batched = n > Model::kVecMax || first.prompt_pass_;
  const std::uint32_t heads = c.n_head[il];
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::uint32_t nope = c.head_k - c.n_rot;
  const float base = c.is_swa[il] ? c.rope_base_swa : c.rope_base;

  const std::uint16_t* cur_half = batched ? s.cur_half : nullptr;
  const std::uint16_t* qr_half = batched ? s.qr_half : nullptr;
  if (!MatVecPair(l.wq_a, s.qr, c.q_lora, l.wkv_a, s.kva, dk, s.cur, c.n_embd,
                  n, ex, error, cur_half))
    return false;
  const std::int32_t* hw = s.rope_3d ? s.rope_hw : nullptr;
  if (!batched) {
    // Decode: both latent norms and the key's rope in one launch, then the
    // query's rope, which also stores its tail in qfull.
    HY4_CHECK(hy4_mla_prep(s.qr, l.q_a_norm.f32(), c.q_lora, s.kva,
                           l.kv_a_norm.f32(), c.kv_lora, dk, s.rope, hw,
                           c.n_rot, base, c.rms_eps, n, st));
    if (!MatVec(l.wq_b, s.qr, s.q, heads * c.head_k, c.q_lora, n, ex, error))
      return false;
    HY4_CHECK(hy4_rope_3d_copy(s.q, s.rope, hw, n, heads,
                               std::uint64_t{heads} * c.head_k, c.head_k, nope,
                               c.n_rot, base, s.qfull + c.kv_lora,
                               std::uint64_t{heads} * dk, dk, st));
  } else {
    HY4_CHECK(hy4_rms_norm_half(s.qr, l.q_a_norm.f32(), s.qr, s.qr_half, n,
                                c.q_lora, c.q_lora, c.rms_eps, st));
    if (!MatVec(l.wq_b, s.qr, s.q, heads * c.head_k, c.q_lora, n, ex, error,
                qr_half))
      return false;
    HY4_CHECK(hy4_rms_norm(s.kva, l.kv_a_norm.f32(), s.kva, n, c.kv_lora, dk,
                           c.rms_eps, st));
    HY4_CHECK(hy4_rope_3d(s.kva, s.rope, hw, n, 1, dk, 0, c.kv_lora, c.n_rot,
                          base, st));
    HY4_CHECK(hy4_rope_3d(s.q, s.rope, hw, n, heads,
                          std::uint64_t{heads} * c.head_k, c.head_k, nope,
                          c.n_rot, base, st));
  }

  // Absorb wk_b into the query: q_abs[h] = wk_b[h] (512 x 192) q_nope[h].
  if (l.wk_b.type != HY4_TYPE_Q8_0 || l.wv_b.type != HY4_TYPE_Q8_0) return Fail(error, "hy4: wk_b/wv_b must be Q8_0");
  // Prefill passes hand MlaPrefill binary16 queries in qfull's storage.
  auto* qh = reinterpret_cast<std::uint16_t*>(s.qfull);
  if (batched) {
    HY4_CHECK(hy4_mla_absorb(l.wk_b_q, l.wk_b_d, s.q, qh, n, heads, nope,
                             c.n_rot, c.kv_lora, st));
  } else {
    HY4_CHECK(hy4_gemv_q8_0_grouped(l.wk_b.data, s.q, s.qfull, c.kv_lora, nope, heads, n,
                                    std::uint64_t{heads} * c.head_k, c.head_k, std::uint64_t{heads} * dk, dk, st));
  }
  if (c.has_indexer[il] && !IndexerProject(parts, il, n, error))
    return false;
  // Each session's rows: its cache, its keys and its selection.
  for (const Rows& part : parts) {
    if (!part.session->Attend(il, part.row0, part.n, error))
      return false;
  }
  if (batched) {
    // The gate first, so that wv_b's stores apply it and round the gated
    // values to binary16 once, which wo then reads.
    auto* att_half = reinterpret_cast<std::uint16_t*>(s.att);
    if (!MatVec(l.attn_gate, s.cur, s.gate, heads * c.head_v, c.n_embd, n, ex,
                error, cur_half))
      return false;
    if (l.wo.type != HY4_TYPE_Q8_0 && l.wo.type != HY4_TYPE_Q4_K &&
        l.wo.type != HY4_TYPE_Q6_K)
      return Fail(error, "hy4: wo must be Q8_0, Q4_K or Q6_K");
    auto* attv_half = reinterpret_cast<std::uint16_t*>(s.attv);
    HY4_CHECK(hy4_f16_dense_gated(
        l.wv_b.type, l.wv_b.data, att_half, attv_half, c.head_v, n, c.kv_lora,
        heads, HY4_F16_X_HALF | HY4_F16_OUT_HALF, s.gate, st));
    HY4_CHECK(hy4_f16_dense(l.wo.type, l.wo.data, attv_half, s.y, c.n_embd, n,
                            heads * c.head_v, 1, HY4_F16_X_HALF, st));
  } else {
    HY4_CHECK(hy4_gemv_q8_0_grouped(
        l.wv_b.data, s.att, s.attv, c.head_v, c.kv_lora, heads, n,
        std::uint64_t{heads} * c.kv_lora, c.kv_lora,
        std::uint64_t{heads} * c.head_v, c.head_v, st));
    if (!MatVecGate(l.attn_gate, s.cur, s.gate, s.attv, heads * c.head_v,
                    c.n_embd, n, ex, error))
      return false;
    if (!MatVec(l.wo, s.attv, s.y, c.n_embd, heads * c.head_v, n, ex, error))
      return false;
  }
  // Decode passes norm s.y in the next boundary (hy4_hc_decode).
  if (batched)
    HY4_CHECK(hy4_rms_norm(s.y, l.attn_post_norm.f32(), s.y, n, c.n_embd,
                           c.n_embd, c.rms_eps, st));
  return true;
}

bool Session::Attend(std::uint32_t il, std::uint32_t row0, std::uint32_t n,
                     std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *scratch_;
  hipStream_t st = m.stream_;
  const bool batched = s.batched;
  const std::uint32_t heads = c.n_head[il];
  const std::uint32_t dk = c.kv_lora + c.n_rot;
  const std::int32_t* pos = s.pos + row0;
  HY4_CHECK(hy4_store_rows_half(s.kva + std::size_t{row0} * dk, dk, cache_[il],
                                ring_[il], pos, n, dk, st));

  const std::int32_t* sel = nullptr;
  std::uint32_t n_sel = 0;
  if (c.has_indexer[il] &&
      !Indexer(il, row0, n, Position(), &sel, &n_sel, error))
    return false;
  if (!tap_index_.empty() && tap_index_[il] >= 0) {
    const auto k = static_cast<std::uint32_t>(tap_index_[il]);
    const std::uint32_t r = std::min(n, kTapRows);
    tap_has_sel_[k] = sel != nullptr;
    if (sel) {
      if (n_sel != tap_sel_count_) return Fail(error, "hy4: tap selection size mismatch");
      HY4_CHECK(hipMemcpyAsync(tap_sel_[k], sel + std::size_t{n - r} * n_sel, std::size_t{r} * n_sel * sizeof(std::int32_t),
                               hipMemcpyDeviceToDevice, st));
    }
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_k));
  if (batched) {
    // Prefill runs each token over exactly its own keys on the matrix cores,
    // reading the binary16 cache rows, so a token's result does not depend on
    // how its span was cut into passes. The ring holds the window and the
    // whole pass. A prefill pass is one session's.
    auto* qh = reinterpret_cast<std::uint16_t*>(s.qfull);
    auto* att_half = reinterpret_cast<std::uint16_t*>(s.att);
    const std::uint32_t window = c.is_swa[il] ? c.swa_window : 0;
    std::int32_t lo = 0;
    if (window)
      lo = std::max<std::int32_t>(0, static_cast<std::int32_t>(Position()) -
                                         static_cast<std::int32_t>(window) + 1);
    HY4_CHECK(hy4_mla_prefill(qh, cache_[il], ring_[il], lo, s.pos, window, sel,
                              n_sel, l.attn_sinks.f32(), scale, att_half, n,
                              heads, dk, c.kv_lora, st));
  } else {
    // Decoding and verify passes: each token over exactly the keys a
    // one-token pass would see.
    HY4_CHECK(hy4_mla_attention(
        s.qfull + std::size_t{row0} * heads * dk, cache_[il], ring_[il], pos,
        c.is_swa[il] ? c.swa_window : 0, sel, n_sel, l.attn_sinks.f32(), scale,
        s.att + std::size_t{row0} * heads * c.kv_lora, n, heads, dk, c.kv_lora,
        s.attn_work, s.attn_work_floats, st));
  }
  return true;
}

bool Session::IndexerProject(std::span<const Rows> parts, std::uint32_t il,
                             std::uint32_t n, std::string* error) {
  Session& first = *parts.front().session;
  Model& m = *first.model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *first.scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_, first.prompt_pass_, m.gemv_scratch_};
  const std::uint32_t d = c.idx_dim;
  const float base = c.rope_base;  // indexer layers are the sparse ones

  // Key side: normed key, then its value and gate projections.
  const bool batched = s.batched;
  const std::uint16_t* cur_half = batched ? s.cur_half : nullptr;
  if (!MatVec(l.idx_k, s.cur, s.ik, d, c.n_embd, n, ex, error, cur_half))
    return false;
  HY4_CHECK(hy4_rms_norm(s.ik, l.idx_k_norm.f32(), s.ik, n, d, d, c.rms_eps, st));
  if (!MatVec(l.idx_value, s.ik, s.ffn_a, d, d, n, ex, error)) return false;
  if (!MatVec(l.idx_gate, s.ik, s.ffn_b, d, d, n, ex, error)) return false;
  // The scratch may be demand-mapped, which hipMemcpy2DAsync rejects.
  HY4_CHECK(hy4_copy_2d(s.ivg, 2 * d, s.ffn_a, d, d, n, st));
  HY4_CHECK(hy4_copy_2d(s.ivg + d, 2 * d, s.ffn_b, d, d, n, st));

  // Query side, when some session's tokens see more pools than the top-k
  // keeps (below that every pool is selected: plain causal attention).
  const std::uint32_t top = c.idx_top_k / c.kpool - 1;
  const bool select = std::ranges::any_of(parts, [&](const Rows& part) {
    return (part.session->Position() + part.n - 1) / c.kpool > top;
  });
  if (!select)
    return true;
  if (!MatVec(l.idx_q_b, s.qr, s.iq, c.idx_heads * d, c.q_lora, n, ex, error,
              batched ? s.qr_half : nullptr))
    return false;
  HY4_CHECK(hy4_rope_3d(s.iq, s.rope, s.rope_3d ? s.rope_hw : nullptr, n,
                        c.idx_heads, std::uint64_t{c.idx_heads} * d, d,
                        d - c.n_rot, c.n_rot, base, st));
  // The 1 / sqrt(d heads) weight scale does not change the selection.
  if (!MatVec(l.idx_proj, s.cur, s.iw, c.idx_heads, c.n_embd, n, ex, error,
              cur_half))
    return false;
  return true;
}

bool Session::Indexer(std::uint32_t il, std::uint32_t row0, std::uint32_t n,
                      std::uint32_t pos0, const std::int32_t** sel,
                      std::uint32_t* n_sel, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *scratch_;
  hipStream_t st = m.stream_;
  const std::uint32_t d = c.idx_dim;
  const float base = c.rope_base;  // indexer layers are the sparse ones
  HY4_CHECK(hy4_store_rows(s.ivg + std::size_t{row0} * 2 * d, 2 * d,
                           idx_rows_[il], Model::kIndexerRing, s.pos + row0, n,
                           2 * d, st));

  // Pool the groups these rows complete; a pooled key is roped at its first
  // member's position.
  std::vector<std::int32_t> groups, group_pos, group_hw;
  bool group_3d = false;
  for (std::uint32_t i = 0; i < n; ++i) {
    const std::uint32_t p = pos0 + i;
    if (p % c.kpool == c.kpool - 1) {
      groups.push_back(static_cast<std::int32_t>(p / c.kpool));
      const auto r = PassRope(p / c.kpool * c.kpool, pos0);
      group_pos.push_back(r[0]);
      group_hw.push_back(r[1]);
      group_hw.push_back(r[2]);
      group_3d = group_3d || r[1] != r[0] || r[2] != r[0];
    }
  }
  const std::uint32_t pool_rows = max_context_ / c.kpool + 1;
  if (!groups.empty()) {
    const std::uint32_t ng = static_cast<std::uint32_t>(groups.size());
    HY4_CHECK(hipMemcpyAsync(s.groups, groups.data(), ng * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
    HY4_CHECK(hipMemcpyAsync(s.group_pos, group_pos.data(), ng * sizeof(std::int32_t), hipMemcpyHostToDevice, st));
    if (group_3d) {
      HY4_CHECK(hipMemcpyAsync(s.group_hw, group_hw.data(),
                               2 * ng * sizeof(std::int32_t),
                               hipMemcpyHostToDevice, st));
    }
    HY4_CHECK(hy4_kpool_pool(idx_rows_[il], Model::kIndexerRing, s.groups, ng, c.kpool, l.idx_ape.f32(), l.idx_comp_norm.f32(),
                             c.rms_eps, idx_pooled_[il], pool_rows, d, st));
    HY4_CHECK(hy4_rope_3d(idx_pooled_[il] + std::size_t(groups[0]) * d,
                          s.group_pos, group_3d ? s.group_hw : nullptr, ng, 1,
                          d, 0, d - c.n_rot, c.n_rot, base, st));
  }

  // Selection, once a token sees more pools than the top-k keeps; below
  // that every pool is selected, which is plain causal attention.
  const std::uint32_t top = c.idx_top_k / c.kpool - 1;
  const std::uint32_t n_pool_max = (pos0 + n - 1) / c.kpool;
  if (n_pool_max <= top) return true;
  std::vector<std::int32_t> n_pool(n);
  for (std::uint32_t i = 0; i < n; ++i) n_pool[i] = static_cast<std::int32_t>((pos0 + i) / c.kpool);
  HY4_CHECK(hipMemcpyAsync(s.n_pool + row0, n_pool.data(),
                           n * sizeof(std::int32_t), hipMemcpyHostToDevice,
                           st));
  // Scores and selections in slices of kScoreRows tokens, so that the score
  // buffer stays [kScoreRows][pools] however wide the pass; each token's
  // selection depends only on its own visible pools.
  const std::uint32_t n_sel_row = (top + 1) * c.kpool;
  for (std::uint32_t t = 0; t < n; t += Model::kScoreRows) {
    const std::uint32_t rows = std::min(Model::kScoreRows, n - t);
    const std::uint32_t pools_max = (pos0 + t + rows - 1) / c.kpool;
    const std::size_t r = std::size_t{row0} + t;
    HY4_CHECK(hy4_indexer_score(
        s.iq + r * c.idx_heads * d, s.iw + r * c.idx_heads, idx_pooled_[il],
        s.n_pool + r, pools_max, score_, rows, c.idx_heads, d, st));
    HY4_CHECK(hy4_kpool_select(score_, s.n_pool + r, pools_max, s.pos + r, top,
                               c.kpool, sel_ + std::size_t{t} * n_sel_row, rows,
                               st));
  }
  *sel = sel_;
  *n_sel = (top + 1) * c.kpool;
  return true;
}

bool Session::Ffn(std::uint32_t il, std::uint32_t n, std::string* error) {
  Model& m = *model_;
  const Config& c = m.config_;
  const Layer& l = m.weights_.layers[il];
  Model::Scratch& s = *scratch_;
  hipStream_t st = m.stream_;
  const Exec ex{st, m.blas_, prompt_pass_, m.gemv_scratch_};
  const std::uint16_t* cur_half =
      n > Model::kVecMax || prompt_pass_ ? s.cur_half : nullptr;

  if (il < c.n_dense_lead) {
    const std::uint32_t f = c.n_ff_dense;
    if (!SwigluVec(l.ffn_gate, l.ffn_up, s.ffn_a, s.ffn_b, s.ffn_c, f, s.cur,
                   c.n_embd, n, ex, error, cur_half))
      return false;
    return MatVec(l.ffn_down, s.ffn_c, s.y, c.n_embd, f, n, ex, error);
  }

  const std::uint32_t f = c.n_ff_exp;
  const std::uint32_t used = c.n_expert_used;
  // Shared expert first, into its own buffer.
  if (!SwigluVec(l.gate_shexp, l.up_shexp, s.ffn_a, s.ffn_b, s.ffn_c, f, s.cur,
                 c.n_embd, n, ex, error, cur_half))
    return false;
  if (!MatVec(l.down_shexp, s.ffn_c, s.shared, c.n_embd, f, n, ex, error)) return false;

  // Router on the full-width input; experts in the latent space.
  if (!MatVec(l.gate_inp, s.cur, s.logits_r, c.n_expert, c.n_embd, n, ex, error)) return false;
  HY4_CHECK(hy4_route(s.logits_r, l.exp_bias ? l.exp_bias.f32() : nullptr,
                      s.ids, s.route_w, n, c.n_expert, used,
                      c.expert_norm ? 1 : 0, c.expert_scale, st));
  if (n > Model::kVecMax || prompt_pass_) {
    // The latent in binary16, the routed kernels' input format: each value
    // rounds once, as the kernels' staging rounded the FP32 latent.
    auto* latent = reinterpret_cast<std::uint16_t*>(s.latent);
    if (!DenseF16(l.routed_down, cur_half, latent, c.n_latent, c.n_embd, n,
                  HY4_F16_X_HALF | HY4_F16_OUT_HALF, ex, error))
      return false;
    if (l.gate_exps.type != l.up_exps.type) return Fail(error, "hy4: gate and up experts differ in type");
    // The routed plan: slots grouped by expert in 128-slot tiles.
    const std::uint32_t slots = n * used;
    HY4_CHECK(hipMemcpyAsync(s.ids_host, s.ids,
                             std::size_t{slots} * sizeof(std::int32_t),
                             hipMemcpyDeviceToHost, st));
    HY4_CHECK(hipStreamSynchronize(st));
    std::int32_t* rows = s.route_plan_host;
    std::int32_t* bounds = rows + slots;
    std::int32_t* tiles = bounds + c.n_expert + 1;
    const std::uint32_t n_tiles =
        hy4_f16_routed_plan(s.ids_host, slots, c.n_expert, rows, bounds, tiles);
    const std::size_t plan = std::size_t{slots} + c.n_expert + 1 + n_tiles;
    HY4_CHECK(hipMemcpyAsync(s.route_plan, s.route_plan_host,
                             plan * sizeof(std::int32_t), hipMemcpyHostToDevice,
                             st));
    const std::int32_t* d_rows = s.route_plan;
    const std::int32_t* d_bounds = d_rows + slots;
    const std::int32_t* d_tiles = d_bounds + c.n_expert + 1;
    // Gate and up with SwiGLU into FP16 rows (in ffn_c), then down.
    auto* mid = reinterpret_cast<std::uint16_t*>(s.ffn_c);
    HY4_CHECK(hy4_f16_routed(l.gate_exps.type, l.gate_exps.data, l.up_exps.data,
                             ExpertBytes(l.gate_exps), latent, used, d_rows,
                             d_bounds, d_tiles, n_tiles, s.route_w,
                             c.swiglu_clamp[il], mid, f, c.n_latent, st));
    HY4_CHECK(hy4_f16_routed(l.down_exps.type, l.down_exps.data, nullptr,
                             ExpertBytes(l.down_exps), mid, 1, d_rows, d_bounds,
                             d_tiles, n_tiles, nullptr, 0.0f, s.ffn_a,
                             c.n_latent, f, st));
    HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
    if (!MatVec(l.routed_up, s.moe, s.y, c.n_embd, c.n_latent, n, ex, error)) return false;
    HY4_CHECK(hy4_add(s.y, s.shared, s.y, std::uint64_t{n} * c.n_embd, st));
    return true;
  }
  if (!MatVec(l.routed_down, s.cur, s.latent, c.n_latent, c.n_embd, n, ex,
              error))
    return false;
  // FP32 expert GEMVs: SwiGLU(gate, up) times the route weight, then down.
  // A verify pass reads each expert's rows once and gives every slot the
  // bits of one-token decoding.
  const auto fp32_type = [](int type) {
    return type == HY4_TYPE_Q8_0 || type == HY4_TYPE_Q4_K ||
           type == HY4_TYPE_Q6_K;
  };
  if (l.gate_exps.type == l.up_exps.type && fp32_type(l.gate_exps.type) &&
      fp32_type(l.down_exps.type) && c.n_latent % 256 == 0 && f % 256 == 0) {
    const std::uint32_t slots = n * used;
    if (n > 1)
      HY4_CHECK(hy4_expert_plan(s.ids, slots, s.route_plan, st));
    const std::int32_t* plan = n > 1 ? s.route_plan : nullptr;
    HY4_CHECK(hy4_gemv_experts(
        l.gate_exps.type, l.gate_exps.data, l.up_exps.data,
        ExpertBytes(l.gate_exps), s.latent, used, s.ids, plan, s.route_w,
        c.swiglu_clamp[il], s.ffn_c, f, c.n_latent, slots, n, st));
    HY4_CHECK(hy4_gemv_experts(l.down_exps.type, l.down_exps.data, nullptr,
                               ExpertBytes(l.down_exps), s.ffn_c, 1, s.ids,
                               plan, nullptr, 0.0f, s.ffn_a, c.n_latent, f,
                               slots, n, st));
    HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
    return MatVecAdd(l.routed_up, s.moe, s.shared, s.y, c.n_embd, c.n_latent, n,
                     ex, error);
  }
  // A verify pass must compute what one-token decoding would. The shared
  // kernels give each slot the one-token launches' arithmetic and read each
  // expert's rows once for the pass (consecutive tokens share about a third
  // of their experts).
  const auto shared_type = [](int type) {
    return type == HY4_TYPE_Q8_0 || type == HY4_TYPE_Q4_K ||
           type == HY4_TYPE_Q6_K;
  };
  if (n > 1 && shared_type(l.gate_exps.type) && shared_type(l.up_exps.type) &&
      shared_type(l.down_exps.type)) {
    const auto f_ = static_cast<int>(f);
    const auto k_ = static_cast<int>(c.n_latent);
    const auto e_ = static_cast<int>(c.n_expert);
    HY4_CHECK(hy4_mmq_moe_vec_tokens_typed(
        l.gate_exps.type, l.gate_exps.data, s.latent, s.ids, s.ffn_a, f_, k_,
        static_cast<int>(n), e_, static_cast<int>(used), st));
    HY4_CHECK(hy4_mmq_moe_vec_tokens_typed(
        l.up_exps.type, l.up_exps.data, s.latent, s.ids, s.ffn_b, f_, k_,
        static_cast<int>(n), e_, static_cast<int>(used), st));
    HY4_CHECK(hy4_swiglu(s.ffn_a, s.ffn_b, s.route_w, s.ffn_c, n * used, f,
                         c.swiglu_clamp[il], st));
    HY4_CHECK(hy4_mmq_moe_vec_slots_typed(l.down_exps.type, l.down_exps.data,
                                          s.ffn_c, s.ids, s.ffn_a, k_, f_,
                                          static_cast<int>(n * used), e_, st));
    HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
    return MatVecAdd(l.routed_up, s.moe, s.shared, s.y, c.n_embd, c.n_latent, n,
                     ex, error);
  }
  // One-token decoding: one token per launch.
  for (std::uint32_t t = 0; t < n; ++t) {
    const float* x = s.latent + std::size_t{t} * c.n_latent;
    const std::int32_t* ids = s.ids + std::size_t{t} * used;
    const std::size_t at = std::size_t{t} * used * f;
    HY4_CHECK(hy4_mmq_moe_vec_typed(l.gate_exps.type, l.gate_exps.data, x, ids, s.ffn_a + at, static_cast<int>(f),
                                    static_cast<int>(c.n_latent), 1, static_cast<int>(c.n_expert),
                                    static_cast<int>(used), st));
    HY4_CHECK(hy4_mmq_moe_vec_typed(l.up_exps.type, l.up_exps.data, x, ids, s.ffn_b + at, static_cast<int>(f),
                                    static_cast<int>(c.n_latent), 1, static_cast<int>(c.n_expert),
                                    static_cast<int>(used), st));
  }
  HY4_CHECK(hy4_swiglu(s.ffn_a, s.ffn_b, s.route_w, s.ffn_c, n * used, f, c.swiglu_clamp[il], st));
  // Down: every slot has its own input, so each slot is a one-expert row.
  HY4_CHECK(hy4_mmq_moe_vec_typed(l.down_exps.type, l.down_exps.data, s.ffn_c, s.ids, s.ffn_a,
                                  static_cast<int>(c.n_latent), static_cast<int>(f), static_cast<int>(n * used),
                                  static_cast<int>(c.n_expert), 1, st));
  HY4_CHECK(hy4_sum_slots(s.ffn_a, s.moe, n, used, c.n_latent, st));
  return MatVecAdd(l.routed_up, s.moe, s.shared, s.y, c.n_embd, c.n_latent, n,
                   ex, error);
}

}  // namespace gufo::models::hy4_flash
