#ifndef HY4FLASH_RUNTIME_HY4_INTERNAL_HPP_
#define HY4FLASH_RUNTIME_HY4_INTERNAL_HPP_

#include <cstdint>
#include <string>

#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

/// Device scratch for one evaluation of up to Model::MaxChunk() tokens.
struct Model::Scratch {
  // Inputs.
  std::int32_t* tokens = nullptr;  // [chunk]
  std::int32_t* pos = nullptr;     // [chunk]
  std::int32_t* groups = nullptr;  // [chunk] complete kpool groups
  std::int32_t* group_pos =
      nullptr;  // their first member's (temporal) rope position
  std::int32_t* group_hw =
      nullptr;  // [chunk][2], with images: its height and width ones
  std::int32_t* n_pool = nullptr;  // [chunk]
  // Rope positions: temporal [chunk], and height | width [chunk][2] when the
  // pass has image rows (rope_3d).
  std::int32_t* rope = nullptr;
  std::int32_t* rope_hw = nullptr;
  bool rope_3d = false;
  // The pass takes the batched (prompt) kernels.
  bool batched = false;
  // Engram rows: pinned host staging and device copies.
  std::uint8_t* engram_host = nullptr;  // [modules][chunk][cols][d]
  float* engram_valid_host = nullptr;   // [modules][chunk][cols]
  float* engram_any_host = nullptr;     // [modules][chunk]
  std::uint8_t* engram_rows = nullptr;
  float* engram_valid = nullptr;
  float* engram_any = nullptr;
  float* engram_normed = nullptr;  // [chunk][cols * d]
  float* engram_kv = nullptr;      // [chunk][(hc + 1) n_embd]
  // Residual streams and per-layer activations.
  float* h = nullptr;     // [chunk][hc][n_embd]
  float* post = nullptr;  // [chunk][hc]
  float* x = nullptr;     // [chunk][n_embd]
  float* cur = nullptr;   // [chunk][n_embd]
  std::uint16_t* cur_half = nullptr;  // cur in binary16, batched passes
  float* y = nullptr;     // [chunk][n_embd]
  // Attention.
  float* qr = nullptr;    // [chunk][q_lora]
  std::uint16_t* qr_half = nullptr;  // qr in binary16, batched passes
  float* q = nullptr;     // [chunk][heads][head_k]
  float* kva = nullptr;   // [chunk][kv_lora + n_rot]
  float* qfull = nullptr; // [chunk][heads][kv_lora + n_rot]
  float* att = nullptr;   // [chunk][heads][kv_lora]
  float* attv = nullptr;  // [chunk][heads][head_v]
  float* gate = nullptr;  // [chunk][heads][head_v]
  float* attn_work = nullptr;  // decode attention, hy4_mla_attention_work
  std::uint64_t attn_work_floats = 0;
  // Indexer.
  float* ik = nullptr;     // [chunk][idx_dim]
  float* ivg = nullptr;    // [chunk][2 idx_dim]
  float* iq = nullptr;     // [chunk][idx_heads][idx_dim]
  float* iw = nullptr;     // [chunk][idx_heads]
  // FFN.
  float* ffn_a = nullptr;   // [chunk * used][max(n_ff_dense, n_ff_exp)]
  float* ffn_b = nullptr;
  float* ffn_c = nullptr;
  float* latent = nullptr;  // [chunk][n_latent]
  float* logits_r = nullptr;  // [chunk][n_expert]
  std::int32_t* ids = nullptr;   // [chunk][used]
  std::int32_t* ids_host = nullptr;  // pinned copy, batched passes
  // Routed plan (hy4_f16_routed_plan): rows [chunk * used], bounds
  // [n_expert + 1], tiles [chunk * used / 128 + n_expert], back to back.
  std::int32_t* route_plan = nullptr;
  std::int32_t* route_plan_host = nullptr;  // pinned copy
  float* route_w = nullptr;      // [chunk][used]
  float* moe = nullptr;          // [chunk][n_latent]
  float* shared = nullptr;       // [chunk][n_embd]
  // Head.
  float* logits = nullptr;  // [chunk][n_vocab]
  float* host_h = nullptr;  // pinned [chunk][hc][n_embd], layer probe
};

/// Where device work runs: the stream and a hipBLAS handle bound to it.
struct Exec {
  hipStream_t stream;
  void* blas_handle;
  bool prompt = false;  // a prompt pass: GEMMs at any token count
};

/// y[t] = W x[t] for any weight type the model uses (n tokens, W [m][k]):
/// matvecs up to Model::kVecMax tokens outside prompt passes, GEMMs beyond.
/// xh, if given, is x rounded to binary16, which the FP16 kernels read
/// instead of rounding x as they stage it (the same results, half the bytes).
bool MatVec(const Tensor& w, const float* x, float* y, std::uint32_t m,
            std::uint32_t k, std::uint32_t n, const Exec& ex,
            std::string* error, const std::uint16_t* xh = nullptr);

/// A batched y = W x on the FP16 kernels (hy4_f16_dense, io as there) for a
/// quantized W [m][k] with k % 256 == 0 and m % 16 == 0; fails otherwise.
bool DenseF16(const Tensor& w, const void* x, void* y, std::uint32_t m,
              std::uint32_t k, std::uint32_t n, std::uint32_t io,
              const Exec& ex, std::string* error);

/// Two MatVecs over the same x; a batched pass of two weights of one quantized
/// type quantizes x once. Results equal two MatVec calls.
bool MatVecPair(const Tensor& w1, float* y1, std::uint32_t m1, const Tensor& w2,
                float* y2, std::uint32_t m2, const float* x, std::uint32_t k,
                std::uint32_t n, const Exec& ex, std::string* error,
                const std::uint16_t* xh = nullptr);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_INTERNAL_HPP_
