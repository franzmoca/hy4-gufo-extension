#ifndef HY4FLASH_RUNTIME_HY4_CONFIG_HPP_
#define HY4FLASH_RUNTIME_HY4_CONFIG_HPP_

// The Hy4 Flash trunk's configuration, read from its GGUF metadata. The
// loading that follows it is in runtime/hy4_model.cpp.

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <string_view>
#include <vector>

namespace gufo::models::hy4_flash {

struct Config {
  /// Transformer layers this engine loads and runs: `block_count` without the
  /// draft layers that follow the backbone (n_nextn).
  std::uint32_t n_layer = 0;
  /// Draft (`nextn`) layers the quantizer wrote after the backbone, at
  /// indices n_layer .. n_layer + n_nextn - 1. DFly2 has its own checkpoint,
  /// so their tensors stay on disk.
  std::uint32_t n_nextn = 0;
  std::uint32_t n_embd = 0;
  std::uint32_t n_vocab = 0;
  std::uint32_t n_ff_dense = 0;
  std::uint32_t n_dense_lead = 0;
  std::vector<std::uint32_t> n_head;  // per layer
  std::vector<bool> is_swa;           // per layer
  std::vector<bool> has_indexer;      // per layer
  std::uint32_t q_lora = 0;
  std::uint32_t kv_lora = 0;
  std::uint32_t head_k = 0;  // nope + rope (256)
  std::uint32_t head_v = 0;  // 256
  std::uint32_t n_rot = 0;   // 64
  float rope_base = 0.0f;
  float rope_base_swa = 0.0f;
  std::uint32_t swa_window = 0;
  float rms_eps = 0.0f;
  // MoE
  std::uint32_t n_expert = 0;
  std::uint32_t n_expert_used = 0;
  std::uint32_t n_ff_exp = 0;
  std::uint32_t n_latent = 0;
  float expert_scale = 1.0f;
  bool expert_norm = true;
  std::vector<float> swiglu_clamp;  // per layer
  // Hyper-connections
  std::uint32_t hc = 0;
  float hc_eps = 0.0f;
  float hc_magnitude = 0.0f;
  // Indexer
  std::uint32_t idx_heads = 0;
  std::uint32_t idx_dim = 0;
  std::uint32_t idx_top_k = 0;
  std::uint32_t kpool = 0;
  // Engram
  std::vector<std::uint32_t> engram_layers;
  std::uint32_t engram_ngram = 0;
  std::uint32_t engram_heads_per_ngram = 0;
  std::uint32_t engram_dim = 0;
  std::vector<std::uint64_t> engram_mult;
  std::vector<std::uint32_t> engram_offsets;
  std::vector<std::uint32_t> engram_primes;
  std::vector<std::int64_t> engram_token_map;
  std::uint32_t engram_pad = 0;

  [[nodiscard]] std::uint32_t EngramCols() const {
    return (engram_ngram - 1) * engram_heads_per_ngram;
  }
  [[nodiscard]] int EngramIndex(std::uint32_t layer) const;
};

/// Reads the `hy_v4_flash` metadata of a trunk GGUF, or reports why it cannot
/// run this artifact. `error` is optional.
[[nodiscard]] bool ReadConfig(const core::GgufReader& r, Config* c,
                              std::string* error);

/// The tensors of the backbone and of the model around it, which the loader
/// uploads. The draft layers' tensors (6.97 GiB of the Xylolabs trunk) are
/// not: they would sit resident and never be read.
[[nodiscard]] bool BackboneTensor(std::string_view name, const Config& c);

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_CONFIG_HPP_
