// ReadConfig over an in-memory, metadata-only GGUF: which layers of a trunk
// this engine runs, and which of its tensors the loader is left to upload.
// A quant that ships a draft layer counts it in block_count (the Xylolabs
// Q4_K_M has 43 blocks for a 42-layer backbone), so the backbone and its
// per-layer arrays have to agree without it. No model, no GPU.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "runtime/hy4_config.hpp"

namespace {

namespace hy4 = gufo::models::hy4_flash;
using gufo::core::GgufReader;

int g_failures = 0;

void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    ++g_failures;
  }
}

using Value = std::variant<std::uint32_t, float, bool, std::string,
                           std::vector<std::uint32_t>, std::vector<bool>,
                           std::vector<float>, std::vector<std::uint64_t>>;
using Metadata = std::map<std::string, Value>;

// A GGUF v3 image of these metadata entries, and no tensors.
std::vector<std::uint8_t> Image(const Metadata& fields) {
  std::vector<std::uint8_t> b;
  const auto pod = [&]<typename T>(T v) {
    const std::size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
  };
  const auto text = [&](const std::string& s) {
    pod(static_cast<std::uint64_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
  };
  pod(std::uint32_t{0x46554747});  // "GGUF"
  pod(std::uint32_t{3});
  // No tensors: ReadConfig reads metadata only.
  pod(std::uint64_t{0});
  pod(static_cast<std::uint64_t>(fields.size()));
  for (const auto& [key, value] : fields) {
    text(key);
    std::visit(
        [&](const auto& item) {
          using T = std::decay_t<decltype(item)>;
          if constexpr (std::is_same_v<T, std::uint32_t>) {
            pod(std::uint32_t{4});
            pod(item);
          } else if constexpr (std::is_same_v<T, float>) {
            pod(std::uint32_t{6});
            pod(item);
          } else if constexpr (std::is_same_v<T, bool>) {
            pod(std::uint32_t{7});
            pod(static_cast<std::uint8_t>(item ? 1 : 0));
          } else if constexpr (std::is_same_v<T, std::string>) {
            pod(std::uint32_t{8});
            text(item);
          } else {
            using E = typename T::value_type;
            constexpr std::uint32_t element = std::is_same_v<E, std::uint32_t>
                                                  ? 4U
                                              : std::is_same_v<E, float> ? 6U
                                              : std::is_same_v<E, bool>  ? 7U
                                                                         : 10U;
            pod(std::uint32_t{9});
            pod(std::uint32_t{element});
            pod(static_cast<std::uint64_t>(item.size()));
            for (const auto e : item) {
              if constexpr (std::is_same_v<E, bool>)
                pod(static_cast<std::uint8_t>(e ? 1 : 0));
              else
                pod(e);
            }
          }
        },
        value);
  }
  b.resize((b.size() + 31) / 32 * 32);  // the data section's alignment
  return b;
}

// The sparse layers of the shipped backbone: 0, 1, then every fourth.
bool Sparse(std::uint32_t layer) {
  return layer < 2 || (layer >= 5 && (layer - 1) % 4 == 0);
}

// What a per-layer array holds for a draft layer, so that reading one of
// those entries as a backbone layer's fails the checks below.
constexpr std::uint32_t kDraftHeads = 128;
constexpr float kDraftClamp = 7.0f;

/// The trunk's metadata. `blocks` counts the backbone's layers and the
/// `drafts` that follow them; `indexers` is how many entries the indexer's
/// types array holds, which the quantizer writes per backbone layer.
Metadata Trunk(std::uint32_t blocks, std::uint32_t drafts,
               std::uint32_t indexers) {
  std::vector<std::uint32_t> heads;
  std::vector<bool> swa;
  std::vector<float> clamp;
  for (std::uint32_t i = 0; i < blocks; ++i) {
    const bool draft = i + drafts >= blocks;
    heads.push_back(draft ? kDraftHeads : (Sparse(i) ? 32U : 64U));
    swa.push_back(draft ? true : !Sparse(i));
    clamp.push_back(draft ? kDraftClamp : 10.0f);
  }
  std::vector<bool> idx;
  for (std::uint32_t i = 0; i < indexers; ++i)
    idx.push_back(Sparse(i));

  const std::string p = "hy_v4_flash.";
  Metadata m{{p + "block_count", blocks},
             {p + "embedding_length", 3072U},
             {p + "vocab_size", 120832U},
             {p + "feed_forward_length", 13824U},
             {p + "leading_dense_block_count", 1U},
             {p + "attention.q_lora_rank", 1024U},
             {p + "attention.kv_lora_rank", 512U},
             {p + "attention.key_length_mla", 256U},
             {p + "attention.value_length_mla", 256U},
             {p + "attention.head_count", heads},
             {p + "attention.sliding_window", 512U},
             {p + "attention.sliding_window_pattern", swa},
             {p + "attention.indexer.head_count", 16U},
             {p + "attention.indexer.key_length", 128U},
             {p + "attention.indexer.top_k", 2048U},
             {p + "attention.indexer.kpool", 4U},
             {p + "attention.indexer.kpool_select_tail", true},
             {p + "attention.indexer.types", idx},
             {p + "attention.layer_norm_rms_epsilon", 1e-5f},
             {p + "hyper_connection.count", 4U},
             {p + "hyper_connection.epsilon", 1e-6f},
             {p + "hyper_connection.magnitude", 2.0f},
             {p + "rope.dimension_count", 64U},
             {p + "rope.freq_base", 1e7f},
             {p + "rope.freq_base_swa", 1e4f},
             {p + "expert_count", 512U},
             {p + "expert_used_count", 16U},
             {p + "expert_feed_forward_length", 1536U},
             {p + "expert_latent_length", 1536U},
             {p + "expert_weights_scale", 3.998f},
             {p + "expert_weights_norm", true},
             {p + "expert_gating_func", 2U},
             {p + "swiglu_clamp_exp", clamp},
             {"general.architecture", std::string("hy_v4_flash")}};
  if (drafts != 0)
    m[p + "nextn_predict_layers"] = drafts;
  return m;
}

hy4::Config Parse(const Metadata& fields, std::string* error, bool* ok) {
  const auto image = Image(fields);
  std::string open_error;
  const auto reader =
      GgufReader::OpenMemory(image.data(), image.size(), &open_error);
  if (!reader) {
    std::fprintf(stderr, "FAIL fixture: %s\n", open_error.c_str());
    ++g_failures;
    *ok = false;
    return {};
  }
  hy4::Config c;
  *ok = hy4::ReadConfig(*reader, &c, error);
  return c;
}

void Check(const char* what, const Metadata& fields, bool want_ok) {
  std::string error;
  bool ok = false;
  const hy4::Config c = Parse(fields, &error, &ok);
  Expect(ok == want_ok, std::string(what) + (ok ? " accepted" : " rejected") +
                            (error.empty() ? "" : ": " + error));
  if (!want_ok || !ok)
    return;
  Expect(c.n_layer == 42, std::string(what) + ": 42 backbone layers");
  Expect(c.n_head.size() == 42 && c.is_swa.size() == 42 &&
             c.has_indexer.size() == 42 && c.swiglu_clamp.size() == 42,
         std::string(what) + ": one entry per backbone layer");
  for (std::uint32_t i = 0; i < c.n_layer; ++i) {
    Expect(c.n_head[i] == (Sparse(i) ? 32U : 64U),
           std::string(what) + ": heads of layer " + std::to_string(i));
    Expect(c.is_swa[i] != Sparse(i),
           std::string(what) + ": window of layer " + std::to_string(i));
    Expect(c.has_indexer[i] == Sparse(i),
           std::string(what) + ": indexer of layer " + std::to_string(i));
    Expect(c.swiglu_clamp[i] == 10.0f,
           std::string(what) + ": clamp of layer " + std::to_string(i));
  }
}

// The shipped cut, and trunks that describe the draft layer in every array,
// in none of them, or not at all.
void CheckLayerCounts() {
  Check("a trunk with a draft layer", Trunk(43, 1, 42), true);
  Check("a trunk whose arrays hold the draft layer too", Trunk(43, 1, 43),
        true);
  Check("a trunk without draft layers", Trunk(42, 0, 42), true);
  Check("an array shorter than the backbone", Trunk(43, 1, 40), false);
  Check("a block_count of draft layers alone", Trunk(1, 1, 1), false);

  std::string error;
  bool ok = false;
  Parse(Trunk(43, 1, 42), &error, &ok);
  Expect(ok, "the shipped trunk read");
}

// A draft layer's tensors are in the trunk but never read: they would sit
// resident beside the backbone.
void CheckBackboneTensors() {
  std::string error;
  bool ok = false;
  const hy4::Config c = Parse(Trunk(43, 1, 42), &error, &ok);
  Expect(ok, "backbone tensors: config read");
  for (const char* name :
       {"blk.0.attn_norm.weight", "blk.4.attn_gate.weight",
        "blk.9.indexer.attn_k.weight", "blk.41.ffn_up_exps.weight",
        "token_embd.weight", "output.weight", "output_hc_fn.weight",
        "per_layer_token_embd.weight", "blk.4x.weight", "blk.weight"}) {
    Expect(hy4::BackboneTensor(name, c), std::string("loaded: ") + name);
  }
  for (const char* name :
       {"blk.42.attn_norm.weight", "blk.42.ffn_gate_exps.weight",
        "blk.42.nextn.eh_proj.weight", "blk.43.attn_norm.weight",
        "blk.142.attn_norm.weight"}) {
    Expect(!hy4::BackboneTensor(name, c), std::string("not loaded: ") + name);
  }
}

// The Engram's metadata, to check that a lookup layer pointing at a draft
// layer is not accepted.
void CheckEngramLayers() {
  constexpr std::uint32_t kCols = 16;  // (3 - 1) grams times 8 heads
  Metadata m = Trunk(43, 1, 42);
  const std::string p = "hy_v4_flash.";
  m[p + "ple.ngram_size"] = 3U;
  m[p + "ple.heads_per_ngram"] = 8U;
  m[p + "embedding_length_per_layer_input"] = 96U;
  m[p + "ple.head_offsets"] = std::vector<std::uint32_t>(kCols, 0U);
  m[p + "ple.head_vocab_sizes"] = std::vector<std::uint32_t>(kCols, 1024U);
  m[p + "ple.layer_multipliers"] = std::vector<std::uint64_t>(3, 1U);
  m[p + "ple.layers"] = std::vector<std::uint32_t>{42};
  m[p + "engram.token_map"] = std::vector<std::uint32_t>{0, 1, 2, 3};
  m[p + "engram.pad_token_id"] = 1U;

  std::string error;
  bool ok = false;
  Parse(m, &error, &ok);
  Expect(!ok, "an Engram layer of a draft layer rejected");

  m[p + "ple.layers"] = std::vector<std::uint32_t>{1, 11, 21, 31};
  m[p + "ple.layer_multipliers"] = std::vector<std::uint64_t>(12, 1U);
  m[p + "ple.head_offsets"] = std::vector<std::uint32_t>(4 * kCols, 0U);
  m[p + "ple.head_vocab_sizes"] = std::vector<std::uint32_t>(4 * kCols, 1024U);
  const hy4::Config c = Parse(m, &error, &ok);
  Expect(ok, "the trunk's Engram layers accepted: " + error);
  Expect(c.EngramIndex(11) == 1 && c.EngramIndex(41) == -1,
         "Engram modules of the backbone");
}

}  // namespace

int main() {
  CheckLayerCounts();
  CheckBackboneTensors();
  CheckEngramLayers();
  if (g_failures) {
    std::fprintf(stderr, "%d failures\n", g_failures);
    return 1;
  }
  std::printf("hy4 trunk config: all checks passed\n");
  return 0;
}
