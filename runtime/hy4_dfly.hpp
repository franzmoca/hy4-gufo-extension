#ifndef HY4FLASH_RUNTIME_HY4_DFLY_HPP_
#define HY4FLASH_RUNTIME_HY4_DFLY_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

/// Tencent's DFly2 block drafter for Hy4 Flash (AngelSpec `dfly2`).
///
/// One pass drafts a block of kBlock slots at positions a .. a + kBlock - 1:
/// the anchor token (the one after the session's frontier a - 1) and mask
/// tokens. Each of its five MLA layers attends to the block and to the
/// target's latent rows of one sparse layer, at the positions that layer's
/// indexer selected for the frontier. The target's final hidden at the
/// frontier is fused into the anchor slot. Slot k >= 1 predicts the token at
/// a + k through the target's LM head, after a correction conditioned on the
/// token before it (the anchor, then the previous slot's draft).
class Drafter {
public:
  static constexpr std::uint32_t kBlock = 8;
  /// Sessions one DraftBatch drafts for.
  static constexpr std::uint32_t kMaxSessions = 8;

  ~Drafter();
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  /// Loads the `model.dfly_layers.*` tensors of a safetensors file (BF16,
  /// quantized to Q8_0 here) for `target`.
  [[nodiscard]] static std::unique_ptr<Drafter> Load(const std::string& path, Model& target,
                                                     std::string* error);

  /// The target layers whose selections and latent rows the drafter reads
  /// (Session::EnableTaps).
  [[nodiscard]] std::span<const std::uint32_t> TargetLayers() const { return target_layers_; }
  [[nodiscard]] std::uint64_t ResidentBytes() const { return resident_bytes_; }

  /// Drafts up to `count` (< kBlock) tokens after `anchor`, the token that
  /// follows the session's frontier, while the confidence head's estimate
  /// that the target accepts every draft so far stays at least `min_accept`.
  /// Needs the session's taps (TapsReady). May draft nothing.
  [[nodiscard]] bool Draft(const Session& session, std::int32_t anchor, std::uint32_t count, float min_accept,
                           std::vector<std::int32_t>* drafts, std::string* error);
  /// Draft for up to kMaxSessions distinct sessions at once: their blocks
  /// share one pass through the drafter's layers and each slot's
  /// correction and LM head, and each session drafts what Draft gives it
  /// alone, up to rounding in the layers' wider matrix products.
  [[nodiscard]] bool DraftBatch(
      std::span<const Session* const> sessions,
      std::span<const std::int32_t> anchors,
      std::span<const std::uint32_t> counts, float min_accept,
      std::span<std::vector<std::int32_t>* const> drafts, std::string* error);

private:
  struct LayerWeights;
  struct Buffers;
  explicit Drafter(Model& target);

  Model& target_;
  std::vector<std::uint32_t> target_layers_;
  std::vector<LayerWeights> layers_;
  Tensor fuse_enorm_, fuse_hnorm_, fuse_eh_, fuse_gate_;
  Tensor corr_hnorm_, corr_enorm_, corr_gate_, corr_up_, corr_down_;
  Tensor final_norm_, confidence_w_;
  // The target's LM head for drafts: a Q8_0 copy of a binary16 head (half the
  // bytes per draft), else the target's own.
  Tensor head_;
  // Drafts score the head's first head_rows_ rows (the BPE's earlier, more
  // frequent tokens) and a table of the later rows whose tokens the session
  // holds (SyncTable), in head_rows_ + ids.size() logits. Each drafting
  // session has its own table, up to kMaxSessions of them.
  struct Table {
    void* rows = nullptr;           // [table_capacity_] head rows
    std::vector<std::int32_t> ids;  // token of each table row
    std::vector<std::uint8_t> in;   // [n_vocab]
    std::size_t synced = 0;         // history tokens the table covers
    const Session* session = nullptr;
    std::uint64_t used = 0;  // the last DraftBatch that used it
  };
  std::uint32_t head_rows_ = 0;
  std::uint64_t row_bytes_ = 0;
  std::uint32_t table_capacity_ = 0;
  std::vector<Table> tables_;
  std::uint64_t batches_ = 0;
  // The session's table: its own, else a new one, else the least recently
  // used one not in this batch, reset; null when every table is in it.
  [[nodiscard]] Table* TableFor(const Session& session);
  [[nodiscard]] bool AddTable();
  [[nodiscard]] bool SyncTable(Table& table, const Session& session,
                               std::string* error);
  float confidence_b_ = 0.0f;
  std::int32_t mask_token_ = 0;
  std::unique_ptr<Buffers> buf_;
  std::vector<void*> allocations_;
  struct HostIo;
  HostIo* io_ = nullptr;  // pinned
  std::uint64_t resident_bytes_ = 0;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_RUNTIME_HY4_DFLY_HPP_
