#ifndef HY4FLASH_CONTINUATION_ADAPTER_HPP_
#define HY4FLASH_CONTINUATION_ADAPTER_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include <gufo/ext/adapter.hpp>
#include <gufo/ext/transfer_pool.hpp>

#include "runtime/hy4_runtime.hpp"

namespace gufo::models::hy4_flash {

// One continuation's state, by component (ID = index + 1).
enum class ContinuationPart : std::uint8_t {
  kTokens,
  kRows,    // a full-attention layer's latent rows
  kPools,   // an indexer layer's pooled keys, one row per kpool tokens
  kRings,   // the sliding-window and indexer value rings of all layers
  kLogits,
  kMetadata,
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
  std::uint32_t ratio{1};  // tokens per row
};

// Serves Hy4 Flash sessions to the component cache. Tokens, the latent rows
// of the full-attention layers and the indexers' pooled keys are append rows
// (the device rows can be lent); the rings, the frontier logits and metadata
// are private components.
class ContinuationAdapter final : public cache::Adapter {
public:
  /// `tap_layers`: the layers each session taps for a drafter (Session::EnableTaps).
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context, cache::Identity,
                      std::vector<std::uint32_t> tap_layers = {});
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(std::size_t ram_bytes) const override;
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components() const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(const cache::Slot&, cache::Rows first,
                                                     cache::Rows end) const override;
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId, std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&, cache::ComponentId, std::size_t,
                                                      std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId, std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&, cache::ComponentId, std::size_t,
                                                   std::span<const std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId, cache::Rows first,
                                              cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] std::span<const std::byte> BorrowRowsView(const cache::Slot&, cache::ComponentId, cache::Rows,
                                                          cache::Rows) const override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>, cache::Stream&) override;
  [[nodiscard]] bool Validate(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(cache::ComponentId) const;
  [[nodiscard]] std::vector<cache::ComponentPosition> PositionsAt(cache::Rows tokens) const;
  [[nodiscard]] cache::Rows Capacity(const ContinuationComponent&) const;
  [[nodiscard]] void* RowData(const State&, const ContinuationComponent&) const;
  [[nodiscard]] std::vector<Piece> PrivateRegion(State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Completion Transfer(State&, bool load, cache::Stream&,
                                           const std::function<void(gufo::hip::TransferStream&)>&);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  std::vector<std::uint32_t> tap_layers_;
};

}  // namespace gufo::models::hy4_flash

#endif  // HY4FLASH_CONTINUATION_ADAPTER_HPP_
