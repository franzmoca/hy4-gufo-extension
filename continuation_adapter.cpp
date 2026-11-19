#include "continuation_adapter.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include <gufo/ext/snapshot_copy.hpp>

namespace gufo::models::hy4_flash {
namespace {

using cache::Rows;
using cache::TransferResult;

void Check(hipError_t result) {
  if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}

// Tokens and pooled keys are small rows; latent rows are 2304 bytes, so
// their chunks hold fewer.
constexpr Rows kSmallChunkRows = 2048;
constexpr Rows kLatentChunkRows = 256;

struct Metadata {
  std::uint32_t version{2};
  std::uint32_t position{};
  // Rope state after images (Session::RestoreRope): the text offset, and the
  // rope position of the open pool group's first member.
  std::int32_t rope_delta{};
  std::int32_t group_rope[3]{};
};
static_assert(std::is_trivially_copyable_v<Metadata>);

using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
// Loaded byte ranges cover [0, end) without a gap.
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier) return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}

}  // namespace

class ContinuationAdapter::State final : public cache::Slot {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard, std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->SetWriteHook(&State::OnWrite, this);
  }
  ~State() override {
    for (const auto& job : jobs) (void)job->Wait();
    BeforeRelease();
    session->SetWriteHook(nullptr, nullptr);
  }
  [[nodiscard]] Rows Tokens() const { return session->Position(); }
  bool IsValid() const noexcept override { return !restoring && !failed; }
  void RequireIdle() const {
    if (!IsValid() || pending) throw std::logic_error("Hy4 slot is not ready for execution");
  }
  static void OnWrite(void* data, std::uint32_t first, std::uint32_t end) {
    static_cast<State*>(data)->BeforeWrite(first, end);
  }
  void BeforeWrite(Rows first, Rows end) {
    if (restoring || failed || pending) throw std::logic_error("Hy4 slot is not ready for mutation");
    for (const auto& component : adapter.components_) {
      if (!cache::HasRows(component.descriptor.kind)) continue;
      // A row is written once its ratio tokens are complete.
      const Rows a = first / component.ratio, b = end / component.ratio;
      if (a < b) guard.BeforeOverwrite(component.descriptor.id, a, b);
    }
  }
  void GuardAll() {
    for (const auto& component : adapter.components_) {
      if (cache::HasRows(component.descriptor.kind)) {
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.Capacity(component));
      }
    }
  }
  void BeforeRelease() noexcept {
    for (const auto& component : adapter.components_) {
      if (cache::HasRows(component.descriptor.kind)) {
        guard.BeforeRelease(component.descriptor.id, 0, adapter.Capacity(component));
      }
    }
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed) owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  // During a restore: the token buffer, positions and loaded byte ranges.
  std::int32_t* restore_tokens{};
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  Metadata metadata{};
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model, std::uint32_t context,
                                         cache::Identity identity, std::vector<std::uint32_t> tap_layers)
    : model_(std::move(model)), context_(context), identity_(std::move(identity)), tap_layers_(std::move(tap_layers)) {
  if (!model_ || identity_.empty() || context < 2) throw std::invalid_argument("invalid Hy4 adapter configuration");
  const Config& c = model_->config();
  const auto add = [&](ContinuationPart part, std::uint32_t layer, cache::ComponentDescriptor d,
                       std::uint32_t ratio = 1) {
    d.id = {static_cast<std::uint32_t>(components_.size() + 1)};
    d.layout_version = 1;
    components_.push_back({d, part, layer, ratio});
  };
  add(ContinuationPart::kTokens, 0,
      {.id = {}, .kind = cache::ComponentKind::kAppendRows, .row_bytes = sizeof(std::int32_t),
       .rows_per_chunk = kSmallChunkRows});
  const std::size_t latent_bytes = std::size_t{c.kv_lora + c.n_rot} * sizeof(float);
  std::size_t ring_bytes = 0;
  for (std::uint32_t l = 0; l < c.n_layer; ++l) {
    if (c.is_swa[l]) {
      ring_bytes +=
          std::size_t{std::min(c.swa_window, model_->KvRing(l, context))} *
          latent_bytes;
    } else {
      add(ContinuationPart::kRows, l,
          {.id = {}, .kind = cache::ComponentKind::kAppendRows, .row_bytes = latent_bytes, .rows_per_chunk = kLatentChunkRows});
    }
    if (c.has_indexer[l]) {
      add(ContinuationPart::kPools, l,
          {.id = {}, .kind = cache::ComponentKind::kAppendRows, .row_bytes = std::size_t{c.idx_dim} * sizeof(float),
           .rows_per_chunk = kSmallChunkRows},
          c.kpool);
      ring_bytes += std::size_t{c.kpool} * 2 * c.idx_dim * sizeof(float);
    }
  }
  add(ContinuationPart::kRings, 0, {.id = {}, .kind = cache::ComponentKind::kPrivateState, .state_bytes = ring_bytes});
  add(ContinuationPart::kLogits, 0,
      {.id = {}, .kind = cache::ComponentKind::kPrivateState, .state_bytes = std::size_t{c.n_vocab} * sizeof(float)});
  add(ContinuationPart::kMetadata, 0, {.id = {}, .kind = cache::ComponentKind::kPrivateState, .state_bytes = sizeof(Metadata)});
  for (const auto& component : components_) descriptors_.push_back(component.descriptor);
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Component layout version.
  put(context);
}

ContinuationAdapter::~ContinuationAdapter() = default;

ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this) throw std::invalid_argument("slot belongs to another Hy4 adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
const ContinuationComponent& ContinuationAdapter::Component(cache::ComponentId id) const {
  if (!id.value || id.value > components_.size()) throw std::invalid_argument("invalid Hy4 component ID");
  return components_[id.value - 1];
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
Rows ContinuationAdapter::Capacity(const ContinuationComponent& c) const { return context_ / c.ratio + 1; }
std::vector<cache::ComponentPosition> ContinuationAdapter::PositionsAt(Rows tokens) const {
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& c : components_) positions.push_back({c.descriptor.id, tokens / c.ratio});
  return positions;
}
std::vector<cache::BackingClass> ContinuationAdapter::BackingPlan(std::size_t ram) const {
  // Rows, private states and metadata share the same committed pages; a
  // payload takes a contiguous page group and charges all of its slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header) return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks) return {};
  return {{page, blocks, header + blocks * per_block}};
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components() const { return descriptors_; }
cache::Identity ContinuationAdapter::CompatibilityIdentity() const { return identity_; }
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(context_, &error);
  if (!session) throw std::runtime_error(error);
  if (!tap_layers_.empty() && !session->EnableTaps(tap_layers_, &error)) throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid()) throw std::logic_error("cannot query an invalid Hy4 slot");
  return PositionsAt(state.Tokens());
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot, Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_) throw std::invalid_argument("invalid Hy4 prefill plan");
  // Session::Evaluate runs passes of kMaxChunk tokens from the frontier.
  std::vector<Rows> result;
  while (first < end) {
    first += std::min<Rows>(end - first, Model::kMaxChunk);
    result.push_back(first);
  }
  return result;
}
void* ContinuationAdapter::RowData(const State& state, const ContinuationComponent& c) const {
  switch (c.part) {
    case ContinuationPart::kRows:
      return state.session->KvRows(c.layer);
    case ContinuationPart::kPools:
      return state.session->IndexerPools(c.layer);
    default:
      throw std::invalid_argument("component has no device rows");
  }
}
std::vector<ContinuationAdapter::Piece> ContinuationAdapter::PrivateRegion(State& state,
                                                                           const ContinuationComponent& c) const {
  const Config& cfg = model_->config();
  const auto total = c.descriptor.state_bytes;
  std::vector<Piece> pieces;
  std::size_t offset = 0;
  const auto add = [&](void* pointer, std::size_t bytes, bool host = false) {
    pieces.push_back({pointer, offset, bytes, host});
    offset += bytes;
  };
  switch (c.part) {
    case ContinuationPart::kRings: {
      // Only the rows a continuation reads: each window layer's last `window`
      // positions before the frontier, and the indexer rows of the open kpool
      // group. Position p sits in ring row p % ring; a span wraps at most once.
      // (Positions before 0, at the start of a short prompt, are rows no pass
      // reads.)
      const auto Tokens = static_cast<std::int64_t>(state.Tokens());
      const auto span = [&](float* base, std::uint32_t ring, std::int64_t first,
                            std::uint32_t count, std::size_t row_bytes) {
        const auto start =
            static_cast<std::uint32_t>(((first % ring) + ring) % ring);
        const std::uint32_t run = std::min(count, ring - start);
        add(reinterpret_cast<std::uint8_t*>(base) +
                std::size_t{start} * row_bytes,
            std::size_t{run} * row_bytes);
        if (run < count)
          add(base, std::size_t{count - run} * row_bytes);
      };
      const std::size_t latent_bytes = std::size_t{cfg.kv_lora + cfg.n_rot} * sizeof(float);
      const std::size_t index_bytes =
          std::size_t{2} * cfg.idx_dim * sizeof(float);
      for (std::uint32_t l = 0; l < cfg.n_layer; ++l) {
        if (cfg.is_swa[l]) {
          const std::uint32_t ring = state.session->KvRing(l);
          const std::uint32_t count = std::min(cfg.swa_window, ring);
          span(state.session->KvRows(l), ring, Tokens - count, count,
               latent_bytes);
        }
        if (cfg.has_indexer[l]) {
          span(state.session->IndexerRing(l), Model::kIndexerRing,
               Tokens - Tokens % cfg.kpool, cfg.kpool, index_bytes);
        }
      }
      break;
    }
    case ContinuationPart::kLogits: {
      auto& logits = state.session->LastLogits();
      if (logits.size() != cfg.n_vocab) logits.assign(cfg.n_vocab, 0.0f);
      add(logits.data(), total, true);
      break;
    }
    case ContinuationPart::kMetadata:
      add(&state.metadata, total, true);
      break;
    default:
      throw std::invalid_argument("component is not private state");
  }
  if (offset != total) throw std::logic_error("Hy4 private state differs from layout");
  return pieces;
}
cache::Completion ContinuationAdapter::Transfer(State& state, bool load, cache::Stream& stream,
                                                const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  if (!load) Check(hipStreamSynchronize(model_->stream()));
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs, [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load) state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}
void ContinuationAdapter::BeginRestore(cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (state.restoring || state.failed || state.pending) {
    throw std::logic_error("reset failed restores and settle transfers before BeginRestore");
  }
  if (positions.size() != descriptors_.size()) throw std::invalid_argument("incomplete Hy4 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    if (std::ranges::count(positions, d.id, &cache::ComponentPosition::id) != 1) {
      throw std::invalid_argument("missing or duplicate Hy4 frontier");
    }
    ordered.push_back(*std::ranges::find(positions, d.id, &cache::ComponentPosition::id));
  }
  const auto n = ordered.front().valid_rows;
  if (!n || n >= context_ || ordered != PositionsAt(n)) {
    throw std::invalid_argument("inconsistent Hy4 component frontiers");
  }
  // The engine's stream may still read the old rows.
  Check(hipStreamSynchronize(model_->stream()));
  state.GuardAll();
  state.restore_tokens = state.session->BeginRestore(static_cast<std::uint32_t>(n));
  state.expected = std::move(ordered);
  state.coverage.assign(descriptors_.size(), {});
  state.metadata = {};
  state.restoring = true;
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot, cache::ComponentId id, Rows first,
                                                   Rows end, std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& c = Component(id);
  if (!state.IsValid()) throw std::logic_error("cannot read an invalid Hy4 slot");
  const auto row = c.descriptor.row_bytes;
  if (!cache::HasRows(c.descriptor.kind) || first > end || end > state.Tokens() / c.ratio ||
      bytes.size() != (end - first) * row) {
    throw std::invalid_argument("invalid Hy4 row capture");
  }
  const std::byte* source = nullptr;
  if (!bytes.empty()) {
    source = c.part == ContinuationPart::kTokens
                 ? reinterpret_cast<const std::byte*>(state.session->History().data())
                 : static_cast<const std::byte*>(RowData(state, c));
    source += first * row;
  }
  return Transfer(state, false, stream, [&](gufo::hip::TransferStream& transfer) {
    if (bytes.empty()) return;
    if (c.part == ContinuationPart::kTokens) {
      std::memcpy(bytes.data(), source, bytes.size());
    } else {
      Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(), transfer.Native()));
    }
  });
}
std::span<const std::byte> ContinuationAdapter::BorrowRowsView(const cache::Slot& slot, cache::ComponentId id,
                                                               Rows first, Rows end) const {
  const auto& state = As(slot);
  const auto& c = Component(id);
  // The host token vector can move when execution appends; device rows stay
  // fixed for the session and are guarded before every write.
  if (c.part != ContinuationPart::kRows && c.part != ContinuationPart::kPools) return {};
  if (!state.IsValid() || first >= end || end > state.Tokens() / c.ratio) {
    throw std::invalid_argument("invalid Hy4 borrowed range");
  }
  const auto row = c.descriptor.row_bytes;
  return {static_cast<const std::byte*>(RowData(state, c)) + first * row, (end - first) * row};
}
cache::Completion ContinuationAdapter::CopyRowsIn(cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
                                                  std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) throw std::logic_error("Hy4 load requires an unfailed restore");
  try {
    const auto& c = Component(id);
    const auto row = c.descriptor.row_bytes;
    if (!cache::HasRows(c.descriptor.kind) || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row) {
      throw std::invalid_argument("invalid Hy4 row restore");
    }
    std::byte* destination = nullptr;
    if (!bytes.empty()) {
      destination = (c.part == ContinuationPart::kTokens ? reinterpret_cast<std::byte*>(state.restore_tokens)
                                                         : static_cast<std::byte*>(RowData(state, c))) +
                    first * row;
    }
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(state, true, stream, [&](gufo::hip::TransferStream& transfer) {
      if (bytes.empty()) return;
      if (c.part == ContinuationPart::kTokens) {
        std::memcpy(destination, bytes.data(), bytes.size());
      } else {
        Check(hip::CopySnapshotToDevice(destination, bytes.data(), bytes.size(), transfer.Native()));
      }
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CapturePrivate(const cache::Slot& slot, cache::ComponentId id,
                                                      std::span<std::byte> bytes, cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes) {
    throw std::invalid_argument("incorrect Hy4 private buffer size");
  }
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(const cache::Slot& slot, cache::ComponentId id,
                                                           std::size_t offset, std::span<std::byte> bytes,
                                                           cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid() || state.Tokens() == 0) throw std::logic_error("cannot capture an empty or invalid Hy4 slot");
  const auto& c = Component(id);
  const auto total = c.descriptor.state_bytes;
  if (c.descriptor.kind != cache::ComponentKind::kPrivateState || offset > total || bytes.size() > total - offset) {
    throw std::invalid_argument("invalid Hy4 private piece");
  }
  Metadata metadata{.position = static_cast<std::uint32_t>(state.Tokens())};
  if (c.part == ContinuationPart::kMetadata) {
    const auto group_first =
        metadata.position / model_->config().kpool * model_->config().kpool;
    metadata.rope_delta = state.session->RopeDelta();
    if (group_first < metadata.position) {
      const auto rope = state.session->RopeAt(group_first);
      std::copy(rope.begin(), rope.end(), metadata.group_rope);
    }
  }
  auto pieces = PrivateRegion(state, c);
  if (c.part == ContinuationPart::kMetadata) pieces = {{&metadata, 0, total, true}};
  return Transfer(state, false, stream, [&](gufo::hip::TransferStream& transfer) {
    for (const auto& piece : pieces) {
      const auto first = std::max(offset, piece.offset);
      const auto end = std::min(offset + bytes.size(), piece.offset + piece.bytes);
      if (first >= end) continue;
      const auto* source = static_cast<const std::byte*>(piece.pointer) + first - piece.offset;
      auto* destination = bytes.data() + first - offset;
      if (piece.host) {
        std::memcpy(destination, source, end - first);
      } else {
        Check(hip::CopySnapshotToHost(destination, source, end - first, transfer.Native()));
      }
    }
  });
}
cache::Completion ContinuationAdapter::LoadPrivate(cache::Slot& slot, cache::ComponentId id,
                                                   std::span<const std::byte> bytes, cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes) {
    As(slot).failed = true;
    throw std::invalid_argument("incorrect Hy4 private buffer size");
  }
  return LoadPrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(cache::Slot& slot, cache::ComponentId id, std::size_t offset,
                                                        std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) throw std::logic_error("Hy4 load requires an unfailed restore");
  try {
    const auto& c = Component(id);
    const auto total = c.descriptor.state_bytes;
    if (c.descriptor.kind != cache::ComponentKind::kPrivateState || offset > total || bytes.size() > total - offset) {
      throw std::invalid_argument("invalid Hy4 private piece");
    }
    const auto pieces = PrivateRegion(state, c);
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(state, true, stream, [&](gufo::hip::TransferStream& transfer) {
      for (const auto& piece : pieces) {
        const auto first = std::max(offset, piece.offset);
        const auto end = std::min(offset + bytes.size(), piece.offset + piece.bytes);
        if (first >= end) continue;
        auto* destination = static_cast<std::byte*>(piece.pointer) + first - piece.offset;
        const auto* source = bytes.data() + first - offset;
        if (piece.host) {
          std::memcpy(destination, source, end - first);
        } else {
          Check(hip::CopySnapshotToDevice(destination, source, end - first, transfer.Native()));
        }
      }
    });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
bool ContinuationAdapter::Validate(cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring) throw std::logic_error("Hy4 validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending || positions.size() != state.expected.size()) return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1) return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes ? expected.valid_rows * descriptor.row_bytes : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total)) return refuse();
  }
  const auto n = state.expected.front().valid_rows;
  if (state.metadata.version != Metadata{}.version || state.metadata.position != n) return refuse();
  const auto group_first = static_cast<std::uint32_t>(
      n / model_->config().kpool * model_->config().kpool);
  const auto& m = state.metadata;
  state.session->RestoreRope(
      m.rope_delta, group_first,
      {m.group_rope[0], m.group_rope[1], m.group_rope[2]});
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    if (state.pending) return false;
    resetting = &state;
    state.BeforeRelease();
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.restore_tokens = nullptr;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting) resetting->failed = true;
    return false;
  }
}

}  // namespace gufo::models::hy4_flash
