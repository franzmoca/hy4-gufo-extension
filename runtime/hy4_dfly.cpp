// DFly2 block drafter (AngelSpec `dfly2`, Hy4 Flash): see hy4_dfly.hpp.
// Reference: angelspec/models/draft/{dflash.py,hy4_mla.py} and the
// training-side block layout in angelspec/models/dflash.py.

#include "runtime/hy4_dfly.hpp"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/managed_alloc.hpp>
#include <gufo/ext/json.hpp>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "kernels/rocm/hy4_kernels.h"
#include "runtime/hy4_internal.hpp"

namespace gufo::models::hy4_flash {

namespace {

constexpr std::uint32_t kHidden = 3072;
constexpr std::uint32_t kInter = 9216;
// Drafts score the head's first kHeadPrefix rows (the BPE's earlier, more
// frequent tokens) and up to kTableRows later rows whose tokens the session
// holds (53 MB): a slot reads 214 MB of the head plus its table instead of 394
// (E35).
constexpr std::uint32_t kHeadPrefix = 65536;
constexpr std::uint32_t kTableRows = 16384;
constexpr std::uint32_t kHeads = 32;
constexpr std::uint32_t kQLora = 1024;
constexpr std::uint32_t kKvLora = 512;
constexpr std::uint32_t kNope = 192;
constexpr std::uint32_t kRope = 64;
constexpr std::uint32_t kHeadK = kNope + kRope;
constexpr std::uint32_t kHeadV = 256;
constexpr std::uint32_t kLatent = kKvLora + kRope;
constexpr std::uint32_t kLayers = 5;
constexpr float kEps = 1e-5f;
// AngelSpec's config: the drafter's layer i reads the target's sparse layer
// kTargetLayers[i] (target_kv_layers = target_index_layer, 0-based).
constexpr std::uint32_t kTargetLayers[kLayers] = {25, 29, 33, 37, 41};
constexpr std::int32_t kMaskToken = 120009;

bool Fail(std::string* error, const std::string& what) {
  if (error) *error = what;
  return false;
}

#define HY4_CHECK(expr)                                                          \
  do {                                                                           \
    if ((expr) != 0) return Fail(error, std::string("hy4 dfly: failed: ") + #expr); \
  } while (0)

template <typename T>
bool DeviceAlloc(T** p, std::size_t count, std::vector<void*>* owned) {
  void* v = nullptr;
  const std::size_t bytes = std::max<std::size_t>(count, 1) * sizeof(T);
  if (gufo::hip::AllocateDevice(&v, bytes) != hipSuccess) return false;
  if (hipMemset(v, 0, bytes) != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
    (void)hipFree(v);
    return false;
  }
  owned->push_back(v);
  *p = static_cast<T*>(v);
  return true;
}

constexpr std::uint64_t kQ8BlockBytes = 34;
constexpr std::uint64_t kQ4KBlockBytes = 144;

float Bf16(std::uint16_t v) {
  const std::uint32_t bits = std::uint32_t{v} << 16;
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

// A mapped safetensors file.
class SafeTensors {
public:
  ~SafeTensors() {
    if (map_ != MAP_FAILED) munmap(map_, size_);
  }
  bool Open(const std::string& path, std::string* error) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return Fail(error, "hy4 dfly: cannot open " + path);
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size < 8) {
      close(fd);
      return Fail(error, "hy4 dfly: cannot read " + path);
    }
    size_ = static_cast<std::size_t>(st.st_size);
    map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map_ == MAP_FAILED) return Fail(error, "hy4 dfly: cannot map " + path);
    std::uint64_t header = 0;
    std::memcpy(&header, map_, 8);
    if (header > size_ - 8) return Fail(error, "hy4 dfly: bad safetensors header");
    data_ = static_cast<const std::uint8_t*>(map_) + 8 + header;
    try {
      header_ = json::parse(std::string_view(static_cast<const char*>(map_) + 8, header));
    } catch (const std::exception& e) {
      return Fail(error, std::string("hy4 dfly: bad safetensors header: ") + e.what());
    }
    return true;
  }
  // Pointer to raw BF16 bytes of tensor `name`, checking its shape.
  bool RawBf16(const std::string& name,
               std::initializer_list<std::uint64_t> shape,
               const std::uint8_t** out, std::uint64_t* count_out,
               std::string* error) const {
    const json::Value* t = header_.find(name);
    if (!t) return Fail(error, "hy4 dfly: missing tensor " + name);
    if (t->member_str("dtype") != "BF16") return Fail(error, "hy4 dfly: " + name + " is not BF16");
    const json::Value* dims = t->find("shape");
    const json::Value* offsets = t->find("data_offsets");
    if (!dims || !offsets || offsets->items().size() != 2) return Fail(error, "hy4 dfly: bad entry " + name);
    std::vector<std::uint64_t> want(shape);
    if (dims->items().size() != want.size()) return Fail(error, "hy4 dfly: unexpected shape of " + name);
    std::uint64_t count = 1;
    for (std::size_t i = 0; i < want.size(); ++i) {
      if (dims->items()[i].as_size() != want[i]) return Fail(error, "hy4 dfly: unexpected shape of " + name);
      count *= want[i];
    }
    const std::uint64_t begin = offsets->items()[0].as_size();
    const std::uint64_t end = offsets->items()[1].as_size();
    if (end - begin != count * 2 || data_ + end > static_cast<const std::uint8_t*>(map_) + size_) {
      return Fail(error, "hy4 dfly: bad extent of " + name);
    }
    *out = data_ + begin;
    *count_out = count;
    return true;
  }
  // The BF16 tensor `name` as floats, checking its shape.
  bool Read(const std::string& name, std::initializer_list<std::uint64_t> shape,
            std::vector<float>* out, std::string* error) const {
    const std::uint8_t* src = nullptr;
    std::uint64_t count = 0;
    if (!RawBf16(name, shape, &src, &count, error))
      return false;
    out->resize(count);
    for (std::uint64_t i = 0; i < count; ++i) {
      std::uint16_t v;
      std::memcpy(&v, src + 2 * i, 2);
      (*out)[i] = Bf16(v);
    }
    return true;
  }

private:
  void* map_ = MAP_FAILED;
  std::size_t size_ = 0;
  const std::uint8_t* data_ = nullptr;
  json::Value header_;
};

}  // namespace

struct Drafter::LayerWeights {
  Tensor attn_norm, q_a, q_a_norm, q_b, kv_a, kv_a_norm, wk_b, wv_b, gate, o, post_proj_norm, sinks;
  Tensor ffn_norm, ffn_gate, ffn_up, ffn_down;
};

struct Drafter::Buffers {
  // R = kMaxSessions * kBlock rows (every session's block), S =
  // kMaxSessions (one slot per session).
  std::int32_t* tokens = nullptr;  // [R]
  std::int32_t* pos = nullptr;     // [R]
  std::int32_t* iota = nullptr;    // [max selection] 0, 1, ...
  std::int32_t* prev = nullptr;    // [S] the token before each slot
  std::int32_t* best = nullptr;    // [S] each slot's draft
  std::int32_t* table_new = nullptr;  // [S][table_capacity_] tokens to add
  float* best_work = nullptr;      // hy4_argmax_work_floats()
  float* conf = nullptr;           // [S] confidence logits
  float* x = nullptr;              // [R][hidden] residual
  float* xn = nullptr;             // [R][hidden]
  float* y = nullptr;              // [R][hidden]
  float* cat = nullptr;            // [S][2 hidden]
  float* f = nullptr;              // [S][hidden]
  float* qr = nullptr;             // [R][q_lora]
  float* q = nullptr;              // [R][heads][head_k]
  float* kva = nullptr;            // [R][latent]
  float* qfull = nullptr;          // [R][heads][latent]
  float* attn_work = nullptr;      // hy4_dfly_attention_work(kBlock, kHeads)
  std::uint64_t attn_work_floats = 0;
  float* att = nullptr;            // [R][heads][kv_lora]
  float* attv = nullptr;           // [R][heads][head_v]
  float* gate = nullptr;           // [R][heads][head_v]
  float* ffa = nullptr;            // [R][inter]
  float* ffb = nullptr;
  float* ffc = nullptr;
  float* hs = nullptr;      // [S][hidden] each slot's hidden
  float* hk = nullptr;      // [S][hidden] corrected slots
  float* prefix = nullptr;  // [S][head_rows_] head prefix logits
  float* logits = nullptr;  // [S][n_vocab]
  std::uint32_t max_sel = 0;
};

Drafter::Drafter(Model& target) : target_(target), buf_(std::make_unique<Buffers>()) {}

// Host side of each slot's copies: the token before it, the confidence
// logit and the draft.
struct Drafter::HostIo {
  std::int32_t prev[kMaxSessions];
  float conf[kMaxSessions];
  std::int32_t best[kMaxSessions];
  std::int32_t* table_new;  // [kMaxSessions * table_capacity_] pinned
};

Drafter::~Drafter() {
  for (void* p : allocations_) (void)hipFree(p);
  if (io_) {
    if (io_->table_new)
      (void)hipHostFree(io_->table_new);
    (void)hipHostFree(io_);
  }
}

std::unique_ptr<Drafter> Drafter::Load(const std::string& path, Model& target, std::string* error) {
  const Config& c = target.config();
  if (c.n_embd != kHidden || c.kv_lora != kKvLora || c.n_rot != kRope || c.head_k != kHeadK) {
    Fail(error, "hy4 dfly: the drafter does not match the target's dimensions");
    return nullptr;
  }
  if (target.weights().tok_embd.type != HY4_TYPE_Q8_0) {
    Fail(error, "hy4 dfly: needs a Q8_0 token embedding");
    return nullptr;
  }
  SafeTensors st;
  if (!st.Open(path, error)) return nullptr;
  std::unique_ptr<Drafter> d(new Drafter(target));
  d->mask_token_ = kMaskToken;
  d->target_layers_.assign(std::begin(kTargetLayers), std::end(kTargetLayers));
  for (const std::uint32_t l : d->target_layers_) {
    if (l >= c.n_layer || c.is_swa[l] || !c.has_indexer[l]) {
      Fail(error, "hy4 dfly: target layer " + std::to_string(l) + " is not a sparse indexer layer");
      return nullptr;
    }
  }

  // Temporary device staging buffer for load-time GPU quantization of BF16
  // safetensors rows (avoids CPU BF16->F32->Q8/Q4_K conversion loops).
  std::uint16_t* bf16_stage = nullptr;
  std::size_t bf16_stage_elems = 0;
  struct FreeStage {
    std::uint16_t*& p;
    ~FreeStage() {
      if (p)
        (void)hipFree(p);
    }
  } free_stage{bf16_stage};

  hipStream_t load_stream = target.stream();

  auto upload_bf16 = [&](const void* bf16_bytes, std::uint64_t m,
                         std::uint64_t k, int dst_type, Tensor* t) -> bool {
    const std::uint64_t count = m * k;
    if (count > bf16_stage_elems) {
      if (bf16_stage)
        (void)hipFree(bf16_stage);
      bf16_stage = nullptr;
      if (hipMalloc(&bf16_stage, count * sizeof(std::uint16_t)) != hipSuccess)
        return Fail(error, "hy4 dfly: staging allocation failed");
      bf16_stage_elems = count;
    }
    if (hipMemcpyAsync(bf16_stage, bf16_bytes, count * sizeof(std::uint16_t),
                       hipMemcpyHostToDevice, load_stream) != hipSuccess)
      return Fail(error, "hy4 dfly: weight upload failed");
    void* dev = nullptr;
    std::size_t bytes = 0;
    int rc = 1;
    if (dst_type == HY4_TYPE_Q4_K) {
      if (k % 256 != 0)
        return Fail(error, "hy4 dfly: Q4_K row not a multiple of 256");
      const std::uint64_t sbs = count / 256;
      bytes = sbs * kQ4KBlockBytes;
      if (!DeviceAlloc(reinterpret_cast<std::uint8_t**>(&dev), bytes,
                       &d->allocations_))
        return Fail(error, "hy4 dfly: weight allocation failed");
      rc = hy4_quantize_q4_k_bf16(bf16_stage, sbs, dev, load_stream);
    } else {
      if (k % 32 != 0)
        return Fail(error, "hy4 dfly: row not a multiple of 32");
      const std::uint64_t blocks = count / 32;
      bytes = blocks * kQ8BlockBytes;
      if (!DeviceAlloc(reinterpret_cast<std::uint8_t**>(&dev), bytes,
                       &d->allocations_))
        return Fail(error, "hy4 dfly: weight allocation failed");
      rc = hy4_quantize_q8_0_bf16(bf16_stage, blocks, dev, load_stream);
    }
    if (rc != 0 || hipStreamSynchronize(load_stream) != hipSuccess)
      return Fail(error, "hy4 dfly: weight quantization failed");
    d->resident_bytes_ += bytes;
    t->data = dev;
    t->type = dst_type;
    t->dims = {k, m};
    return true;
  };
  std::vector<float> host;
  auto upload_f32 = [&](std::uint64_t m, std::uint64_t k, Tensor* t) -> bool {
    void* dev = nullptr;
    const std::size_t bytes = host.size() * sizeof(float);
    if (!DeviceAlloc(reinterpret_cast<std::uint8_t**>(&dev), bytes,
                     &d->allocations_))
      return Fail(error, "hy4 dfly: weight allocation failed");
    if (hipMemcpy(dev, host.data(), bytes, hipMemcpyHostToDevice) != hipSuccess)
      return Fail(error, "hy4 dfly: weight upload failed");
    d->resident_bytes_ += bytes;
    t->data = dev;
    t->type = HY4_TYPE_F32;
    t->dims = {k, m};
    return true;
  };
  auto matrix = [&](const std::string& name, std::uint64_t m, std::uint64_t k,
                    Tensor* t, bool q4k = false) {
    const std::uint8_t* raw = nullptr;
    std::uint64_t count = 0;
    if (!st.RawBf16(name, {m, k}, &raw, &count, error))
      return false;
    const bool use_q4k = q4k && k % 256 == 0;
    return upload_bf16(raw, m, k, use_q4k ? HY4_TYPE_Q4_K : HY4_TYPE_Q8_0, t);
  };
  auto vector = [&](const std::string& name, std::uint64_t n, Tensor* t) {
    return st.Read(name, {n}, &host, error) && upload_f32(1, n, t);
  };

  const std::string top = "model.dfly_layers.";
  d->layers_.resize(kLayers);
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    LayerWeights& l = d->layers_[i];
    const std::string p = top + std::to_string(i) + ".";
    const std::string a = p + "self_attn.";
    bool ok =
        vector(p + "input_layernorm.weight", kHidden, &l.attn_norm) &&
        matrix(a + "q_a_proj.weight", kQLora, kHidden, &l.q_a) &&
        vector(a + "q_a_layernorm.weight", kQLora, &l.q_a_norm) &&
        matrix(a + "q_b_proj.weight", kHeads * kHeadK, kQLora, &l.q_b) &&
        matrix(a + "kv_a_proj_with_mqa.weight", kLatent, kHidden, &l.kv_a) &&
        vector(a + "kv_a_layernorm.weight", kKvLora, &l.kv_a_norm) &&
        matrix(a + "linear_gate.weight", kHeads * kHeadV, kHidden, &l.gate) &&
        matrix(a + "o_proj.weight", kHidden, kHeads * kHeadV, &l.o) &&
        vector(a + "post_proj_norm.weight", kHidden, &l.post_proj_norm) &&
        vector(a + "learnable_sink_param", kHeads, &l.sinks) &&
        vector(p + "post_attention_layernorm.weight", kHidden, &l.ffn_norm) &&
        matrix(p + "mlp.gate_proj.weight", kInter, kHidden, &l.ffn_gate,
               true) &&
        matrix(p + "mlp.up_proj.weight", kInter, kHidden, &l.ffn_up, true) &&
        matrix(p + "mlp.down_proj.weight", kHidden, kInter, &l.ffn_down, true);
    if (!ok) return nullptr;
    // kv_b [heads][nope + v][kv_lora]: the key half goes in transposed
    // ([heads][kv_lora][nope]) to absorb into the query, as the target's wk_b.
    const std::uint8_t* kv_b_raw = nullptr;
    std::uint64_t kv_b_count = 0;
    if (!st.RawBf16(a + "kv_b_proj.weight",
                    {kHeads * (kNope + kHeadV), kKvLora}, &kv_b_raw,
                    &kv_b_count, error))
      return nullptr;
    const auto* kv_b = reinterpret_cast<const std::uint16_t*>(kv_b_raw);
    std::vector<std::uint16_t> wk_bf16(std::size_t{kHeads} * kKvLora * kNope);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      for (std::uint32_t r = 0; r < kNope; ++r) {
        const std::uint16_t* row =
            kv_b + (std::size_t{h} * (kNope + kHeadV) + r) * kKvLora;
        for (std::uint32_t m = 0; m < kKvLora; ++m)
          wk_bf16[(std::size_t{h} * kKvLora + m) * kNope + r] = row[m];
      }
    }
    if (!upload_bf16(wk_bf16.data(), std::uint64_t{kHeads} * kKvLora, kNope,
                     HY4_TYPE_Q8_0, &l.wk_b))
      return nullptr;
    std::vector<std::uint16_t> wv_bf16(std::size_t{kHeads} * kHeadV * kKvLora);
    for (std::uint32_t h = 0; h < kHeads; ++h) {
      std::memcpy(wv_bf16.data() + std::size_t{h} * kHeadV * kKvLora,
                  kv_b + (std::size_t{h} * (kNope + kHeadV) + kNope) * kKvLora,
                  std::size_t{kHeadV} * kKvLora * sizeof(std::uint16_t));
    }
    if (!upload_bf16(wv_bf16.data(), std::uint64_t{kHeads} * kHeadV, kKvLora,
                     HY4_TYPE_Q8_0, &l.wv_b))
      return nullptr;
  }
  bool ok = vector(top + "mtp_fusion.enorm.weight", kHidden, &d->fuse_enorm_) &&
            vector(top + "mtp_fusion.hnorm.weight", kHidden, &d->fuse_hnorm_) &&
            matrix(top + "mtp_fusion.eh_proj.weight", kHidden, 2 * kHidden,
                   &d->fuse_eh_, true) &&
            vector(top + "mtp_fusion.gate", kHidden, &d->fuse_gate_) &&
            vector(top + "hidden_correction.hidden_norm.weight", kHidden,
                   &d->corr_hnorm_) &&
            vector(top + "hidden_correction.embed_norm.weight", kHidden,
                   &d->corr_enorm_) &&
            matrix(top + "hidden_correction.gate_proj.weight", kHidden,
                   2 * kHidden, &d->corr_gate_, true) &&
            matrix(top + "hidden_correction.up_proj.weight", kHidden,
                   2 * kHidden, &d->corr_up_, true) &&
            matrix(top + "hidden_correction.down_proj.weight", kHidden, kHidden,
                   &d->corr_down_, true) &&
            vector(top + "final_norm.weight", kHidden, &d->final_norm_) &&
            st.Read(top + "confidence_head.proj.weight", {1, kHidden}, &host,
                    error) &&
            upload_f32(1, kHidden, &d->confidence_w_) &&
            st.Read(top + "confidence_head.proj.bias", {1}, &host, error);
  if (!ok) return nullptr;
  d->confidence_b_ = host[0];

  Buffers& b = *d->buf_;
  b.max_sel = c.idx_top_k / c.kpool * c.kpool;
  b.attn_work_floats = hy4_dfly_attention_work(kBlock, kHeads);
  std::vector<void*>* own = &d->allocations_;
  constexpr std::size_t R = std::size_t{kMaxSessions} * kBlock;
  constexpr std::size_t S = kMaxSessions;
  ok = DeviceAlloc(&b.tokens, R, own) && DeviceAlloc(&b.pos, R, own) &&
       DeviceAlloc(&b.iota, b.max_sel, own) && DeviceAlloc(&b.prev, S, own) &&
       DeviceAlloc(&b.best, S, own) &&
       DeviceAlloc(&b.best_work, hy4_argmax_work_floats(), own) &&
       DeviceAlloc(&b.conf, S, own) && DeviceAlloc(&b.x, R * kHidden, own) &&
       DeviceAlloc(&b.xn, R * kHidden, own) &&
       DeviceAlloc(&b.y, R * kHidden, own) &&
       DeviceAlloc(&b.cat, S * 2 * kHidden, own) &&
       DeviceAlloc(&b.f, S * kHidden, own) &&
       DeviceAlloc(&b.qr, R * kQLora, own) &&
       DeviceAlloc(&b.q, R * kHeads * kHeadK, own) &&
       DeviceAlloc(&b.kva, R * kLatent, own) &&
       DeviceAlloc(&b.qfull, R * kHeads * kLatent, own) &&
       DeviceAlloc(&b.attn_work, b.attn_work_floats, own) &&
       DeviceAlloc(&b.att, R * kHeads * kKvLora, own) &&
       DeviceAlloc(&b.attv, R * kHeads * kHeadV, own) &&
       DeviceAlloc(&b.gate, R * kHeads * kHeadV, own) &&
       DeviceAlloc(&b.ffa, R * kInter, own) &&
       DeviceAlloc(&b.ffb, R * kInter, own) &&
       DeviceAlloc(&b.ffc, R * kInter, own) &&
       DeviceAlloc(&b.hs, S * kHidden, own) &&
       DeviceAlloc(&b.hk, S * kHidden, own) &&
       DeviceAlloc(&b.logits, S * c.n_vocab, own);
  if (!ok) {
    Fail(error, "hy4 dfly: scratch allocation failed");
    return nullptr;
  }
  if (hipHostMalloc(reinterpret_cast<void**>(&d->io_), sizeof(HostIo)) !=
      hipSuccess) {
    d->io_ = nullptr;
    Fail(error, "hy4 dfly: pinned allocation failed");
    return nullptr;
  }
  d->io_->table_new = nullptr;
  const Tensor& out = target.weights().output;
  d->head_ = out;
  // Drafts read a Q4_K copy of the trunk's LM head (nearly half the bytes of
  // Q8_0 and a quarter of binary16/BF16); proposals are verified by the exact
  // trunk head.
  if ((out.type == HY4_TYPE_F16 || out.type == HY4_TYPE_BF16 ||
       out.type == HY4_TYPE_Q8_0) &&
      out.dims.size() >= 2 && out.dims[0] % 256 == 0) {
    const std::uint64_t superblocks = out.dims[0] * out.dims[1] / 256;
    void* q4k = nullptr;
    if (!DeviceAlloc(reinterpret_cast<std::uint8_t**>(&q4k),
                     superblocks * kQ4KBlockBytes, own)) {
      Fail(error, "hy4 dfly: head allocation failed");
      return nullptr;
    }
    const int quantized =
        out.type == HY4_TYPE_F16
            ? hy4_quantize_q4_k_half(
                  static_cast<const std::uint16_t*>(out.data), superblocks, q4k,
                  load_stream)
        : out.type == HY4_TYPE_BF16
            ? hy4_quantize_q4_k_bf16(
                  static_cast<const std::uint16_t*>(out.data), superblocks, q4k,
                  load_stream)
            : hy4_quantize_q4_k_q8_0(out.data, superblocks, q4k, load_stream);
    if (quantized != 0 || hipStreamSynchronize(load_stream) != hipSuccess) {
      Fail(error, "hy4 dfly: head quantization failed");
      return nullptr;
    }
    d->head_.data = q4k;
    d->head_.type = HY4_TYPE_Q4_K;
    d->resident_bytes_ += superblocks * kQ4KBlockBytes;
  }
  d->head_rows_ = c.n_vocab;
  if ((d->head_.type == HY4_TYPE_Q4_K || d->head_.type == HY4_TYPE_Q8_0) &&
      kHeadPrefix < c.n_vocab) {
    d->head_rows_ = kHeadPrefix;
    d->row_bytes_ = d->head_.type == HY4_TYPE_Q4_K
                        ? std::uint64_t{kHidden} / 256 * kQ4KBlockBytes
                        : std::uint64_t{kHidden} / 32 * kQ8BlockBytes;
    d->table_capacity_ =
        std::min<std::uint32_t>(kTableRows, c.n_vocab - kHeadPrefix);
    d->tables_.reserve(kMaxSessions);
    if (!d->AddTable() ||
        !DeviceAlloc(&b.table_new,
                     std::size_t{kMaxSessions} * d->table_capacity_, own) ||
        hipHostMalloc(reinterpret_cast<void**>(&d->io_->table_new),
                      std::size_t{kMaxSessions} * d->table_capacity_ *
                          sizeof(std::int32_t)) != hipSuccess) {
      Fail(error, "hy4 dfly: head table allocation failed");
      return nullptr;
    }
  }
  std::vector<std::int32_t> iota(b.max_sel);
  std::iota(iota.begin(), iota.end(), 0);
  if (hipMemcpy(b.iota, iota.data(), iota.size() * sizeof(std::int32_t), hipMemcpyHostToDevice) != hipSuccess) {
    Fail(error, "hy4 dfly: upload failed");
    return nullptr;
  }
  return d;
}

bool Drafter::AddTable() {
  Table table;
  if (!DeviceAlloc(reinterpret_cast<std::uint8_t**>(&table.rows),
                   table_capacity_ * row_bytes_, &allocations_))
    return false;
  table.in.assign(target_.config().n_vocab, 0);
  tables_.push_back(std::move(table));
  resident_bytes_ += table_capacity_ * row_bytes_;
  return true;
}

Drafter::Table* Drafter::TableFor(const Session& session) {
  for (Table& t : tables_) {
    if (t.session == &session)
      return &t;
  }
  // A session's first draft: a new table while there is room for one, else
  // the one used longest ago, never one this batch uses. Load reserves
  // kMaxSessions tables, so pointers into tables_ stay valid.
  if (tables_.size() < kMaxSessions && tables_.back().session != nullptr)
    (void)AddTable();
  Table* pick = nullptr;
  for (Table& t : tables_) {
    if (t.used == batches_ && t.session != nullptr)
      continue;
    if (t.session == nullptr) {
      pick = &t;
      break;
    }
    if (pick == nullptr || t.used < pick->used)
      pick = &t;
  }
  if (pick == nullptr)
    return nullptr;
  for (const std::int32_t t : pick->ids)
    pick->in[t] = 0;
  pick->ids.clear();
  pick->synced = 0;
  pick->session = &session;
  return pick;
}

bool Drafter::SyncTable(Table& table, const Session& session,
                        std::uint32_t slot, std::string* error) {
  const std::vector<std::int32_t>& history = session.History();
  if (history.size() < table.synced) {
    for (const std::int32_t t : table.ids)
      table.in[t] = 0;
    table.ids.clear();
    table.synced = 0;
  }
  std::int32_t* host_new = io_->table_new + std::size_t{slot} * table_capacity_;
  std::int32_t* dev_new = buf_->table_new + std::size_t{slot} * table_capacity_;
  std::uint32_t added = 0;
  const auto vocab = static_cast<std::int32_t>(table.in.size());
  for (std::size_t i = table.synced; i < history.size(); ++i) {
    const std::int32_t t = history[i];
    if (t < static_cast<std::int32_t>(head_rows_) || t >= vocab || table.in[t])
      continue;
    if (table.ids.size() + added == table_capacity_)
      break;
    table.in[t] = 1;
    host_new[added++] = t;
  }
  table.synced = history.size();
  if (added == 0)
    return true;
  hipStream_t st = target_.stream();
  HY4_CHECK(hipMemcpyAsync(dev_new, host_new, added * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, st));
  HY4_CHECK(hy4_gather_rows(
      head_.data, row_bytes_, dev_new, added,
      static_cast<std::uint8_t*>(table.rows) + table.ids.size() * row_bytes_,
      st));
  table.ids.insert(table.ids.end(), host_new, host_new + added);
  return true;
}

bool Drafter::Draft(const Session& session, std::int32_t anchor, std::uint32_t count, float min_accept,
                    std::vector<std::int32_t>* drafts, std::string* error) {
  const Session* sessions[] = {&session};
  std::vector<std::int32_t>* out[] = {drafts};
  return DraftBatch(sessions, std::span(&anchor, 1), std::span(&count, 1),
                    min_accept, out, error);
}

bool Drafter::DraftBatch(std::span<const Session* const> sessions,
                         std::span<const std::int32_t> anchors,
                         std::span<const std::uint32_t> counts,
                         float min_accept,
                         std::span<std::vector<std::int32_t>* const> drafts,
                         std::string* error) {
  const auto ns = static_cast<std::uint32_t>(sessions.size());
  if (ns == 0)
    return true;
  if (ns > kMaxSessions || anchors.size() != ns || counts.size() != ns ||
      drafts.size() != ns)
    return Fail(error, "hy4 dfly: bad draft batch");
  std::uint32_t max_count = 0;
  for (std::uint32_t s = 0; s < ns; ++s) {
    drafts[s]->clear();
    if (counts[s] >= kBlock)
      return Fail(error, "hy4 dfly: at most 7 drafts per block");
    if (!sessions[s]->TapsReady())
      return Fail(error, "hy4 dfly: the session has no taps at its frontier");
    if (sessions[s]->Position() == 0)
      return Fail(error, "hy4 dfly: no frontier to draft from");
    for (std::uint32_t r = 0; r < s; ++r)
      if (sessions[r] == sessions[s])
        return Fail(error, "hy4 dfly: a session twice in one draft batch");
    max_count = std::max(max_count, counts[s]);
  }
  if (max_count == 0)
    return true;
  const Config& c = target_.config();
  const Weights& tw = target_.weights();
  Buffers& b = *buf_;
  hipStream_t st = target_.stream();
  const Exec ex{st, target_.blas_, false, target_.gemv_scratch_};
  // Session s's block is rows s * kBlock .. s * kBlock + kBlock - 1.
  const std::uint32_t n = ns * kBlock;
  ++batches_;
  Table* tables[kMaxSessions] = {};
  if (table_capacity_ != 0) {
    for (std::uint32_t s = 0; s < ns; ++s) {
      // Without a table to spare, the session scores the head's prefix only.
      tables[s] = TableFor(*sessions[s]);
      if (tables[s] == nullptr)
        continue;
      if (!SyncTable(*tables[s], *sessions[s], s, error))
        return false;
      tables[s]->used = batches_;
    }
  }

  std::int32_t tokens[kMaxSessions * kBlock];
  std::int32_t pos[kMaxSessions * kBlock];
  for (std::uint32_t s = 0; s < ns; ++s) {
    const std::uint32_t a = sessions[s]->Position();  // the anchor's position
    for (std::uint32_t i = 0; i < kBlock; ++i) {
      tokens[s * kBlock + i] = i == 0 ? anchors[s] : mask_token_;
      pos[s * kBlock + i] = static_cast<std::int32_t>(a + i) +
                            sessions[s]->RopeDelta();  // text after any image
    }
  }
  HY4_CHECK(hipMemcpyAsync(b.tokens, tokens, n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, st));
  HY4_CHECK(hipMemcpyAsync(b.pos, pos, n * sizeof(std::int32_t),
                           hipMemcpyHostToDevice, st));
  HY4_CHECK(hy4_embed_q8_0(tw.tok_embd.data, b.tokens, n, kHidden, b.x, st));

  // MTP fusion into each anchor slot: x[0] += gate * eh(enorm(E(anchor)) ::
  // hnorm(h)).
  for (std::uint32_t s = 0; s < ns; ++s) {
    float* cat = b.cat + std::size_t{s} * 2 * kHidden;
    HY4_CHECK(hy4_rms_norm(b.x + std::size_t{s} * kBlock * kHidden,
                           fuse_enorm_.f32(), cat, 1, kHidden, kHidden, kEps,
                           st));
    HY4_CHECK(hy4_rms_norm(sessions[s]->TapHidden(), fuse_hnorm_.f32(),
                           cat + kHidden, 1, kHidden, kHidden, kEps, st));
  }
  if (!MatVec(fuse_eh_, b.cat, b.f, kHidden, 2 * kHidden, ns, ex, error))
    return false;
  for (std::uint32_t s = 0; s < ns; ++s)
    HY4_CHECK(hy4_mul_add(b.x + std::size_t{s} * kBlock * kHidden,
                          b.f + std::size_t{s} * kHidden, fuse_gate_.f32(),
                          kHidden, kHidden, st));

  const float base = c.rope_base;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadK));
  float* x_buf = b.x;
  float* y_buf = b.y;
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const LayerWeights& l = layers_[i];
    const std::uint32_t tl = target_layers_[i];
    HY4_CHECK(hy4_rms_norm(x_buf, l.attn_norm.f32(), b.xn, n, kHidden, kHidden,
                           kEps, st));
    if (!MatVecPair(l.q_a, b.qr, kQLora, l.kv_a, b.kva, kLatent, b.xn, kHidden,
                    n, ex, error))
      return false;
    HY4_CHECK(hy4_mla_prep(b.qr, l.q_a_norm.f32(), kQLora, b.kva,
                           l.kv_a_norm.f32(), kKvLora, kLatent, b.pos, nullptr,
                           kRope, base, kEps, n, st));
    if (!MatVec(l.q_b, b.qr, b.q, kHeads * kHeadK, kQLora, n, ex, error))
      return false;
    HY4_CHECK(hy4_rope_3d_copy(b.q, b.pos, nullptr, n, kHeads,
                               std::uint64_t{kHeads} * kHeadK, kHeadK, kNope,
                               kRope, base, b.qfull + kKvLora,
                               std::uint64_t{kHeads} * kLatent, kLatent, st));
    HY4_CHECK(hy4_gemv_q8_0_grouped(l.wk_b.data, b.q, b.qfull, kKvLora, kNope,
                                    kHeads, n, std::uint64_t{kHeads} * kHeadK,
                                    kHeadK, std::uint64_t{kHeads} * kLatent,
                                    kLatent, st));

    for (std::uint32_t s = 0; s < ns; ++s) {
      const Session& session = *sessions[s];
      const std::uint32_t a = session.Position();
      const std::int32_t frontier = static_cast<std::int32_t>(a) - 1;
      // Keys: the target's latent rows at its selection for the frontier
      // (or every earlier position), then the block's own rows, all visible.
      std::uint32_t sel_count = 0;
      const std::int32_t* sel = session.TapSelection(i, &sel_count);
      if (!sel) {
        sel = b.iota;
        sel_count = std::min(a, b.max_sel);
      }
      const float* qfull = b.qfull + std::size_t{s} * kBlock * kHeads * kLatent;
      const float* kva = b.kva + std::size_t{s} * kBlock * kLatent;
      float* att = b.att + std::size_t{s} * kBlock * kHeads * kKvLora;
      HY4_CHECK(hy4_dfly_attention(
          qfull, session.KvRows(tl), session.KvRing(tl), sel, sel_count,
          frontier, kva, l.sinks.f32(), scale, att, kBlock, kHeads, kLatent,
          kKvLora, b.attn_work, b.attn_work_floats, st));
    }
    HY4_CHECK(hy4_gemv_q8_0_grouped(l.wv_b.data, b.att, b.attv, kHeadV, kKvLora, kHeads, n,
                                    std::uint64_t{kHeads} * kKvLora, kKvLora, std::uint64_t{kHeads} * kHeadV, kHeadV,
                                    st));
    if (!MatVecGate(l.gate, b.xn, b.gate, b.attv, kHeads * kHeadV, kHidden, n,
                    ex, error))
      return false;
    if (!MatVec(l.o, b.attv, y_buf, kHidden, kHeads * kHeadV, n, ex, error))
      return false;
    HY4_CHECK(hy4_rms_norm(y_buf, l.post_proj_norm.f32(), y_buf, n, kHidden,
                           kHidden, kEps, st));
    HY4_CHECK(hy4_add(x_buf, y_buf, x_buf, std::uint64_t{n} * kHidden, st));

    HY4_CHECK(hy4_rms_norm(x_buf, l.ffn_norm.f32(), b.xn, n, kHidden, kHidden,
                           kEps, st));
    if (!SwigluVec(l.ffn_gate, l.ffn_up, b.ffa, b.ffb, b.ffc, kInter, b.xn,
                   kHidden, n, ex, error))
      return false;
    if (!MatVecAdd(l.ffn_down, b.ffc, x_buf, y_buf, kHidden, kInter, n, ex,
                   error))
      return false;
    std::swap(x_buf, y_buf);
  }
  HY4_CHECK(hy4_rms_norm(x_buf, final_norm_.f32(), b.xn, n, kHidden, kHidden,
                         kEps, st));

  // Slot k predicts the token at a + k from its hidden, corrected by the
  // token before it, through the target's LM head. The confidence head reads
  // the corrected hidden, so a slot it stops at skips the LM head. The
  // sessions still drafting share each slot's correction and head.
  HostIo& io = *io_;
  std::int32_t prev[kMaxSessions];
  float accept[kMaxSessions];
  bool going[kMaxSessions];
  for (std::uint32_t s = 0; s < ns; ++s) {
    prev[s] = anchors[s];
    accept[s] = 1.0f;
    going[s] = counts[s] > 0;
  }
  const std::size_t vstride = c.n_vocab;
  for (std::uint32_t k = 1; k <= max_count; ++k) {
    std::uint32_t active[kMaxSessions];
    std::uint32_t na = 0;
    for (std::uint32_t s = 0; s < ns; ++s)
      if (going[s] && k <= counts[s])
        active[na++] = s;
    if (na == 0)
      break;
    for (std::uint32_t j = 0; j < na; ++j) {
      io.prev[j] =
          prev[active[j]];  // read by the copy before the synchronization below
      HY4_CHECK(
          hipMemcpyAsync(b.hs + std::size_t{j} * kHidden,
                         b.xn + (std::size_t{active[j]} * kBlock + k) * kHidden,
                         kHidden * sizeof(float), hipMemcpyDeviceToDevice, st));
    }
    HY4_CHECK(hipMemcpyAsync(b.prev, io.prev, na * sizeof(std::int32_t),
                             hipMemcpyHostToDevice, st));
    HY4_CHECK(hy4_embed_q8_0(tw.tok_embd.data, b.prev, na, kHidden, b.y, st));
    HY4_CHECK(hipMemcpy2DAsync(b.cat, 2 * kHidden * sizeof(float), b.hs,
                               kHidden * sizeof(float), kHidden * sizeof(float),
                               na, hipMemcpyDeviceToDevice, st));
    HY4_CHECK(hipMemcpy2DAsync(b.cat + kHidden, 2 * kHidden * sizeof(float),
                               b.y, kHidden * sizeof(float),
                               kHidden * sizeof(float), na,
                               hipMemcpyDeviceToDevice, st));
    HY4_CHECK(hy4_rms_norm(b.cat, corr_hnorm_.f32(), b.cat, na, kHidden,
                           2 * kHidden, kEps, st));
    HY4_CHECK(hy4_rms_norm(b.cat + kHidden, corr_enorm_.f32(), b.cat + kHidden,
                           na, kHidden, 2 * kHidden, kEps, st));
    if (!SwigluVec(corr_gate_, corr_up_, b.ffa, b.ffb, b.ffc, kHidden, b.cat,
                   2 * kHidden, na, ex, error))
      return false;
    if (!MatVecAdd(corr_down_, b.ffc, b.hs, b.hk, kHidden, kHidden, na, ex,
                   error))
      return false;
    // The confidence head: the chance the target accepts this slot's draft.
    if (!MatVec(confidence_w_, b.hk, b.conf, 1, kHidden, na, ex, error))
      return false;
    HY4_CHECK(hipMemcpyAsync(io.conf, b.conf, na * sizeof(float),
                             hipMemcpyDeviceToHost, st));
    HY4_CHECK(hipStreamSynchronize(st));
    std::uint32_t kept = 0;
    for (std::uint32_t j = 0; j < na; ++j) {
      const std::uint32_t s = active[j];
      accept[s] *= 1.0f / (1.0f + std::exp(-(io.conf[j] + confidence_b_)));
      if (accept[s] < min_accept)
        going[s] = false;
      else
        ++kept;
    }
    if (kept == 0)
      break;
    // The head's prefix for every row at once (a stopped session's row is
    // computed and ignored), then each session's own table and argmax.
    Tensor prefix = head_;
    prefix.dims = {kHidden, head_rows_};
    float* first = na == 1 ? b.logits : b.prefix;
    if (na > 1 && b.prefix == nullptr) {
      if (!DeviceAlloc(&b.prefix, std::size_t{kMaxSessions} * head_rows_,
                       &allocations_))
        return Fail(error, "hy4 dfly: head prefix allocation failed");
      first = b.prefix;
    }
    if (!MatVec(prefix, b.hk, first, head_rows_, kHidden, na, ex, error))
      return false;
    for (std::uint32_t j = 0; j < na; ++j) {
      const std::uint32_t s = active[j];
      if (!going[s])
        continue;
      float* logits = b.logits + j * vstride;
      if (na > 1)
        HY4_CHECK(hipMemcpyAsync(logits, b.prefix + std::size_t{j} * head_rows_,
                                 head_rows_ * sizeof(float),
                                 hipMemcpyDeviceToDevice, st));
      const auto table_rows =
          tables[s] ? static_cast<std::uint32_t>(tables[s]->ids.size()) : 0u;
      if (table_rows) {
        const Tensor table{tables[s]->rows, head_.type, {kHidden, table_rows}};
        if (!MatVec(table, b.hk + std::size_t{j} * kHidden, logits + head_rows_,
                    table_rows, kHidden, 1, ex, error))
          return false;
      }
      HY4_CHECK(hy4_argmax(logits, head_rows_ + table_rows, b.best_work,
                           b.best + j, st));
    }
    HY4_CHECK(hipMemcpyAsync(io.best, b.best, na * sizeof(std::int32_t),
                             hipMemcpyDeviceToHost, st));
    HY4_CHECK(hipStreamSynchronize(st));
    for (std::uint32_t j = 0; j < na; ++j) {
      const std::uint32_t s = active[j];
      if (!going[s])
        continue;
      prev[s] = static_cast<std::uint32_t>(io.best[j]) < head_rows_
                    ? io.best[j]
                    : tables[s]->ids[io.best[j] - head_rows_];
      drafts[s]->push_back(prev[s]);
    }
  }
  return true;
}

}  // namespace gufo::models::hy4_flash
