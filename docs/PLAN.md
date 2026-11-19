# Hy4 Flash as a Gufo extension: plan

Status: draft, 2026-11-19. Nothing is implemented yet.

## Goal

An experimental Gufo source extension that serves Tencent's Hy4 Flash on Strix
Halo (`gfx1151`). The Q4_K_M trunk stays resident, the Engram tables are read
from disk, and DFly2 drafts tokens for speculative decoding. Vision is in scope
but comes after text. The extension borrows DS4.1 code where Hy4 shares its
design.

## Model files

In `~/models/hy4-flash/`:

| File | Size | Use |
| --- | ---: | --- |
| `Hy4-flash-Q4_K_M-00001-of-00002.gguf` (Xylolabs) | 97.8 GB | Trunk, resident |
| `Hy4-flash-Q4_K_M-00002-of-00002.gguf` | 63.4 GB | Engram tables, FP8 E4M3, read from disk |
| `mmproj-Hy4-flash-BF16.gguf` | 1.0 GB | Vision tower |
| `tencent-dfly2/model-00148-of-00148.safetensors` | 1.7 GB | DFly2 drafter, BF16 (87 tensors, nothing else) |
| `tencent-dfly2/{config,tokenizer,chat_template,...}` | small | Config, tokenizer, chat template |

Why Xylolabs over AngelSlim: KL 0.244 vs 0.281 against BF16, and 7 GB
smaller. Both GGUFs follow the tensor naming of llama.cpp PR #32517. Neither
ships DFly2: AngelSlim has one `nextn` layer, Xylolabs has none.

## Architecture, from `config.json` and the llama.cpp PR

- **Backbone:** 42 layers, hidden size 3072, vocabulary 120,832. Layer 0 is
  dense (FFN 13,824); layers 1-41 are MoE.
- **Hyper-connections:** iHC, 4 residual streams. Unlike DS4.1's mHC, the comb
  is the identity: each stream keeps its own residual and there is no Sinkhorn
  mix. Pre/post weights (`hc_*_fn/base/scale`) work as in DS4.1, with
  `hc_eps` 1e-6 and `hc_magnitude` 2.0.
- **Attention (gated MLA):**
  - Every layer: q_lora 1024, kv_lora 512, qk 192 nope + 64 rope, v 256,
    learnable sinks, element-wise output gate (`attn_gate`), qk norm, sandwich
    norms (`post_attention_norm`, `post_ffw_norm`). MLA runs as MQA, so each
    layer caches one 576-wide latent row per token.
  - **Sparse layers (12)** at 0, 1, 5, 9, ... 41: 32 heads, rope θ 1e7. A
    compressed "kpool" indexer picks which tokens to attend. It works on groups
    of 4 tokens, pooled by a gated projection of the normed key. It scores 512
    groups and selects up to 2048 positions. The indexer has 16 heads of
    dimension 128, rotated only on its trailing rope part.
  - **SWA layers (30):** 64 heads, a 512-token window (511 previous plus the
    current one), rope θ 1e4.
  - **Positions:** interleaved MRoPE, sections [12, 10, 10].
- **LatentMoE:** the router scores the full-width input: a sigmoid with an
  expert bias, top 16 of 512 experts, normalized, then scaled by 3.998. The
  routed experts run in a 1536-wide latent space (`ffn_routed_up` /
  `ffn_routed_down` project in and out), with 1536 intermediate width. There
  is one shared expert, and the SwiGLU limit is 10 on routed experts only.
- **Engram:** bigram and trigram hashes, 8 heads per order, at
  `engram_layer_ids` [2, 12, 22, 32]. The model card says 1, 11, 21, 31, so
  check against the PR. Unlike DS4.1, each looked-up row is RMS-normed
  (eps = FP32 machine epsilon) and scaled by its head weight. The key/value
  projection has a bias, the gate is DS4.1's, and a token with no valid n-gram
  (or an image position) gets no Engram delta.
- **LM head:** F16 in this GGUF; the reference computes it in FP32.
- **Vision:** SigLIP2-So400m adaptation, 27 layers, hidden size 1152, patch
  16, any resolution up to 2048, 2×2 merge, MLP projector. The llama.cpp PR
  does not touch `mtmd`, so its vision path is Tencent's own build or
  transformers 5.23.
- **DFly2:** 5 draft layers, each paired with a sparse target layer (25, 29,
  33, 37, 41) whose KV cache and indexer selection it reuses. It drafts a
  5-token masked block in one pass (`dfly_mask_token_id` 120009) and samples
  the draft probabilistically. Its weights are an MTP fusion (`eh_proj`, norms,
  `gate`, slot 0) for the anchor, a hidden-correction MLP, dense SwiGLU MLPs
  (no MoE), gated MLA attention with sinks, and a confidence head, as DSpark
  has.

## Reuse from DS4.1

| Hy4 part | DS4.1 source | Work |
| --- | --- | --- |
| GGUF loader, tokenizer | `runtime/model_data.cpp`, `tokenizer.cpp` | New tensor table; check the tokenizer type |
| HC pre/collapse/post | `ds41_v41_hc_*` | Drop Sinkhorn and comb (identity); smaller kernels |
| Engram hash, async disk reads, gate | `v41_decode.cpp` Engram path | New hash constants, row norm and head weight, bias, dead-token mask |
| Gated MLA with sinks, SWA ring | DS4.1 attention, `window` ring | 64-head SWA layers; a 512-row window instead of DS4.1's |
| Sparse selection | DS4.1 CSA2 indexer | **New:** kpool group pooling plus group top-k is a different algorithm |
| Q8_0/Q4_K/Q6_K GEMVs, expert GEMV, mmq | `kernels/rocm/` | Reuse; the latent projections are plain dense GEMVs |
| Router | sqrtsoftplus router | Sigmoid plus bias; new top-16-of-512 |
| Sessions, prompt cache, continuation | `v41_session.cpp`, `continuation_adapter.cpp` | Adapt to the new state: SWA rings, kpool cache, latent rows |
| Speculative cycle, width policy, device sampler | DSpark cycle (E87, E99), `dspark_sampler.hpp` | Reuse the cycle and policy; a new drafter |
| Vision | `vision/encoder.hip`, `lt_gemm` | Different ViT; reuse the GEMM wrapper and prompt plumbing |

Not needed: expert streaming. The trunk is resident.

## Memory budget (124 GiB host)

- Trunk 91.1 GiB (97.8 GB), vision 1 GB, DFly2 about 1 GB at Q8_0.
- KV: the 12 sparse layers × 576 × 2 bytes come to 13.8 KB per token, plus the
  kpool cache. That's about 1.8 GB at 128K and 14 GB at 1M. The 30 SWA layers
  hold 512 rows each.
- What's left goes to the page cache for the 63.4 GB Engram file (its hot rows),
  the prompt cache and the runtime. Decode reads 64 rows per token
  (4 layers × 2 orders × 8 heads), 96 bytes each.

## Expected decode speed (estimate, to check)

Per token, about 7.7 GB of weights are read: Q8_0 attention and shared experts
(~105 MB per layer), 16 Q4_K latent experts (~64 MB per layer), and the 0.74 GB
F16 head. At ~215 GB/s achieved, that caps AR decode near 28 tok/s. The Q8_0
attention outweighs the routed experts, so requantizing attention is a later
lever. For scale, an M5 Max reaches 48 tok/s on the same quant with llama.cpp.

## Correctness oracle

llama.cpp PR #32517 (`cf70454a6`, fetched as branch `hy4-flash-pr` in
`~/projects/llama.cpp`) runs this exact GGUF. Build it for HIP/CPU and dump
per-layer hidden states and logits for fixed prompts with `llama-eval-callback`.
Every phase compares against those dumps. Transformers BF16 is out of reach in
memory (440 GB) except for single-layer checks. DFly2 has no llama.cpp
implementation: its oracle is a BF16 PyTorch run of the five draft layers over
target states dumped from our engine.

## Phases

0. **Repository.** Skeleton after `gemma4-gufo-extension`/`ds41-gufo-extension`:
   `extension.json`, flake pinned to Gufo `main` `0a473ebf` (API 0.3.29),
   `AGENTS.md`, `.clang-format`, licenses. The provider takes key
   `gguf:hy_v4_flash`.
1. **Oracle.** Build the llama.cpp PR, check that it loads the download, and
   record golden dumps: 3 text prompts, one with more than 512 tokens so the
   window wraps and kpool selects. Record llama.cpp's own speed as a baseline.
2. **Loader and tokenizer.** Tensor inventory, resident upload, chat template
   (`chat_template.jinja`, reasoning and tool syntax), tokenizer contract tests.
3. **Decoder, AR, one session.** Layer by layer against the dumps: embeddings
   and HC → dense layer 0 → SWA layer → LatentMoE → Engram → sparse layer with
   kpool → head. A `hip` test for each kernel.
4. **Serving.** Sessions, `gufo serve`, prompt cache and continuation. Then the
   `conversation`, `tools`, `long-context` and `cache` suites.
5. **Prefill.** Batched prefill: SWA, kpool pooling over groups, Engram batch
   reads.
6. **DFly2.** Convert the safetensors shard to a GGUF drafter with a tool in
   `tools/`. Build the PyTorch reference, then the drafter path, reusing the
   DS4.1 speculative cycle and width policy. Gate: identical greedy replies
   with and without the drafter.
7. **Vision.** Encoder, projector, MRoPE positions. `image-inputs` and
   `tool-images` suites.
8. **Measurement and documentation.** Benchmark card, `EXPERIMENTS.md`,
   `QUALITY.md` (KL against llama.cpp on the same GGUF).

## Read from the Tencent files

- **Tokenizer:** BPE with 832 added tokens and its own pretokenizer sequence.
  The splits are digit runs of 1-3, isolated CJK/kana runs, then a
  punctuation-letter regex. It needs a new pretokenizer; DS4.1's will not do.
- **Chat template** (`chat_template.jinja`, 342 lines):
  - Turn markers are `<｜hy_start｜>`, `<｜hy_middle｜>` and `<｜hy_end｜>`.
  - Reasoning uses `<think>…</think>`, and `<｜reasoning_mode｜>` takes the
    `reasoning_effort` values `no_think` and `high`.
  - Tool calls are `<tool_calls><tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call></tool_calls>`.
    Tool results go in `<tool_response>`, and the tool list in `<tools>`.
  - The template also has image and video pad markers.
- **DFly2 shapes** (BF16, 1.69 GB):
  - Each draft layer has gated MLA in the sparse-layer shape: 32 heads, q_b
    8192, kv_b 14336, sinks [32], `post_proj_norm`. It is followed by a dense
    SwiGLU MLP of width 9216.
  - `mtp_fusion.eh_proj` maps 6144 → 3072, with a `gate` vector [3072].
  - `hidden_correction` is a SwiGLU from 6144 to 3072, with intermediate width
    3072.
  - `confidence_head` is 3072 → 1.
  - At Q8_0 it comes to about 0.9 GB.

## Open questions

- Names: repository `hy4-gufo-extension`, extension `hy4flash`, provider
  `hy4flash`?
- Engram layer indices: config [2, 12, 22, 32] or card [1, 11, 21, 31].
- Is the Q6_K expert `down` present in Xylolabs Q4_K_M (llama.cpp's mixed
  recipe), or is everything Q4_K? Read the GGUF tensor types.
- Does the DFly2 draft need the target's final hidden state at all 5 positions,
  or only the anchor's (`dfly_mtp_fusion` "slot0")? Transformers 5.23's
  `hy_v4_flash` code should say; the Tencent repo ships no modeling code.
