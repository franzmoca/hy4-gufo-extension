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
- **Engram:** bigram and trigram hashes, 8 heads per order, at layers
  1, 11, 21 and 31 (0-based). Unlike DS4.1, each looked-up row is RMS-normed
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

## Expected decode speed (from the GGUF tensor table)

Weights read per decoded token, from the trunk's real tensor types (all
non-expert tensors, 16 of 512 experts, one embedding row):

| Part | GB/token |
| --- | ---: |
| Attention and indexer (Q8_0) | 5.31 |
| Routed experts (16 × Q4_K gate/up, Q4_K or Q6_K down) | 2.81 |
| Dense layer 0, shared experts, latent projections (Q8_0) | 1.42 |
| LM head (F16) | 0.74 |
| Engram projections (Q8_0) | 0.10 |
| **Total** | **10.4** |

At ~215 GB/s achieved, that caps AR decode near 20 tok/s. Attention is half
the traffic: the 30 SWA layers carry 64 heads, so their `attn_gate`
(3072 × 16384), `attn_output` and `attn_q_b` are twice the sparse layers'. The
first lever after correctness is therefore a lower-bit attention quant, not
expert work. For scale, an M5 Max reaches 48 tok/s on the same quant with
llama.cpp.

## Correctness oracle

llama.cpp PR #32517 (`cf70454a6`, fetched as branch `hy4-flash-pr` in
`~/projects/llama.cpp`) runs this exact GGUF. Build it for HIP/CPU and dump
per-layer hidden states and logits for fixed prompts with `llama-eval-callback`.
Every phase compares against those dumps. Transformers BF16 is out of reach in
memory (440 GB) except for single-layer checks. DFly2 has no llama.cpp
implementation: its oracle is a BF16 PyTorch run of the five draft layers over
target states dumped from our engine.

Status (2026-11-19): `tools/oracle/llama-ref.nix` builds the PR. It must load
with `--load-mode none --override-tensor '^per_layer_token_embd\.weight$=CPU'
--lazy-mode on`; the default mmap load thrashes, since the 98 GB trunk on the
GPU leaves too little page cache for its own mapping, and stalls for over 15
minutes. With buffered loading it loads in 25 s and decodes greedy at
**14.4 tok/s** (33 ms/token prompt, 5 tokens), the baseline to beat.
`"The capital of France is"` tokenizes to `802 8778 299 12749 341` (no BOS) and
continues ` ___. Paris is also known as the "City of Light". ...`.

Compare by token ids through `llama-server` (`/completion` with an id list,
`n_probs`, `post_sampling_probs: false`): `llama-completion` tokenizes a text
prompt its own way and is not a like-for-like reference.

Long-context comparisons must aggregate. Next-token distributions past
2,048 tokens depend on which pools the indexer keeps, and those flip with
rounding: on one 3,712-token prompt, llama.cpp itself gives the top token
p 0.68 with its default F16 cache and p 0.20 with an F32 cache. Single
positions there are inconclusive. The gate is perplexity in the kpool regime:
`llama-perplexity -c 4096` (second halves, positions 2,048-4,095) against
`hy4_eval_probe --ppl 4096` on the same text. Reference, two chunks of Gufo
docs: PPL 14.927 ± 0.964 (F16 cache) and 14.930 (F32 cache). This engine:
**14.762**, within the reference's error. Short prompts match greedy token
for token.

Perplexity is still too coarse a gate: this model is chaotic under rounding
(a 1e-6 relative change in one attention output moves a next-token
probability by 0.05, through near-tied expert choices), so numerically
equivalent kernels give chunk perplexities 5% apart. The gate is the KL
divergence from llama.cpp's own logits on the same tokens:
`llama-perplexity -c 4096 --chunks 2 --kl-divergence-base base.kld` once,
then `hy4_eval_probe MODEL - --kl base.kld`. Reference noise floor (its F32
against its F16 cache): mean KL 0.070, same top token 92.0%. This engine,
batched path: 0.096 ± 0.010 and 91.4%; with the routed experts on the matvec
path instead: 0.087 and 91.8% (indistinguishable). Greedy-token identity is
not a gate for kernel changes here.

## Engine design (decided 2026-11-19, while working unattended)

These are judgment calls made without the maintainer; revisit any of them.

1. **A new engine, not a fork of DS4.1's.** DS4.1's 26K lines carry expert
   streaming, DSpark, adapters, steering and the bounded prefill replay, none
   of which Hy4 needs first. The Hy4 engine starts small and copies specific
   DS4.1 pieces in when it needs them, recording each in `UPSTREAM.md`.
2. **Correctness first, on llama.cpp's own kernels.** Quantized projections go
   through the llama.cpp-derived `mmvq`/`mmq` kernels that DS4.1 vendors
   (`kernels/rocm/mmq/`), copied here with the `ds41_` prefix renamed to
   `hy4_`. Two extensions in one Gufo binary must not share symbols. The vector
   path already takes Q8_0, Q4_K and Q6_K at run time. The arithmetic then
   matches the oracle's closely, so per-layer comparisons stay tight. Custom
   GEMVs are an optimization after correctness.
3. **Small custom kernels** cover what the vendored path cannot: `attn_k_b`
   (K = 192, not a multiple of 256), the F16 head, Q8_0 embedding rows, norms,
   iHC pre/post, the router, MLA attention with sinks, the kpool indexer and
   Engram.
4. **Evaluation in chunks of up to 8 tokens.** `mmvq` serves N ≤ 8, so one
   evaluation takes 1-8 tokens: decode is N = 1, and prefill loops over chunks
   of 8. Large-batch MMQ prefill is phase 5.
5. **KV in FP32 at first.** Each layer stores one 576-wide latent row per token
   (512 normed `kv_cmpr` + 64 roped `k_pe`). Sparse layers keep every row;
   SWA layers keep a 512-row ring. FP32 costs 27.6 KB per token for the 12
   sparse layers (1.8 GB at 64K). F16 like llama.cpp's default cache is a later
   memory/speed change.
6. **Full attention below 2,048 tokens.** A sparse layer selects at most 511
   pooled 4-token groups plus its own group, so at up to 2,048 tokens it
   attends to all of them. The indexer's pooled keys are still written from
   the first token, and selection is enabled and tested past 2,048.
7. **Engram rows are read on the host** from the memory-mapped FP8 shard
   (page cache), 64 rows of 96 bytes per token, then uploaded. There's no
   async read pipeline until profiles show it matters.
8. **Development builds** run in a Gufo worktree at the flake pin
   (`~/projects/gufo-hy4`, `0a473ebf`) with this repository as the extension
   directory. The llama.cpp oracle is built by
   `tools/oracle/llama-ref.nix`: the PR on Gufo's `gfx1151` ROCm 10 set, without
   the nixpkgs web UI step and backport.
9. **Nothing leaves the machine** while the maintainer is away: commits stay
   local, with no GitHub repository or push.
10. **Prefill in 256-token passes** (`Model::kMaxChunk`): dense and routed
    projections through the vendored MMQ (the routed grids bounded by the real
    expert buckets), F32 projections through hipBLAS SGEMM, `wk_b` (K = 192)
    dequantized to F32 for a strided-batched SGEMM, `wv_b` through the grouped
    MMQ, and attention as two SGEMMs around a softmax kernel with sinks, the
    SWA window and the indexer's selection mask. Passes of up to 8 tokens keep
    the matvecs.
11. **The gate for kernel changes is the KL divergence** from llama.cpp's
    logits over 4,094 positions (see Correctness oracle), not greedy identity
    or perplexity, both of which rounding moves on this model.
12. **Serving without DSpark-era machinery.** The runner is serial (one
    session per step, no batched decode), prefill is exact in 256-token
    passes, and the prompt cache stores tokens, the full-attention layers'
    latent rows and the indexers' pooled keys as append rows; the SWA and
    indexer rings (55 MB per checkpoint), logits and position are private
    state. Window-row components for the SWA rings would make checkpoints
    smaller and are a later change.
13. **Default serving context 32,768** when `--context` is absent: the native
    context would take 7 GiB of latent rows per session.
14. **Tool arguments are typed by the scanner.** Hy4's `<arg_value>` carries no
    type and a registered dialect's scanner gets no schema, so a value that
    parses as non-string JSON is typed and anything else is a string. A
    string parameter whose value looks like a number is then rejected by the
    schema check. Fixing this needs a Gufo API change (schema-aware
    interpretation for registered dialects, as the host does for Qwen).

15. **DFly2 runs from the safetensors file, quantized at load.** No GGUF
    converter: the loader reads the BF16 `model.dfly_layers.*` tensors of the
    Tencent shard (a file or the checkpoint directory, `--dfly-model`) and
    uploads Q8_0 matrices and F32 norms (0.9 GB). The draft only decides how
    many target tokens one pass checks, so its rounding costs acceptance,
    never output. The design, from AngelSpec's `dfly2` code (public, read
    only): a block of 8 slots at positions a .. a + 7, the anchor a (sampled
    from the target's frontier logits) then mask tokens (120009), through the
    target's embedding; the anchor slot adds `gate * eh_proj(enorm(E(anchor))
    :: hnorm(h))`, h the target's final hidden after the output norm at a - 1
    (only that one state: the open question below is settled). Five MLA layers
    with the sink, the sigmoid output gate and the post-projection norm; layer
    i attends to the block (bidirectional) and to the target's latent rows of
    sparse layer 25 + 4i at the positions that layer's indexer selected for
    a - 1 (every earlier position below 2,048), filtered to <= a - 1. RoPE in
    the target's pair layout at the sparse base (1e7). Slot k >= 1 predicts
    a + k: its final-normed hidden plus the hidden correction on the previous
    token's embedding (the anchor, then slot k - 1's draft), through the
    target's LM head. The confidence head is not used yet.
16. **Speculative steps inside the runner.** `DecodeStep` samples the anchor,
    drafts `--dfly-draft-tokens` (default 3) tokens, verifies anchor and drafts
    in one pass (`Session::Verify`), and walks the rows with the request's own
    sampler: a draw equal to the draft keeps it; the first that differs is
    deferred (`SamplerState::DeferSample`) to the next step, which samples it
    from the same logits. Sampling, penalties and grammars therefore stay
    exact, and every emitted token is in the KV (`Session::Keep` drops the
    rejected rows; rings and pools are rewritten by position, and their stale
    rows are never read). The session taps the frontier's hidden and the five
    layers' selections on every pass; a restored state has none, so its first
    step is a plain one. `--speculative` is not used: Gufo's serve knows no
    `dfly` backend, so the drafter is a provider option.
17. **The DFly2 gate is the verify pass's KL, not identical text.** A verify
    pass of n > 1 tokens runs the vendored matvecs with n columns, whose
    summation order differs from the 1-column decode. Against one-token steps
    over the same tokens it gives KL 0.0018 (0.0021 for a 16-token batched
    pass), 40 times under the llama.cpp noise floor; greedy replies then match
    until a near tie and may differ after it. Bit-identical replies would need
    column-count-independent matvecs, which is not worth their cost.

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
6. **DFly2.** Load the drafter from the safetensors shard (decision 15), run it
   in the runner's decode step (16). Gate: verify-pass KL (17), acceptance and
   decode speed against the plain decode.
7. **Vision.** Encoder, projector, MRoPE positions. `image-inputs` and
   `tool-images` suites.
8. **Measurement and documentation.** Benchmark card, `EXPERIMENTS.md`,
   `QUALITY.md` (KL against llama.cpp on the same GGUF).

## Serving status (2026-11-19)

`gufo serve llm` serves the model with the prompt cache (RAM and disk). Gufo's
functional suites at the pin (`0a473ebf`), one session, `--context` default,
development build:

- `conversation`, `long-context` and `cache` (disk restart included) passed.
- `tools`: `native-tools` passed; `auto-tools` fails one assertion that a
  12-token limit cuts any call short. Hy4's tags are single tokens, so
  `record(value="alpha")` completes in 11. With that limit at 8 (a local copy
  of the test) the whole suite passed. The assertion is a Gufo test
  assumption to raise upstream, not a runner defect.

Greedy decode through the server runs at 16.5 tok/s.

DFly2 (three greedy chat prompts, 166 to 237 tokens each, decode tok/s and
drafts accepted; plain decode 16.4, 16.4 and 15.7):

| drafts/step | code | prose | list |
| --- | --- | --- | --- |
| 1 | 23.4 (94%) | 19.0 (57%) | 21.8 (82%) |
| 2 | 27.1 (86%) | 18.5 (42%) | 22.6 (65%) |
| 3 | 28.6 (80%) | 17.6 (35%) | 22.4 (55%) |
| 4 | 27.0 (68%) | 15.5 (28%) | 20.7 (46%) |
| 5 | 26.6 (62%) | 13.7 (22%) | 18.4 (37%) |
| 7 | 24.8 (52%) | 11.1 (15%) | 14.2 (24%) |

At three drafts a step costs 16.5 ms of drafting and 101 ms to verify four
tokens (a plain step: 61 ms). Two and three drafts tie (geometric mean 22.4);
three is the default. The confidence head could pick the length per step.

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

## Read from the trunk GGUF

- 1,379 tensors, `general.architecture` `hy_v4_flash`. Sampling defaults are
  temperature 0.9, top-p 1.0, no top-k.
- **Engram layers** are `ple.layers` [1, 11, 21, 31] (0-based). The config's
  [2, 12, 22, 32] counts from 1. Its metadata stores the hash multipliers,
  head offsets, head vocabulary sizes (64 heads), a 120,832-entry token map and
  the pad token 120002. Per Engram layer there are `engram_kv` (Q8_0,
  1536 × 15360), `engram_kv_b`, `engram_gate`, `engram_hc_scale`,
  `engram_lookup_norm` [96, 16] and `engram_pre_gamma`.
- **Types:** Q8_0 for attention, the indexer, the shared experts, the latent
  projections, layer 0 and the token embedding. Routed gate/up are Q4_K.
  Routed down is Q6_K on 20 layers and Q4_K on 21, so the expert GEMV needs
  both. F32 for the router, norms, HC, sinks and indexer compressor; F16 for
  the head.
- **HC weights:** `hc_*_fn` [12288, 8] and `hc_*_base` [8], i.e. 4 pre plus 4
  post weights and no comb matrix, which confirms iHC. The head reduces the
  streams through `output_hc_fn/base/scale`.
- **Chat template:** the embedded template matches Tencent's except that it
  also accepts `reasoning_effort` `max`. Every special token carries the
  `:opensource` suffix (`<think:opensource>`, `<｜hy_start:opensource｜>`).

## Open questions

- What does `reasoning_effort` `max` render? It is in the Xylolabs template only.
