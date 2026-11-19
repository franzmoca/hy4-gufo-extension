# Changelog

## Unreleased

- Serve Tencent's Hy4 Flash (`hy_v4_flash`) from the Xylolabs Q4_K_M GGUF:
  chat, reasoning effort, tool calls and the prompt cache (RAM and disk). The
  prompt matches the GGUF's chat template, which leaves out tools marked
  `defer_loading`.
- Draft with Tencent's DFly2 (`--dfly-model`), with the draft length chosen
  by its confidence head ([E2](docs/EXPERIMENTS.md#e2-draft-length-from-the-confidence-head-2026-11-19));
  replies are identical to plain decoding ([E3](docs/EXPERIMENTS.md#e3-exact-verify-passes-2026-11-19)).
- Accept image input through the `mmproj` vision tower (`--mmproj`, or a
  single `mmproj*.gguf` beside the model), with M-RoPE image positions
  ([E4](docs/EXPERIMENTS.md#e4-fused-binary16-attention-in-the-vision-tower-2026-11-19),
  [E5](docs/EXPERIMENTS.md#e5-binary16-gemm-operands-in-the-vision-tower-2026-11-19)).
- Prefill 23% faster (206 → 253 tok/s on 3,712 tokens) with fused
  matrix-core attention, and closer to llama.cpp (KL 0.096 → 0.081)
  ([E7](docs/EXPERIMENTS.md#e7-fused-matrix-core-prefill-attention-2026-11-19)).
- Prefill in 2,048-token passes: 360 tok/s on 3,712 tokens, and the first
  token of a 2048² image in 17.6 s instead of 25.3 s. Prompt-cache snapshots
  keep only the ring rows a continuation reads (about 35 MB instead of 56 MB);
  older disk caches are not reused
  ([E8](docs/EXPERIMENTS.md#e8-wider-prefill-passes-snapshots-of-live-ring-rows-2026-11-19)).
