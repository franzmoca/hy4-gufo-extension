# Changelog

## Unreleased

- Serve Tencent's Hy4 Flash (`hy_v4_flash`) from the Xylolabs Q4_K_M GGUF:
  chat, reasoning effort, tool calls and the prompt cache (RAM and disk).
- Draft with Tencent's DFly2 (`--dfly-model`), with the draft length chosen
  by its confidence head ([E2](docs/EXPERIMENTS.md#e2-draft-length-from-the-confidence-head-2026-11-19));
  replies are identical to plain decoding ([E3](docs/EXPERIMENTS.md#e3-exact-verify-passes-2026-11-19)).
- Accept image input through the `mmproj` vision tower (`--mmproj`, or a
  single `mmproj*.gguf` beside the model), with M-RoPE image positions
  ([E4](docs/EXPERIMENTS.md#e4-fused-binary16-attention-in-the-vision-tower-2026-11-19),
  [E5](docs/EXPERIMENTS.md#e5-binary16-gemm-operands-in-the-vision-tower-2026-11-19)).
