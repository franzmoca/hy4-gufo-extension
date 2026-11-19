# Quality

Evidence that the engine computes what the reference computes, on the
Xylolabs Q4_K_M GGUF. How each gate was chosen is in
[PLAN.md](PLAN.md#correctness-oracle).

## Text: KL against llama.cpp (2026-11-19)

Reference: llama.cpp PR #32517 (`cf70454a6`) on the same GGUF,
`llama-perplexity -c 4096 --chunks 2 --kl-divergence-base`. Two 4,096-token
chunks of Gufo documentation, scored on their second halves (positions 2,048
to 4,095, where the sparse layers' indexer selects pools).

| run | mean KL | same top token | PPL |
| --- | --- | --- | --- |
| llama.cpp, F32 cache against its F16 cache (noise floor) | 0.070 | 92.0% | 14.930 |
| this engine (`hy4_eval_probe --kl`) | 0.096 ± 0.010 | 91.4% | 15.031 |

llama.cpp's own PPL on the base run is 14.466. The model is chaotic under
rounding past 2,048 tokens (near-tied pool and expert choices), so single
positions and greedy identity are not gates there. Short prompts match
llama.cpp greedy token for token.

## Speculative decoding: exact

With DFly2 (decision 17), a verify pass computes the logits one-token
decoding would: `hy4_eval_probe --passcheck N` gives 0 difference at N = 2 to
8, on short and 2,700-token contexts. Greedy replies with and without the
drafter are identical (three prompts, ABBA arms), and draft counts match
between drafter arms.

## Vision

There is no reference implementation for this projector type in llama.cpp,
and Transformers has no `hy_v4_flash` model. The gate is a float64 numpy
transcription of Transformers' `HYV4VLVisionModel` (the same tower and merger)
run on the GGUF's weights, compared with `hy4_vision_probe` rows:

| image | patches | FP32 tower | binary16 tower (current) |
| --- | --- | --- | --- |
| synthetic 128×96 | 48 | rel. RMS 1.0e-5 | 6.2e-4 |
| synthetic 352×320 | 440 | 6.4e-6 | 1.3e-3 |
| owl logo 2048² (against the FP32 tower) | 16,384 | — | 1.1e-3, min cosine 0.99998 |

End to end, the 2048² owl logo is described correctly at full size and at
`detail: low`, and llama.cpp's matmul diagram (2875×1500) is transcribed
exactly, axis labels included. The image positions (M-RoPE) and Engram
treatment follow decision 19; their only check is these answers.

## Reproducibility

Greedy replies are reproducible for the same request history, and the release
(`nix build`) and development builds give byte-identical replies. They are
not invariant to the history: a prompt-cache restore moves the prefill pass
boundaries, and the pass width changes results from the first layer on (KL
0.006, 2% of greedy tokens flip;
[E6](EXPERIMENTS.md#e6-prefill-results-against-the-pass-split-2026-11-19)).
An image prompt served cold and after other requests can therefore word its
answer differently.

## Functional suites

Gufo's suites at the pin (`0a473ebf`) against `gufo serve` of this build,
with DFly2 and the vision sidecar:

| suite | result |
| --- | --- |
| `conversation`, `long-context`, `reasoning-budget` | passed |
| `cache` | passed but `legacy-tool-cancel-disk` `content-preserve1-sampled0` (a greedy near tie after a restore re-prefilled in other chunk shapes; passes without the drafter) |
| `tools` | passed but the 12-token `auto-tools` limit (Hy4's tags are single tokens) |
| `image-inputs` | passed |
| `tool-images` | passed but `tool_images_two_calls`: the model answers `red, blue` to "separated by a comma and no spaces" |

The harness marks drafter runs failed in its measurements ("AR request
executed speculative drafts"): it knows only Gufo's own speculative backends.
