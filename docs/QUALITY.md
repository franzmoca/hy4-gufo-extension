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
| this engine (`hy4_eval_probe --kl`), first build | 0.096 ± 0.010 | 91.4% | 15.031 |
| this engine after E17 (FP16 matrix-core projections) | 0.083 ± 0.008 | 92.1% | 15.388 |
| this engine after E19 (binary16 attention output) | 0.083 ± 0.009 | 91.6% | 14.978 |

llama.cpp's own PPL on the base run is 14.466. The model is chaotic under
rounding past 2,048 tokens (near-tied pool and expert choices), so single
positions and greedy identity are not gates there. Short prompts match
llama.cpp greedy token for token.

## Text: KL against exact arithmetic (2026-11-21)

`tools/oracle/reference.py` runs Tencent's Transformers implementation
(huggingface/transformers PR #50903 at `b865850d`) in FP32 on the CPU over
the same GGUF's dequantized weights, and writes a KL base file over the same
two chunks. Against it the differences are the arithmetic alone:
quantization is common to all rows. `--kl` scores the prompt passes,
`--kl-decode` the scored half as one-token decoding passes.

| run | mean KL | same top token | PPL |
| --- | --- | --- | --- |
| exact reference | — | — | 14.491 |
| llama.cpp, F16 cache | 0.083 ± 0.008 | 91.7% | 14.466 |
| this engine, prompt passes | 0.054 ± 0.006 | 94.2% | 14.682 |
| this engine, decoding passes | 0.051 ± 0.006 | 94.6% | 14.676 |

All PPLs here count a token below the row's maximum minus 16 as that bound,
as the base files record it. The probe prints its PPL both ways: unclamped,
the engine's rows read 15.12 and 15.07. That gap is how the engine rows above
came to read about 4% above llama.cpp. On a 3,072-token prefix scored with
full logits on both sides, the engine's next-token NLL is within 0.004 ±
0.013 of the reference's in every 1,024-token range, and its entropy matches.
Per-layer hyper-connection states (`hy4_eval_probe --hidden`, `reference.py
--hidden`) drift smoothly with depth, to about 10% relative at the last
layer, with no step at the sparse layers' selection past 2,048 tokens.

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
boundaries. Since
[E18](EXPERIMENTS.md#e18-prompt-passes-on-the-batched-kernels-at-any-size-2026-11-20)
a prompt gives identical logits however it is split into passes
(`--chunkcheck` at tails of 800 down to 1 token, and since
[E36](EXPERIMENTS.md#e36-prefill-attention-below-position-2048-independent-of-the-pass-2026-11-20)
at splits below 2,048 in longer prompts), so a reply after a restore matches
the same request served cold. Before E17 every split changed results
from the first layer on (KL 0.006, 2% of greedy tokens flipped;
[E6](EXPERIMENTS.md#e6-prefill-results-against-the-pass-split-2026-11-19)).

## Functional suites

Gufo's suites at the pin (`0a473ebf`) against `gufo serve` of this build,
with DFly2 and the vision sidecar:

| suite | result |
| --- | --- |
| `conversation`, `long-context`, `reasoning-budget` | passed |
| `cache` | passed (since E18; before it, `legacy-tool-cancel-disk` `content-preserve1-sampled0` failed: a greedy near tie after a restore re-prefilled in other pass shapes) |
| `tools` | passed but the 12-token `auto-tools` limit (Hy4's tags are single tokens) |
| `image-inputs` | passed |
| `batch`, `prefill-scheduling` | passed (2026-11-20, after E48-E51) |
| `tool-images` | passed but `tool_images_two_calls`: the model answers `red, blue` to "separated by a comma and no spaces" |

Release 0.1.0 (2026-11-20), the `nix build` of this extension at Gufo
`0a473ebf`, `--context 32768`: `conversation`, `long-context`, `cache`,
`reasoning-budget`, `image-inputs`, `batch` and `prefill-scheduling` passed;
`tools` and `tool-images` passed but the two cases above.

After the prefill kernel work (E10-E16, 2026-11-20) and again after E17,
`conversation`, `long-context` and `cache` gave the same results: all passed
but the same `legacy-tool-cancel-disk` case. After E18 and E19 every case of
the three suites passes.

The harness marks drafter runs failed in its measurements ("AR request
executed speculative drafts"): it knows only Gufo's own speculative backends.
