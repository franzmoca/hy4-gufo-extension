# Experiments

One section per experiment, retained or rejected. Measurements are on Strix
Halo `gfx1151` with the Xylolabs Q4_K_M GGUF, through `gufo serve` unless
noted. Design decisions are numbered in [PLAN.md](PLAN.md).

| id | experiment | outcome |
| --- | --- | --- |
| E1 | DFly2 at fixed draft lengths | superseded by E2 |
| E2 | Draft length from DFly2's confidence head | retained (threshold 0.45) |
| E3 | Verify passes equal to one-token decoding | retained |
| E4 | Fused binary16 attention in the vision tower | retained |
| E5 | Binary16 GEMM operands in the vision tower | retained |
| E6 | Prefill results against the pass split | measurement |
| E7 | Fused matrix-core prefill attention | retained |
| E8 | 2,048-token prefill passes, snapshots of live ring rows only | retained |
| E9 | Prefill attention: 16-key tiles with transposed values | rejected |
| E10 | Prefill attention: 32 heads per block | retained |
| E11 | Matrix-core query absorption writing binary16 queries | retained |

## E1: DFly2 at fixed draft lengths (2026-11-19)

Greedy decode tok/s and acceptance on three chat prompts (code, prose, list;
166 to 237 tokens each). Plain decode: 16.4, 16.4 and 15.7.

| drafts/step | code | prose | list |
| --- | --- | --- | --- |
| 1 | 23.4 (94%) | 19.0 (57%) | 21.8 (82%) |
| 2 | 27.1 (86%) | 18.5 (42%) | 22.6 (65%) |
| 3 | 28.6 (80%) | 17.6 (35%) | 22.4 (55%) |
| 4 | 27.0 (68%) | 15.5 (28%) | 20.7 (46%) |
| 5 | 26.6 (62%) | 13.7 (22%) | 18.4 (37%) |
| 7 | 24.8 (52%) | 11.1 (15%) | 14.2 (24%) |

At three drafts a step costs 16.5 ms of drafting and 101 ms to verify four
tokens (a plain step: 61 ms). Long fixed blocks lose on prose, where
acceptance falls fastest.

## E2: Draft length from the confidence head (2026-11-19)

DFly2's head predicts each slot's accept rate; drafting stops once the
product over the slots falls under a threshold (cap 7). Same prompts:

| threshold | code | prose | list | geo mean | drafted/step |
| --- | --- | --- | --- | --- | --- |
| 0.15 | 27.1 | 17.1 | 20.3 | 21.1 | 3.9 |
| 0.30 | 26.5 | 18.2 | 22.6 | 22.2 | 3.1 |
| 0.45 | 28.5 | 19.0 | 22.9 | 23.2 | 2.4 |
| 0.60 | 30.8 | 17.8 | 21.7 | 22.8 | 1.6 |

Retained at 0.45 (decision 18). One run per arm: its 4% over fixed three
drafts (geometric mean 22.4) is inside run-to-run noise, but it avoids the
prose losses of long blocks without a per-workload setting.

## E3: Exact verify passes (2026-11-19)

A verify pass (anchor plus drafts, up to 8 tokens) differed from one-token
decoding by mean KL 0.0018, so greedy requests could diverge after a near tie
depending on how their tokens were grouped, which Gufo's `reasoning-budget`
suite caught. Bisection by layer found two sources: the vendored multi-token
MoE kernel sums in another order than the one-token kernel, and hipBLAS picks
attention kernels by shape. Passes up to 8 tokens now launch the routed gate
and up projections per token and run each token's attention GEMMs over
exactly its own keys. `hy4_eval_probe --passcheck N`: 0 difference at N = 2
to 8. Verify cost 101 → 96 ms at 2.08 drafts per step; replies are identical
with and without the drafter. Retained (decision 17).

## E4: Fused binary16 attention in the vision tower (2026-11-19)

The first tower ran attention as strided SGEMMs over the 72-wide heads, with
a materialized score matrix and a separate softmax: at 2048² (16,384
patches) the encode took 37.5 s, of which the attention GEMMs were 31.7 s and
the softmax 5.8 s. Gemma 4's fused WMMA attention (the same head width, an
online softmax, binary16 operands with FP32 accumulation) with the scale
folded into Q: 7.4 s, 6e-4 relative RMS from the float64 reference.
Retained.

## E5: Binary16 GEMM operands in the vision tower (2026-11-19)

After E4 the FP32 projections took 5.0 s (about 2.7 TFLOPS). BF16 weights
converted to binary16 at load and activations converted per GEMM, through
hipBLAS `GemmEx` with FP32 accumulation: 7.4 → 4.8 s, 1.1e-3 relative RMS from
the FP32 rows, and the 2048² request's first token 58 → 25 s with the same
answers. hipBLAS still picks a non-WMMA kernel (2.4 s of the 4.8 s). Retained.

## E6: Prefill results against the pass split (2026-11-19)

`hy4_eval_probe --chunkcheck S` prefills a prompt whole (256-token passes
from token 0) and split at token S, as after a prompt-cache restore of S
tokens, and compares the two. First 1,000 tokens of the 3,712-token
document:

| split | token S differs from | same top after S | mean KL |
| --- | --- | --- | --- |
| 12 | layer 0 (7.6e-8) | 969 / 988 | 0.0063 |
| 200 | layer 0 (7.5e-8) | 782 / 800 | 0.0061 |
| 300 | layer 0 (1.4e-6) | 684 / 700 | 0.0060 |

The split changes results from the first layer on, so attention (whose GEMMs
hipBLAS picks by shape) is not the only source: layer 0's projections and
experts also see a different pass width. About 2% of the following greedy
tokens flip. This is why an image request served cold can word its reply
differently from the same request after unrelated text requests (whose
cached system prefix moves the pass boundaries): both replies describe the
image correctly, and every reply is reproducible for the same request
history. The release build gives byte-identical replies to the development
build for the same history.

## E7: Fused matrix-core prefill attention (2026-11-19)

Prefill passes (more than 8 tokens) computed attention as FP32 SGEMMs of every
row of the pass against every key up to the pass end, then masked: 6.8 s of an
18.0 s 3,712-token prefill. `hy4_mla_prefill` replaces them with one kernel:
block (token, 16 heads), four waves; each token walks only its own causal,
window or selected keys in 32-key tiles staged as binary16; Q.K on WMMA split
by dimension across the waves, FP32 online softmax seeded by the sink, P.V on
WMMA split by output dimension. Verify passes (8 tokens or fewer) keep their
per-token GEMMs, so E3's exactness holds (`--passcheck 4`: 0 difference).

| 3,712-token prefill (`hy4_eval_probe`) | run 1 | run 2 |
| --- | --- | --- |
| SGEMM attention | 18.00 s | 18.03 s |
| fused attention | 14.67 s, 14.66 s | 14.67 s |

206 → 253 tok/s (+23%); the kernel takes 2.7 s. KL against llama.cpp
improves from 0.096 ± 0.010 to 0.081 ± 0.007 (same top 91.4% → 91.9%;
llama.cpp's own floor 0.070 and 92.0%). `--chunkcheck 12`: layer 0 no longer
depends on the split (the first difference moves to layer 1), mean KL after
the split 0.0063 → 0.0049. `kernels_test` checks the kernel against the
double precision reference at Hy4's shape (causal, gathered window, indexer
selections). Retained.

## E8: Wider prefill passes, snapshots of live ring rows (2026-11-19)

With attention per token (E7), a wider pass no longer costs attention work,
and each pass reads the routed experts once. 3,712-token prefill
(`hy4_eval_probe`, two runs each):

| pass | prefill | tok/s |
| --- | --- | --- |
| 256 | 14.67 s, 14.66 s | 253 |
| 512 | 12.61 s, 12.62 s | 294 |
| 1,024 | 11.10 s, 11.09 s | 335 |
| 2,048 | 10.32 s, 10.32 s | 360 |

Retained at 2,048. Through `gufo serve` (DFly2 and vision loaded): the
3,747-token document prefills at 352 tok/s (was 202) and the 2048² image's
first token arrives in 17.6 s (was 25.3 s), with correct answers. GPU memory
at load: 107,263 MiB, 3.1 GiB more than with 256-token passes (the window
layers' rings hold `window + pass` rows); the verify-only selection mask now
takes 8 rows instead of a pass.

The prompt-cache snapshot used to copy each window layer's whole ring and the
indexer ring, which would have grown with the pass (about 56 MB per snapshot
at 256, 200 MB at 2,048). It now copies only what a continuation reads: each
window layer's last `window` rows before the frontier and the open kpool
group's indexer rows (about 35 MB, independent of the pass). State ABI v3.
Gates: KL 0.086 ± 0.008 (same top 91.9%), `--passcheck 8` exact, `cache` and
`long-context` suites pass apart from the case recorded in QUALITY.md, every
disk and RAM restore included.

## E9: Prefill attention with transposed values (2026-11-19)

E7's P.V gathers each value fragment strided from the key tile. Staging the
tile a second time transposed (`[512][16 + 8]` binary16, 16-key tiles to fit
LDS) makes the fragments contiguous loads, but doubles the barriers per key
and the transposed stores conflict: 3,712-token prefill 10.32 → 10.95 s and
11.00 s. Rejected.

## E10: Prefill attention, 32 heads per block (2026-11-19)

With 16 heads per block, every key tile was read from memory and converted
to binary16 twice for the 32-head layers and four times for the 64-head
window layers. Blocks of 32 heads (eight waves, two groups of four sharing
the staged tile; 56 KB LDS) halve that. Each head's arithmetic is unchanged.
3,712-token prefill (`hy4_eval_probe`): 10.32 → 9.00 s and 9.02 s
(360 → 412 tok/s). KL against llama.cpp 0.0855 ± 0.008, same top 91.89%, as
in E8; `kernels_test` passes. Retained.

## E11: Matrix-core query absorption (2026-11-19)

Prefill passes absorbed `wk_b` into the queries with a hipBLAS FP32 strided
batched GEMM on a dequantized FP32 copy (0.76 s of a 9.0 s prefill), wrote
FP32 queries, then copied their rope tails; the attention rounded them to
binary16 on load. `hy4_mla_absorb` does it in one kernel on the matrix cores
and writes the binary16 [576] rows the attention reads. `wk_b` is split at
load into its int8 values as binary16 (exact) and its block scales; a 16-wide
k step lies inside one Q8_0 block, so each block's two steps accumulate apart
and take the row's scale, and the weights stay exact. Each output sums in an
order independent of the pass width.

Per call at the largest shape (2,048 tokens, 64 heads), in a standalone
benchmark (a device copy of the same bytes takes 1.3 ms):

| variant | ms |
| --- | --- |
| hipBLAS FP32 (from the profile) | 5.5 |
| 16 tokens per block, Q8_0 blocks read in place, FP32 out | 6.0 |
| split weights, 64 tokens per block, binary16 rows through LDS | 4.9 |
| same, heads fastest across the grid | 1.8 |

Without the matrix products the first kernels took as long, so they were
bound by memory access, not arithmetic: the blocks in flight walked one
head's rows 64 KB apart. Ordering the grid by head first halved the time.
3,712-token prefill: 9.00 → 8.24 s (412 → 451 tok/s). KL against llama.cpp
0.081 ± 0.008, same top 91.3% (E10: 0.086, 91.9%; within the interval).
`--passcheck 8` exact. The FP32 `wk_b` copy is replaced by one about half its
size. `--chunkcheck 200` on the first 1,000 tokens (whole is now one pass):
first difference in layer 0 (6e-8), 789 / 800 same top, mean KL 0.0055, so
the other batched kernels still depend on the pass width. Retained.
