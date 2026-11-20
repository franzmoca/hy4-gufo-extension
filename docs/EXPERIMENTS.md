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
| E12 | Hyper-connection mixing for 8 tokens per block | retained |
| E13 | One input quantization for paired projections | retained |
| E14 | Prefill attention softmax on every thread | retained |
| E15 | Prefill attention keys converted to binary16 once per pass | retained |
| E16 | Routed expert tile width per expert | retained |
| E17 | FP16 matrix-core projections and routed experts instead of MMQ | retained |
| E18 | Prompt passes on the batched kernels at any size | retained |
| E19 | Prefill attention output in binary16 | retained |
| E20 | FP32 GEMM for the router instead of hipBLAS | retained |
| E21 | Attention gate applied in the value projection's stores, binary16 input to `wo` | retained |
| E22 | `MlaPrefill` output staged through LDS for 16-byte stores | rejected |
| E23 | Hyper-connection post-mix folded into the next pre-mix | retained |
| E24 | Binary16 activations for the FP16 projections: normed inputs and the expert latent | retained |
| E25 | Indexer scores with a thread per group | retained |

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

## E12: Hyper-connection mixing, 8 tokens per block (2026-11-19)

The hyper-connection mix (four streams of 4,096 against eight mixing rows,
plus the sum of squares) ran one token per block, so every token read all of
the 512 KB mixing weights again: 0.48 s of the prefill. Passes of 8 tokens or
more now mix 8 tokens per block, each weight value loaded once for all 8,
with the same per-token thread partition, products and reduction order, so
the results are bit-identical (checked in `kernels_test`; KL and PPL against
llama.cpp unchanged to the last digit). Smaller passes keep one token per
block. 3,712-token prefill: 8.24 → 7.95 s (451 → 467 tok/s). Retained.

## E13: One input quantization for paired projections (2026-11-19)

Batched projections of one input and one weight type now go through the MMQ
pair entry, which quantizes the input once: `wq_a` with `wkv_a`, the dense
layer's gate with its up, and the shared expert's gate with its up. Same
kernels on the same quantized input, so results are bit-identical (KL and
PPL unchanged to the last digit). 3,712-token prefill: 7.95 → 7.90 s (both
runs). Retained as a small measured gain.

## E14: Prefill attention softmax on every thread (2026-11-19)

In `hy4_mla_prefill` the online softmax of each 32-key tile ran on 32 of the
block's 256 threads, one per head walking all 32 keys, while the other
threads waited at the barrier. Eight lanes per head now take four keys each
and combine their maximum and sum with shuffles. 3,712-token prefill:
7.90 → 7.49 s (both runs; 470 → 495 tok/s). The denominators sum in another
order: KL against llama.cpp 0.084 ± 0.008, same top 91.5% (E13: 0.081,
91.3%). Retained.

## E15: Prefill attention keys as binary16, once per pass (2026-11-20)

A standalone benchmark of `hy4_mla_prefill` at the largest shapes (2,048
tokens; a window layer of 64 heads over 512 keys, a sparse layer of 32 heads
over 2,048 selected keys), removing one phase at a time:

| variant | window layer | sparse layer |
| --- | --- | --- |
| E14 | 10.07 ms | 23.95 ms |
| no key staging | 5.25 ms | 11.21 ms |
| no Q.K products | 8.47 ms | 21.23 ms |
| no P.V products | 8.15 ms | 19.69 ms |

Half the time went to staging: every block read FP32 key rows (2.3 KB each)
and converted them. The pass's visible key rows are now converted to
binary16 once (`hy4_gather_rows_half`, which also replaces the window
gather) and the kernel stages them with 16-byte copies: 8.19 and 17.41 ms.
The rounding is the one the kernel applied, so results are bit-identical to
E14 (KL 0.08438, PPL 15.1116). 3,712-token prefill: 7.49 → 7.29 s (both
runs). Each session holds a binary16 buffer of its context's key rows
(37.7 MB at 32K); the FP32 window buffer shrinks to the verify passes' rows.

Not retained, from the same benchmark:

- 16-key tiles (half the LDS, so two blocks per CU): 7.60 and 16.28 ms, but
  7.29 / 7.23 s end to end against 7.29 / 7.29 s, inconclusive, and the
  softmax groups change the results.
- Loading the next tile into registers during the products: the queries
  already hold 72 of the 256 registers, and every prefetch depth (2 to 5 of
  the 5 groups per thread) spilled and ran slower (8.4 to 8.9 ms, 22 to
  23 ms).

## E16: Routed expert tile width per expert (2026-11-20)

The routed experts (Q4_K and Q6_K MMQ, 2.6 s of the prefill) took one
column-tile width per launch from a cost model fitted on DS41's experts.
Forcing one width for every layer (3,712-token prefill): 16 → 7.79 s,
32 → 7.25 s, 48 → 7.38 s, 64 → 7.41 s, 80 → 7.53 s, the model's choice
7.29 s; wider tiles, which re-read an expert's weights less often, are
slower, so these kernels are not bound by weight reloads. The vendored
per-expert classes (16, 32 or 64 columns by bucket size, from DS41's fit),
in ABAB order: 7.18, 7.22 s against 7.29, 7.28 s; as the default, 7.19 and
7.18 s. Tile widths change the accumulation split: KL against llama.cpp
0.084 ± 0.008, same top 91.5%. Rejected on the way: a register-held query
variant of `IndexerScore` (bit-identical, but 0.21 s against 0.18 s: the
warp sums, not the query reads, bound it). Retained.

## E17: FP16 matrix-core projections and routed experts instead of MMQ (2026-11-20)

Every batched Q8_0, Q4_K and Q6_K projection left the vendored MMQ (and its
Q8_1 activation quantization) for `kernels/rocm/hy4_f16_mm.hip.cpp`: each
weight decodes to FP16 with its sub-block scale folded in, the activations
round to FP16 as they are staged, and the matrix-core products accumulate in
FP32. The dense kernel is DS41's (`ds41_dense_f16.hip.cpp`: 256 rows x 128
tokens per workgroup, a half-wave decoding each 8-value group and exchanging
it with `permlanex16`) with a Q4_K format and rows past the last 256-row
block guarded (`attn_kv_a_mqa` has 576). DS41's routed kernel loads whole
blocks per lane, which suits its 50-82-byte IQ blocks but not 144-byte Q4_K
and 210-byte Q6_K; the routed kernel here reuses the dense K loop on one
expert's 128-slot tile, gathering the slots' activation rows, with waves 0-7
on gate rows and 8-15 on the same up rows, the up sums passed through LDS
and SwiGLU with the clamp and route weight stored as FP16 for the down pass.
With 64-slot tiles (four 16-slot products) the compiler hoisted every
fragment load to 256 VGPRs and 9-83 spills; at 128 slots it holds 166-190
VGPRs without spills, and 16-slot tiles past a bucket are skipped, so the
wider tile pads no more.

Standalone at Hy4 shapes (2,048 tokens, random weights; MMQ including its
quantization): dense 1.06-1.61x (`attn_gate` 8.05 → 6.64 ms, `wo` 9.07 →
7.63 ms, shared gate+up 1.68 → 1.52 ms), the grouped `wv_b` equal (5.16 /
5.24 ms); routed per layer with a skewed expert distribution, Q4_K gate/up
20.3 → 10.9 ms (15 → 28 TFLOPS), Q4_K down 11.4 → 5.4 ms, Q6_K down 15.4 →
6.7 ms. FP16 and MMQ outputs differ by 5.4e-3 (dense) and 1.5e-2 (routed,
relative L2), MMQ's Q8_1 rounding; `hy4flash.kernels` checks the new kernels
against the ggml dequantization in double precision.

3,712-token prefill 7.18 → 5.58 s (517 → 665 tok/s; two runs, both 5.58 s).
Kernels 7.05 → 5.35 s: routed 2.57 → 1.36 s, dense 1.72 s plus 0.49 s of
quantization → 1.87 s. KL against llama.cpp 0.0834 ± 0.0079, same top
92.1% (E15: 0.0844, 91.5%); PPL 15.388 against E15's 15.112 (llama.cpp's
noise-floor run 14.930), a shift inside what two chaotic chunks move by
when KL is unchanged; `--passcheck 8` exact. Batched passes are now
independent of the pass split: `--chunkcheck` at 200 and 517 gives identical
logits (was KL 0.0055 from layer 0, E6), since a row's sums no longer depend
on its place in the batch. A tail of 8 or fewer tokens still takes the
single-token MMQ path and differs (`--chunkcheck 995`: KL 0.0028). Retained.

## E18: Prompt passes on the batched kernels at any size (2026-11-20)

After E17 only passes of 8 or fewer tokens still depended on the pass split:
they took the single-token kernels (MMQ matvecs, per-token hipBLAS
attention), which verify passes need so that they compute what one-token
decoding does. Prompt passes have no such constraint, so `Session::Prompt`
(the runner's prefill, and `Evaluate(..., prompt = true)`) now takes the
batched kernels at any size, and only decoding and verify passes take the
single-token ones. One more kernel depended on the row count: hipBLAS runs a
one-row FP32 GEMM (the router, the indexer weights) with another kernel, so
prompt passes under 16 rows run it as 16 rows of the chunk scratch.
`--chunkcheck` on 1,000 tokens at 200, 517, 995, 996, 998 and 999 (tails of
800 down to 1 token): identical logits at every split, against KL 0.0028 at
995 with E17. `hy4flash.kernels` checks that the FP16 kernels give a token's
rows the same bits in a 5-token and a 150-token pass. The `cache` suite now
passes `legacy-tool-cancel-disk`, which a restore re-prefilling a gap in
other pass shapes had failed since the first build. A 5-token prompt pass
takes 0.142 s instead of 0.205 s (the single-token attention ran one token
at a time).

Two changes to the dense kernel's stores came with it. Ablation on the
grouped `wv_b` (64 groups of 256 rows, k = 512; 5.15 ms) found the stores
half its time (2.85 ms without them) and FP16 input 0.9 ms. Each 16-token
tile now goes through LDS and leaves as contiguous 16-byte stores of a
token's rows (5.24 → 4.96 ms; `wo` 7.63 → 7.48 ms), and the group index is
the fastest grid index, so the workgroups in flight read and write
neighbouring parts of the same token rows (4.96 → 4.13 ms). Same values.
3,712-token prefill 5.58 → 5.52, 5.53 s. KL 0.0834 ± 0.0079 and
`--passcheck 8` unchanged. Retained.

## E19: Prefill attention output in binary16 (2026-11-20)

`MlaPrefill` wrote its FP32 output (n x heads x 512) only for the `wv_b`
projection, which rounds its input to binary16 as it stages it. It now
stores binary16 and `wv_b` reads it as such, halving that write and read
(268 MB per 2,048-token pass of a 64-head layer). Not bit-identical: the
compiler turns the store's `(_Float16)(acc * inv)` into `v_fma_mix`, which
rounds the exact product to binary16 once, where the FP32 store and the
staging rounded it twice; the staging itself matches a pre-rounded input bit
for bit. KL against llama.cpp 0.0828 ± 0.0085, same top 91.6%, PPL 14.978
(was 0.0834 ± 0.0079, 92.1%, 15.388: within the run's error);
`--passcheck 8` and `--chunkcheck` at 517 and 999 exact; the
`conversation`, `long-context` and `cache` suites pass every case.
3,712-token prefill 5.52 → 5.43 s (684 tok/s; both runs). Retained.

## E20: FP32 GEMM for the router instead of hipBLAS (2026-11-20)

The batched FP32 projections (the router, 512 x 3,072, and the indexer's
small FP32 matrices) ran on hipBLAS, which picked a 32 x 32 macro-tile kernel
for this layout: 2.6 ms per 2,048-token router pass (2.5 TFLOPS, 0.2 s of
the 3,712-token prefill). `hy4_sgemm_nt` takes 64 tokens x 128 rows per
block, 4 x 8 outputs per thread, K in steps of 32 through LDS transposed for
float4 reads, with the next step's global loads issued before the current
step's products: 2.28 ms without that prefetch, 1.05 ms with it (6.1 TFLOPS).
Products and sums stay FP32; each output sums its k products in order, so a
token's row does not depend on the pass (`hy4flash.kernels` checks 5 against
70 tokens bit for bit), and E18's 16-row padding of small prompt passes is
gone. The sum order differs from hipBLAS's: KL 0.0842 ± 0.0081, same top
91.9%, PPL 15.117 (E19: 0.0828 ± 0.0085); `--passcheck 8` and `--chunkcheck`
at 517 and 999 exact. 3,712-token prefill 5.43 → 5.355 s (693 tok/s; both
runs). Retained.

## E21: Attention gate applied in the value projection's stores, binary16 input to `wo` (2026-11-20)

The batched attention tail wrote `wv_b`'s output in FP32, multiplied it by
sigmoid(gate) in a separate pass (`SigmoidMul`, 0.12 s), and `wo` read it
back in FP32 to round it to binary16 as it staged it. The gate projection
only reads the layer input, so it now runs before `wv_b`, whose stores
(`hy4_f16_dense_gated`) multiply by sigmoid(gate) and store binary16, which
`wo` reads as such: one pass less, and the 134 MB `attv` write and read per
2,048-token pass of a 64-head layer halve. Bit-identical to E20 (KL 0.08424,
PPL 15.1167 to the last digit: the FP32 gated value is rounded to binary16
exactly as `wo`'s staging rounded it); `--passcheck 8` and `--chunkcheck` at
517 and 999 exact. 3,712-token prefill 5.355 → 5.18, 5.14 s (about
720 tok/s). Retained.

## E22: `MlaPrefill` output staged through LDS for 16-byte stores (2026-11-20)

`MlaPrefill` stores its output as 2-byte scalars (16 lanes write 32
contiguous bytes per head). Ablation in a micro-benchmark (2,048 tokens,
the 64-head window layer and the 32-head sparse layer): skipping the stores
saves 0.44 of 7.58 ms and 0.1 of 17.4 ms, the bound for any store change.
Staging the 32 heads' outputs in the then-free key tile and storing the
block's contiguous 32 KB in 16-byte groups made the window layer slower,
7.71–7.95 ms, and 7.67–7.69 ms with a pitch that removes the halves' bank
conflict: the direct stores already overlap other blocks' work, the staged
ones add a barrier and an LDS round trip. Rejected.

## E23: Hyper-connection post-mix folded into the next pre-mix (2026-11-20)

Each layer half ended with `HcPost` (h = h + y * post over the four
streams) and the next half began with `HcMixTokens`, which reads the same
streams again. `hy4_hc_post_pre` does the post-mix inside the pre-mix's
pass: each value is updated with the same expression, stored back and
mixed. The forward pass defers a half's post-mix to the next pre-mix (the
FFN's, the next layer's attention, the head) unless the Engram or the
probe reads the streams in between. Passes under 8 tokens keep the two
kernels. Bit-identical (KL 0.08424, PPL 15.1167; `--passcheck 8`,
`--chunkcheck` 517 and 999 exact; the unit test compares both paths
bitwise). 3,712-token prefill 5.16 → 5.07 s (732 tok/s). Retained.

## E24: Binary16 activations for the FP16 projections: normed inputs and the expert latent (2026-11-20)

The FP16 kernels staged FP32 activations, rounding them to binary16 as they
went, and every row block of a projection rereads its token tile's
activations. Ablated at 2,048 tokens in a micro-benchmark, skipping the
activation loads took `attn_gate` (16,384 x 3,072) from 6.14 to 5.09 ms and
`q_b` (16,384 x 1,024) from 2.29 to 1.73 ms. The routed gate/up kernel reads
each slot's gathered 1,536-value latent row once per row block: about
2.4 GB of FP32 reads per pass, three times its weight bytes.

In batched passes the RMS norms of `cur` and of the query latent now also
write a binary16 copy (the FP32 result rounded once: a compiler barrier
keeps the weight product from fusing into the conversion), which every FP16
projection reading them takes instead (`wq_a`/`wkv_a`, `wq_b`, `attn_gate`,
the shared expert, the indexer's projections, the latent projection). The
latent projection stores binary16, the routed gate/up kernel's input. The
router keeps its FP32 input. Bit-identical (KL 0.08424, PPL 15.1167;
`--passcheck 8`, `--chunkcheck` 517 and 999 exact; a unit test compares the
copy with the rounded FP32 output bitwise). 3,712-token prefill 5.07 → 5.02 s
with the binary16 latent alone, 4.85-4.87 s with both (763 tok/s).
Retained.

## E25: Indexer scores with a thread per group (2026-11-20)

`IndexerScore` took a warp per (token, group) and summed each of the 16
heads' 128-value dot products across the warp: four products per lane, then
five shuffle steps. That was 14.4 ms per sparse layer pass of 2,048 tokens
(0.17 s of the prefill) for 8 GFLOP. A thread per (token, group) now keeps
its group's row in registers, reads the queries of its block's four tokens
from LDS as broadcasts, and sums each head as the warp did: lane l's
partial over values l + 32 j, then the xor tree's pairs (addition commutes,
so every lane of the tree held the same sum). The scores are bitwise
identical (1.9 M scores in a standalone comparison; KL 0.08424, PPL
15.1167, `--passcheck 8` and `--chunkcheck` 517 and 999 exact). Standalone:
2,048 tokens over 928 groups 12.8 → 1.7 ms; one-token decode 10 → 9 µs at
250 groups and 57 → 18 µs at 8,192 (32K context). With few blocks of groups
a block takes one token instead of four, which keeps 3- to 8-token verify
passes at short contexts no slower. 3,712-token prefill 4.86 → 4.70, 4.72 s
(788 tok/s). Retained.
