# Benchmarks

Hy4 Flash on Strix Halo `gfx1151` (128 GB), Xylolabs Q4_K_M GGUF (91 GiB
resident), BF16 `mmproj` vision sidecar, through `gufo serve` at Gufo
`0a473ebf` with this extension at `cedd8b3`. Context 32,768. Greedy,
`reasoning_effort: none`. `tuned` profile `accelerator-performance`, clocks
not pinned.

Four arms, a fresh server each, in ABBA order: DFly2 drafter (A) and plain
decoding (B). Every reply is identical across all four arms, and the two
drafter arms drafted and accepted the same tokens. Numbers are the mean of
the two arms of each kind; the arms agree within 3%. Script and per-arm
numbers: [artifacts/2026-11-19-benchmark/](artifacts/2026-11-19-benchmark/).

## Decode (tok/s)

| prompt | tokens | plain | DFly2 | accepted / drafted |
| --- | --- | --- | --- | --- |
| code (Fibonacci function) | 162 | 16.6 | **22.4** | 115 / 189 |
| prose (why the sky is blue) | 228 | 16.6 | **17.0** | 90 / 179 |
| list (first 20 primes) | 189 | 16.6 | **21.9** | 117 / 168 |

The llama.cpp reference implementation decodes this GGUF at 14.4 tok/s
([PLAN.md](PLAN.md#correctness-oracle)).

## Prefill and first token

These rows predate the fused prefill attention and the 2,048-token passes
([E7](EXPERIMENTS.md#e7-fused-matrix-core-prefill-attention-2026-11-19),
[E8](EXPERIMENTS.md#e8-wider-prefill-passes-snapshots-of-live-ring-rows-2026-11-19)):
the 3,747-token document now prefills at about 352 tok/s and the 2048² image's
first token arrives in about 17.6 s (one run each). Decode is unchanged.

| request | prompt tokens | prefill tok/s | first token |
| --- | --- | --- | --- |
| short chat | 35-42 | 84-105 | 0.38-0.50 s |
| 3,747-token document | 3,747 | 202 | 18.4 s |
| image, `detail: low` (512² cap) | 305 | 201 | 1.5 s |
| image, 2048² full size | 4,193 | 169 | 25.3 s |

The full-size image's first token includes 4.8 s of vision encoding
([E5](EXPERIMENTS.md#e5-binary16-gemm-operands-in-the-vision-tower-2026-11-19)).
Prefill evaluates the text in exact 256-token chunks with the batched path.
After the 2048² image, the description decodes at 13.7 tok/s plain and
14.3 tok/s with DFly2 (34 of 77 drafts accepted); after the low-detail image,
16.6 and 17.1.
