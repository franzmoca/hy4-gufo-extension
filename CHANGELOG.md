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
- Prefill 44% faster again (360 → 517 tok/s on 3,712 tokens). The prefill
  attention stages each key tile once for 32 heads, from keys converted to
  binary16 once per pass, and runs its softmax on every thread; the query
  absorption runs on the matrix cores with exact weights; the
  hyper-connection mix reads its weights once per 8 tokens; paired
  projections quantize their input once; and each routed expert gets a tile
  width for its bucket
  ([E10](docs/EXPERIMENTS.md#e10-prefill-attention-32-heads-per-block-2026-11-19),
  [E11](docs/EXPERIMENTS.md#e11-matrix-core-query-absorption-2026-11-19),
  [E12](docs/EXPERIMENTS.md#e12-hyper-connection-mixing-8-tokens-per-block-2026-11-19),
  [E13](docs/EXPERIMENTS.md#e13-one-input-quantization-for-paired-projections-2026-11-19),
  [E14](docs/EXPERIMENTS.md#e14-prefill-attention-softmax-on-every-thread-2026-11-19),
  [E15](docs/EXPERIMENTS.md#e15-prefill-attention-keys-as-binary16-once-per-pass-2026-11-20),
  [E16](docs/EXPERIMENTS.md#e16-routed-expert-tile-width-per-expert-2026-11-20)).
- Prefill 29% faster again (517 → 665 tok/s on 3,712 tokens): the quantized
  projections and routed experts run on FP16 matrix-core kernels instead of
  the vendored MMQ, and a prompt prefilled in different passes now gives the
  same results, except for passes of 8 or fewer tokens
  ([E17](docs/EXPERIMENTS.md#e17-fp16-matrix-core-projections-and-routed-experts-instead-of-mmq-2026-11-20)).
- A prompt gives the same results however it is cut into passes, so a reply
  after a prompt-cache restore matches the reply to the same prompt served
  cold; short prompt passes are faster
  ([E18](docs/EXPERIMENTS.md#e18-prompt-passes-on-the-batched-kernels-at-any-size-2026-11-20)).
- Prefill about 800 tok/s on 3,712 tokens: the attention's output and its
  gated values pass between kernels in binary16, and the router runs on its
  own FP32 GEMM
  ([E19](docs/EXPERIMENTS.md#e19-prefill-attention-output-in-binary16-2026-11-20),
  [E20](docs/EXPERIMENTS.md#e20-fp32-gemm-for-the-router-instead-of-hipblas-2026-11-20),
  [E21](docs/EXPERIMENTS.md#e21-attention-gate-applied-in-the-value-projections-stores-binary16-input-to-wo-2026-11-20)),
  each hyper-connection post-mix runs inside the next pre-mix
  ([E23](docs/EXPERIMENTS.md#e23-hyper-connection-post-mix-folded-into-the-next-pre-mix-2026-11-20)),
  the projections read their normed inputs and the expert latent in
  binary16
  ([E24](docs/EXPERIMENTS.md#e24-binary16-activations-for-the-fp16-projections-normed-inputs-and-the-expert-latent-2026-11-20)),
  and the sparse attention's indexer scores take a thread per group
  ([E25](docs/EXPERIMENTS.md#e25-indexer-scores-with-a-thread-per-group-2026-11-20)).
- Decode no longer slows with the context in the sparse layers' top-k
  selection: 13.75 → 15.18 tok/s after a 3,712-token prompt, about 33 ms per
  token less at 32K
  ([E26](docs/EXPERIMENTS.md#e26-kpool-selection-by-radix-select-and-sort-2026-11-20)).
- DFly2 decoding 6-15% faster: a verify pass runs its routed experts in one
  launch with the tokens that share an expert back to back, with the same
  results as one-token decoding
  ([E27](docs/EXPERIMENTS.md#e27-verify-passes-routed-experts-in-one-launch-slots-ordered-by-expert-2026-11-20)).
- DFly2 decoding a further 3-6% faster: a verify pass's float and grouped
  GEMVs compute four rows per warp, with the same results as one-token
  decoding
  ([E28](docs/EXPERIMENTS.md#e28-verify-passes-four-rows-per-warp-in-the-float-and-grouped-gemvs-2026-11-20)).
- Decode no longer slows with the context in the sparse and sliding-window
  layers: attention reads only each token's own keys in one fused kernel,
  15.3 → 16.2 tok/s after a 3,712-token prompt and 13.2 → 16.1 after 14,849
  tokens; with DFly2, 15% faster after a 4,192-token prompt
  ([E29](docs/EXPERIMENTS.md#e29-fused-decode-attention-over-each-tokens-own-keys-2026-11-20)).
- The KV cache is binary16: about half the session memory per token, so the
  model's full 1,048,576-token context fits beside DFly2 and vision. Prefill
  results are unchanged; prompt-cache entries from earlier builds are not
  reused
  ([E30](docs/EXPERIMENTS.md#e30-binary16-kv-cache-2026-11-20)).
- DFly2 drafts through a Q8_0 copy of the LM head and stops before scoring
  a slot its confidence head rejects: decoding with DFly2 7% faster, greedy
  and sampled (24.3 → 26.0 and 23.6 → 25.3 tok/s), with the same replies
  ([E31](docs/EXPERIMENTS.md#e31-dfly2-drafts-through-a-q8_0-copy-of-the-lm-head-2026-11-20)).
- Copy repeated context into the verify pass (prompt lookup), with or
  without DFly2: rewriting a file decodes 16% faster with DFly2 and 2.5
  times faster without it, with the same replies
  ([E33](docs/EXPERIMENTS.md#e33-prompt-lookup-2026-11-20)); greedy requests
  verify up to 15 copied tokens per pass
  ([E34](docs/EXPERIMENTS.md#e34-16-token-verify-passes-for-copies-2026-11-20)).
- Prompt-cache checkpoints are captured inside a prefill pass instead of
  ending it, and Engram rows are read directly from the drive while the
  layers run: served prefill of new prompts 32-37% faster (653-694 → 890-910
  tok/s)
  ([E45](docs/EXPERIMENTS.md#e45-cache-checkpoints-inside-a-prefill-pass-2026-11-20),
  [E46](docs/EXPERIMENTS.md#e46-engram-rows-read-directly-beside-the-layers-2026-11-20)).
- The server holds 3.7 GiB less device memory: the prompt scratch keeps
  logits for 16 rows instead of 8,192
  ([E50](docs/EXPERIMENTS.md#e50-prompt-scratch-logits-for-16-rows-2026-11-20)).
- DFly2 drafts for concurrent requests in one drafter pass: four coding-agent
  sessions finish in 420 s instead of 465 s, with the same replies
  ([E49](docs/EXPERIMENTS.md#e49-several-sessions-drafts-in-one-drafter-pass-2026-11-20)).
- Concurrent requests decode together in one pass: four requests generate
  39 tokens per second in all (16.5 before), 35 with DFly2 (17.6), with the
  same replies; a single request is unchanged
  ([E48](docs/EXPERIMENTS.md#e48-several-sessions-decoding-in-one-pass-2026-11-20)).
- Other sessions keep decoding while a prompt prefills, between its layer
  groups
  ([E47](docs/EXPERIMENTS.md#e47-decode-rounds-between-a-prompt-passs-layer-groups-2026-11-20)).
- Prefill evaluates 8,192 tokens per pass when the context is at most 256K
  tokens: 3% faster on prompts over 4,096 tokens
  ([E43](docs/EXPERIMENTS.md#e43-8192-token-prefill-passes-for-contexts-up-to-256k-2026-11-20)).
- Faster prefill attention: 3-4% faster prefill (919 tok/s on 3,712 tokens)
  ([E41](docs/EXPERIMENTS.md#e41-prefill-attention-staging-loads-in-flight-together-paired-value-reads-2026-11-20)).
- Prefill evaluates 4,096 tokens per pass: 7% faster on 3.7K to 16K-token
  prompts (827 → 888 tok/s on 3,712 tokens)
  ([E37](docs/EXPERIMENTS.md#e37-4096-token-prefill-passes-2026-11-20)).
- A prompt longer than 2,048 tokens gives the same reply after a
  prompt-cache restore as served cold, wherever the restore splits it
  ([E36](docs/EXPERIMENTS.md#e36-prefill-attention-below-position-2048-independent-of-the-pass-2026-11-20)).
- DFly2 drafts score the LM head's 65,536 most frequent rows and the rows of
  the request's own tokens: decoding with DFly2 1.3% faster on chat prompts
  and 1.6% on an agent session, with the same replies
  ([E35](docs/EXPERIMENTS.md#e35-a-smaller-draft-head-the-frequent-rows-and-the-requests-own-2026-11-20)).
